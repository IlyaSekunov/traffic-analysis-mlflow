"""
Salary prediction model — Fully Connected Network (FCN).

Key design choices for tabular salary regression
-------------------------------------------------
* LayerNorm instead of BatchNorm — stable with any batch size.
* GELU activation — empirically better than ReLU on tabular tasks.
* log1p target transform — salary distributions are right-skewed; log makes
  them roughly normal, which greatly helps gradient-based optimisers.
* Huber loss — robust to outlier salaries (common in hh.ru data).
* AdamW + cosine-annealing scheduler — solid default for tabular FCNs.
* Early stopping on validation loss with best-weights restoration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BATCH_SIZE,
    DROPOUT_RATE,
    EARLY_STOPPING_PATIENCE,
    HIDDEN_LAYERS,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_PATH,
    RANDOM_STATE,
    RESOURCES_DIR,
    SCALER_PATH,
    TEST_SIZE,
    TRAIN_X_FILE,
    TRAIN_Y_FILE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Network definition
# ---------------------------------------------------------------------------


class FCNRegressor(nn.Module):
    """Fully connected feed-forward network for regression.

    Architecture per hidden layer:
        Linear -> LayerNorm -> GELU -> Dropout

    The final layer is a single linear unit (no activation).

    Parameters
    ----------
    input_dim:
        Number of input features.
    hidden_layers:
        Sizes of the hidden layers.
    dropout_rate:
        Dropout probability after each hidden activation.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_layers: List[int] = HIDDEN_LAYERS,
            dropout_rate: float = DROPOUT_RATE,
    ) -> None:
        super().__init__()

        dims = [input_dim] + hidden_layers
        blocks: List[nn.Module] = []

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            blocks += [
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),  # stable for any batch size
                nn.GELU(),  # smoother gradients than ReLU
                nn.Dropout(dropout_rate),
            ]

        blocks.append(nn.Linear(hidden_layers[-1], 1))
        self.network = nn.Sequential(*blocks)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform init for Linear layers; ones/zeros for LayerNorm."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------


class SalaryModel:
    """Train, persist, and serve the FCN salary predictor.

    Parameters
    ----------
    x_train_path:
        Path to the feature matrix (.npy).
    y_train_path:
        Path to the target vector (.npy).
    device:
        PyTorch device string.  Auto-detected when ``None``.
    """

    def __init__(
            self,
            x_train_path: Optional[Path] = None,
            y_train_path: Optional[Path] = None,
            device: Optional[str] = None,
    ) -> None:
        self.x_train_path = x_train_path or TRAIN_X_FILE
        self.y_train_path = y_train_path or TRAIN_Y_FILE
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.net: Optional[FCNRegressor] = None
        self.scaler = None
        self.input_dim: Optional[int] = None
        self.is_trained: bool = False

        torch.manual_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_or_train(self) -> None:
        """Load a persisted model, or train from scratch if none exists."""
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            self._load()
        else:
            self._train()

    def train_and_log(self) -> Dict:
        """Train and return metrics dict for MLflow logging.

        Returns
        -------
        dict with keys: r2, mse, history, n_samples, n_features.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        logger.info("Loading training data ...")
        X = np.load(self.x_train_path)
        y = np.load(self.y_train_path)

        # Remove rows with NaN in features or target
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X, y = X[mask], y[mask]
        logger.info("Samples: %d  Features: %d", X.shape[0], X.shape[1])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Feature scaling
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train).astype(np.float32)
        X_test_s = self.scaler.transform(X_test).astype(np.float32)

        # Log1p-transform the target: salary ~ LogNormal in practice.
        # We train on log-space and invert (expm1) for final R² and MSE.
        y_train_log = np.log1p(np.clip(y_train, 0, None)).astype(np.float32)
        y_test_log = np.log1p(np.clip(y_test, 0, None)).astype(np.float32)

        self.input_dim = X_train_s.shape[1]
        self.net = FCNRegressor(input_dim=self.input_dim).to(self.device)

        train_loader = self._make_loader(X_train_s, y_train_log)
        val_loader = self._make_loader(X_test_s, y_test_log, shuffle=False)

        history = self._fit(train_loader, val_loader)

        self.is_trained = True
        self._save()

        r2, mse = self._evaluate(val_loader, y_test)
        logger.info("Training complete | R2=%.4f | MSE=%.2f", r2, mse)

        return {
            "r2": r2,
            "mse": mse,
            "history": history,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
        }

    def predict(self, x_path: str) -> list:
        """Return salary predictions for samples in *x_path*.

        Parameters
        ----------
        x_path:
            Path to a .npy feature matrix of shape (n_samples, n_features).

        Returns
        -------
        list of float
            Predicted salaries rounded to 2 decimal places.
        """
        if not self.is_trained:
            self.load_or_train()

        X = np.load(x_path)
        logger.info("Loaded %d samples for prediction.", len(X))

        # Impute missing values with column means
        if np.isnan(X).any():
            logger.warning("NaN detected -- replacing with column means.")
            col_means = np.nanmean(X, axis=0)
            nan_idx = np.where(np.isnan(X))
            X[nan_idx] = np.take(col_means, nan_idx[1])

        X_scaled = self.scaler.transform(X).astype(np.float32)
        tensor = torch.tensor(X_scaled).to(self.device)

        self.net.eval()
        with torch.no_grad():
            log_preds = self.net(tensor).cpu().numpy()

        # Invert the log1p transform applied during training
        preds = np.expm1(log_preds)
        preds = np.round(np.clip(preds, 0, None), 2)

        logger.info("Produced %d predictions.", len(preds))
        return preds.tolist()

    # ------------------------------------------------------------------
    # Training internals
    # ------------------------------------------------------------------

    def _train(self) -> None:
        """Thin wrapper used by load_or_train() when no saved model exists."""
        self.train_and_log()

    def _fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Epoch loop with early stopping.

        Returns
        -------
        dict with train_loss and val_loss lists (one value per epoch).
        """
        # Huber loss is robust to the large salary outliers in hh.ru data
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
        )
        # Cosine annealing smoothly reduces LR to near-zero over MAX_EPOCHS
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state: Optional[Dict] = None

        history: Dict = {"train_loss": [], "val_loss": []}

        for epoch in range(1, MAX_EPOCHS + 1):
            # Training pass
            self.net.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net(X_batch), y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(X_batch)

            train_loss /= len(train_loader.dataset)
            scheduler.step()

            # Validation pass
            self.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    val_loss += (
                            criterion(self.net(X_batch), y_batch).item() * len(X_batch)
                    )

            val_loss /= len(val_loader.dataset)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if epoch % 20 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d | train=%.5f | val=%.5f",
                    epoch, MAX_EPOCHS, train_loss, val_loss,
                )

            # Early stopping with best-weights restoration
            if val_loss < best_val_loss - 1e-7:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    logger.info("Early stopping triggered at epoch %d.", epoch)
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        return history

    def _evaluate(self, loader: DataLoader, y_true: np.ndarray):
        """Compute R2 and MSE in the original salary space.

        The network outputs log1p(salary), so we invert with expm1 before
        computing metrics to make R2 comparable to non-log baselines.
        """
        from sklearn.metrics import mean_squared_error, r2_score

        self.net.eval()
        log_preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                log_preds.append(
                    self.net(X_batch.to(self.device)).cpu().numpy()
                )

        y_pred = np.expm1(np.concatenate(log_preds))  # back to salary space
        y_pred = np.clip(y_pred, 0, None)

        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        return r2, mse

    @staticmethod
    def _make_loader(
            X: np.ndarray, y: np.ndarray, shuffle: bool = True
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "input_dim": self.input_dim,
                "hidden_layers": HIDDEN_LAYERS,
                "dropout_rate": DROPOUT_RATE,
            },
            MODEL_PATH,
        )
        joblib.dump(self.scaler, SCALER_PATH)
        logger.info("Model artifacts saved to %s", RESOURCES_DIR)

    def _load(self) -> None:
        checkpoint = torch.load(MODEL_PATH, map_location=self.device)
        self.input_dim = checkpoint["input_dim"]
        self.net = FCNRegressor(
            input_dim=checkpoint["input_dim"],
            hidden_layers=checkpoint["hidden_layers"],
            dropout_rate=checkpoint["dropout_rate"],
        ).to(self.device)
        self.net.load_state_dict(checkpoint["state_dict"])
        self.scaler = joblib.load(SCALER_PATH)
        self.is_trained = True
        logger.info("Model loaded from %s", MODEL_PATH)

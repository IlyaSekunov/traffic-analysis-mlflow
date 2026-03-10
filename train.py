"""
MLflow experiment runner.

Usage
-----
    python train.py path/to/x_data.npy path/to/y_data.npy

The script trains the FCN model, logs all hyperparameters and metrics to the
remote MLflow server defined in ``config.py``, and registers the best model.

Experiment : LIne Regression HH
Model name : <lastname_firstname>_fcn  (set MLFLOW_MODEL_NAME in config.py)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch

from config import (
    BATCH_SIZE,
    DROPOUT_RATE,
    EARLY_STOPPING_PATIENCE,
    HIDDEN_LAYERS,
    LEARNING_RATE,
    MAX_EPOCHS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
    RESOURCES_DIR,
    TEST_SIZE,
)
from model import SalaryModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_experiment(x_path: Path, y_path: Path) -> str:
    """Train the model and log everything to MLflow.

    Parameters
    ----------
    x_path:
        Path to the feature matrix (.npy).
    y_path:
        Path to the target vector (.npy).

    Returns
    -------
    str
        The MLflow run_id of the completed experiment.
    """
    # ------------------------------------------------------------------
    # MLflow setup
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # ------------------------------------------------------------------
    # Remove any previously saved model so we always train from scratch
    # ------------------------------------------------------------------
    if RESOURCES_DIR.exists():
        logger.info("Removing previous model artifacts from %s", RESOURCES_DIR)
        shutil.rmtree(RESOURCES_DIR)
    RESOURCES_DIR.mkdir(parents=True)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    salary_model = SalaryModel(x_train_path=x_path, y_train_path=y_path)

    with mlflow.start_run(run_name=MLFLOW_MODEL_NAME) as run:
        run_id = run.info.run_id
        logger.info("MLflow run started: %s", run_id)

        # --- Log hyperparameters ---
        mlflow.log_params(
            {
                "model_type": "FCN",
                "hidden_layers": str(HIDDEN_LAYERS),
                "dropout_rate": DROPOUT_RATE,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            }
        )

        # --- Run training ---
        results = salary_model.train_and_log()

        # --- Log per-epoch losses ---
        for epoch, (t_loss, v_loss) in enumerate(
                zip(results["history"]["train_loss"], results["history"]["val_loss"]),
                start=1,
        ):
            mlflow.log_metric("train_loss", t_loss, step=epoch)
            mlflow.log_metric("val_loss", v_loss, step=epoch)

        # --- Log final test metrics ---
        r2 = results["r2"]
        mse = results["mse"]

        mlflow.log_metric("r2_score_test", r2)
        mlflow.log_metric("mse_test", mse)
        mlflow.log_metric("n_samples", results["n_samples"])
        mlflow.log_metric("n_features", results["n_features"])

        logger.info("r2_score_test = %.4f", r2)
        logger.info("mse_test      = %.4f", mse)

        # --- Log the PyTorch model to MLflow registry ---
        mlflow.pytorch.log_model(
            pytorch_model=salary_model.net,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME,
        )

        # --- Log scaler as a generic artifact ---
        mlflow.log_artifact(str(RESOURCES_DIR / "scaler.joblib"),
                            artifact_path="scaler")

        logger.info("Experiment finished. run_id=%s", run_id)
        return run_id


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FCN salary model and log to MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("x_path", type=Path, help="Path to X feature matrix (.npy)")
    parser.add_argument("y_path", type=Path, help="Path to y target vector (.npy)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # noqa: D401
    args = _parse_args(argv)

    for p in (args.x_path, args.y_path):
        if not p.exists():
            logger.error("File not found: %s", p)
            sys.exit(1)
        if p.suffix != ".npy":
            logger.error("Expected a .npy file, got: %s", p)
            sys.exit(1)

    run_id = run_experiment(args.x_path, args.y_path)
    print(f"\n✅  Training complete.  MLflow run_id: {run_id}")
    print(f"    Tracking UI : {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()

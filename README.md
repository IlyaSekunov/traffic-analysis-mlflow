# Salary Prediction — FCN + MLflow

Machine learning pipeline that predicts hh.ru salaries using a
**Fully Connected Network (FCN)** built with PyTorch and tracked via MLflow.

---

## Project Structure

```
regression/
├── app.py            # CLI entry point  (predict / train commands)
├── train.py          # MLflow experiment runner
├── model.py          # FCNRegressor + SalaryModel wrapper
├── config.py         # All hyperparameters and paths in one place
├── utils.py          # Stateless helpers (data loading, persistence)
├── requirements.txt
└── resources/        # Auto-created; stores trained weights and scaler
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Configuration

| Variable | Description | Default |
|---|---|---|
| `MLFLOW_TRACKING_URI` | Remote MLflow server | `http://kamnsv.com:55000/` |
| `MLFLOW_EXPERIMENT_NAME` | Experiment label | `LIne Regression HH` |
| `MLFLOW_MODEL_NAME` | Registered model name | `lastname_firstname_fcn` |
| `HIDDEN_LAYERS` | FCN hidden layer sizes | `[256, 128, 64]` |
| `DROPOUT_RATE` | Dropout probability | `0.3` |
| `LEARNING_RATE` | Adam learning rate | `1e-3` |
| `BATCH_SIZE` | Mini-batch size | `64` |
| `MAX_EPOCHS` | Maximum training epochs | `200` |
| `EARLY_STOPPING_PATIENCE` | Val-loss patience | `15` |
| `TEST_SIZE` | Fraction of data for evaluation | `0.2` |
| `RANDOM_STATE` | Random seed | `42` |

---

## Usage

### Train and log to MLflow

```bash
python app.py train ../parsing/x_data.npy ../parsing/y_data.npy
```

Or use the training module directly:

```bash
python train.py ../parsing/x_data.npy ../parsing/y_data.npy
```

Both commands will:
1. Load and clean the data (NaN removal, infinite-value clipping).
2. Split into train / test sets.
3. Fit a `StandardScaler` on the training split.
4. Train the FCN with early stopping.
5. Log hyperparameters, per-epoch losses, and final metrics to MLflow.
6. Register the PyTorch model in the MLflow Model Registry.
7. Print the `run_id` for the run with the best `r2_score_test`.

### Predict

```bash
python app.py predict ../parsing/x_data.npy
```

Outputs one salary value per line (rubles, 2 decimal places).

---

## MLflow Metrics

| Metric | Description |
|---|---|
| `r2_score_test` | R² on the held-out test set **(primary metric)** |
| `mse_test` | Mean squared error on the test set |
| `train_loss` | Per-epoch MSE loss on the training set |
| `val_loss` | Per-epoch MSE loss on the validation set |

---

## Model Architecture (FCN)

```
Input(n_features)
  → Linear(256) → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(128) → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(64)  → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(1)
```

Weights initialised with Xavier uniform.
Trained with Adam + ReduceLROnPlateau scheduler.

---

## Input / Output Format

### Input

- `x_data.npy` — float array of shape `(n_samples, n_features)`
- `y_data.npy` — float array of shape `(n_samples,)`  *(training only)*

### Output

One salary prediction per line, rounded to 2 decimal places.

---

## Dependencies

- Python ≥ 3.9
- torch ≥ 2.0
- scikit-learn ≥ 1.2
- mlflow ≥ 2.10
- numpy, pandas, joblib
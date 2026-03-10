"""
Application configuration.

Central place for all paths, hyperparameters, and MLflow settings.
Modify values here rather than scattering magic constants across modules.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
REGRESSION_DIR = Path(__file__).resolve().parent
PARSING_DIR = BASE_DIR / "parsing"
RESOURCES_DIR = REGRESSION_DIR / "resources"

# Ensure the resources directory exists at import time
RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Default data paths
# ---------------------------------------------------------------------------

TRAIN_X_FILE = PARSING_DIR / "x_data.npy"
TRAIN_Y_FILE = PARSING_DIR / "y_data.npy"

# ---------------------------------------------------------------------------
# Persisted artifact filenames
# ---------------------------------------------------------------------------

MODEL_FILENAME = "trained_model.pt"          # PyTorch state-dict
SCALER_FILENAME = "scaler.joblib"            # sklearn StandardScaler

MODEL_PATH = RESOURCES_DIR / MODEL_FILENAME
SCALER_PATH = RESOURCES_DIR / SCALER_FILENAME

# ---------------------------------------------------------------------------
# Train / evaluation split
# ---------------------------------------------------------------------------

TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Neural-network hyperparameters (FCN)
# ---------------------------------------------------------------------------

HIDDEN_LAYERS = [256, 128, 64]   # fits well for 11 input features
DROPOUT_RATE = 0.1          # low dropout — few features, model needs to memorise them
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 30

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "http://kamnsv.com:55000/"
MLFLOW_EXPERIMENT_NAME = "LIne Regression HH"

MLFLOW_MODEL_NAME = "sekunov_ilya_fcn"
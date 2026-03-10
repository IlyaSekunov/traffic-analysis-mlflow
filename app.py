"""
CLI application entry point.

Commands
--------
predict  — score a feature matrix with the saved model.
train    — train (and MLflow-log) the model on new data.

Usage
-----
    python app.py predict path/to/x_data.npy
    python app.py train   path/to/x_data.npy path/to/y_data.npy
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_npy(path: Path, label: str = "file") -> None:
    """Raise SystemExit if *path* is missing or not a .npy file."""
    if not path.exists():
        logger.error("❌  %s not found: %s", label, path)
        sys.exit(1)
    if path.suffix != ".npy":
        logger.error("❌  %s must have .npy extension: %s", label, path)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def handle_predict(x_path: Path) -> None:
    """Load the saved model and print one prediction per line."""
    _validate_npy(x_path, "Feature file")

    from model import SalaryModel

    model = SalaryModel()
    try:
        predictions = model.predict(str(x_path))
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        sys.exit(1)

    for salary in predictions:
        print(salary)


def handle_train(x_path: Path, y_path: Path) -> None:
    """Train the FCN model and log the run to MLflow."""
    _validate_npy(x_path, "Feature file (X)")
    _validate_npy(y_path, "Target file (y)")

    # Delegate to the dedicated training module which owns all MLflow logic
    from train import run_experiment

    try:
        run_id = run_experiment(x_path, y_path)
        print(f"\n✅  Model trained successfully.  run_id: {run_id}")
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Salary prediction tool for hh.ru resume data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python app.py predict ../parsing/x_data.npy\n"
            "  python app.py train   ../parsing/x_data.npy ../parsing/y_data.npy\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    # predict
    p_predict = sub.add_parser("predict", help="Predict salaries for input data.")
    p_predict.add_argument(
        "x_path", type=Path, help="Path to .npy feature matrix."
    )

    # train
    p_train = sub.add_parser("train", help="Train model and log to MLflow.")
    p_train.add_argument(
        "x_path", type=Path, help="Path to .npy feature matrix."
    )
    p_train.add_argument(
        "y_path", type=Path, help="Path to .npy target vector."
    )

    return parser


def main() -> None:  # noqa: D401
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "predict":
        handle_predict(args.x_path)
    elif args.command == "train":
        handle_train(args.x_path, args.y_path)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

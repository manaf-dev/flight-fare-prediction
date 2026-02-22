"""
End-to-end training pipeline entry point.

Runs:
1. Load raw CSV
2. Clean data
3. Engineer features
4. Run EDA (visualisations + KPI tables)
5. Train and evaluate models (with optional hyperparameter tuning)
6. Write model interpretation report
7. Save all artifacts

Usage
-----
    python pipeline/main.py                   # full run with tuning
    python pipeline/main.py --no-tuning       # skip RandomizedSearchCV (faster)
    python pipeline/main.py --test-size 0.15  # override holdout fraction

Exit codes
----------
0 = success | 1 = failure (logged to logs/pipeline.log)
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.eda import run_eda
from src.config import TEST_SIZE
from src.data.clean import clean
from src.data.load import load_raw
from src.features.build import build
from src.modeling.train import train_and_select
from src.utils import get_logger

logger = get_logger(__name__)

# Make sure the project root is on sys.path when running as a script.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flight Fare Prediction — end-to-end training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Fraction of data reserved for the chronological holdout test set.",
    )
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        default=False,
        help="Skip RandomizedSearchCV hyperparameter tuning (faster, useful for quick runs).",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        default=False,
        help="Skip EDA visualisations and KPI tables (useful when re-training only).",
    )
    return parser.parse_args()


def main() -> int:
    """Run the full pipeline. Returns 0 on success, 1 on failure."""
    args = parse_args()
    start = time.time()

    logger.info("=" * 60)
    logger.info("Flight Fare Prediction Pipeline — START")
    logger.info(
        "Config: test_size=%.2f | tuning=%s | skip_eda=%s",
        args.test_size,
        not args.no_tuning,
        args.skip_eda,
    )
    logger.info("=" * 60)

    try:
        # Step 1: Load
        logger.info("Step 1/5 — Loading raw data")
        raw_df = load_raw()

        # Step 2: Clean
        logger.info("Step 2/5 — Cleaning data")
        clean_df = clean(raw_df)

        # Step 3: Feature engineering
        logger.info("Step 3/5 — Engineering features")
        feature_df = build(clean_df)

        # Step 4: EDA
        if not args.skip_eda:
            logger.info("Step 4/5 — Running EDA")
            run_eda(feature_df)
        else:
            logger.info("Step 4/5 — EDA skipped (--skip-eda flag set)")

        # Step 5: Train, tune, evaluate, save
        logger.info("Step 5/5 — Training and evaluating models")
        results = train_and_select(
            feature_df,
            test_size=args.test_size,
            tuning=not args.no_tuning,
        )

        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info("Pipeline COMPLETE in %.1f seconds", elapsed)
        logger.info("Best model: %s", results["best_model_name"])
        logger.info("=" * 60)
        return 0

    except Exception as exc:
        logger.exception("Pipeline FAILED: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
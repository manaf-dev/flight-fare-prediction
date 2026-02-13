"""Main pipeline entrypoint for end-to-end training and reporting."""

import argparse
import sys

sys.path.append(".")

from src.config import TEST_SIZE
from src.data_loader import get_dataframe
from src.eda import create_visualizations, descriptive_statistics, kpi_exploration
from src.modeling import train_and_select_model, write_model_report
from src.utils import get_logger, setup_logging

logger = get_logger(__name__)



def parse_args():
    parser = argparse.ArgumentParser(description="Flight fare model training pipeline")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="Chronological holdout ratio")
    parser.add_argument("--no-tuning", action="store_true", help="Disable RandomizedSearchCV tuning")
    return parser.parse_args()



def main():
    args = parse_args()
    setup_logging()

    logger.info("Starting Flight Fare Prediction Pipeline")
    logger.info("Config: test_size=%s, tuning=%s", args.test_size, not args.no_tuning)

    df = get_dataframe()
    logger.info("Canonical dataframe ready with shape %s", df.shape)

    descriptive_statistics(df)
    create_visualizations(df)
    kpi_exploration(df)

    results = train_and_select_model(df, test_size=args.test_size, tuning=not args.no_tuning)
    write_model_report(
        metrics_df=results["metrics_df"],
        metadata=results["metadata"],
        output_path="reports/model_interpretation_and_insights.md",
    )

    logger.info("Pipeline completed. Best model: %s", results["best_model_name"])


if __name__ == "__main__":
    main()

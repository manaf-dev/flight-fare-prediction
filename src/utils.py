"""Utility functions for the project."""

import logging
from pathlib import Path

from src.config import LOGS_PATH



def setup_logging():
    """Set up logging configuration for the entire pipeline."""
    log_dir = Path(LOGS_PATH)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "pipeline.log"),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(__name__)



def get_logger(name):
    """Get a logger instance for a module."""
    return logging.getLogger(name)

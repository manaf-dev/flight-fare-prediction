"""Utility functions for the project"""

import logging
from pathlib import Path

from config import LOGS_PATH


def setup_logging():
    """
    Setup logging configuration for the entire pipeline.

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_dir = Path(LOGS_PATH).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove existing handlers to avoid duplicate logs
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{LOGS_PATH}/pipeline.log"),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(__name__)


def get_logger(name):
    """
    Get a logger instance for a module.

    Parameters
    ----------
    name : str
        Module name for the logger

    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)

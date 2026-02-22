"""Utility functions for the project."""

import logging
from pathlib import Path

from src.config import CONFIG


def get_logger(name):
    """Create or retrieve a configured logger."""
    CONFIG.logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(CONFIG.logs_dir) / "pipeline.log"

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

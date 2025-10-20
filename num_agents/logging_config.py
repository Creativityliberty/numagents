"""
Logging configuration for the Nüm Agents SDK.

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import logging
import sys
from typing import Optional


# Default log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a configured logger for the SDK.

    Args:
        name: Name of the logger (typically __name__)
        level: Optional logging level. If not provided, uses INFO.

    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers exist (avoid duplicate handlers)
    if not logger.handlers:
        if level is None:
            level = logging.INFO

        logger.setLevel(level)

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

    return logger


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure logging for the entire SDK.

    Args:
        level: The logging level to use
    """
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def set_log_level(level: int) -> None:
    """
    Set the log level for all Nüm Agents loggers.

    Args:
        level: The logging level to use (e.g., logging.DEBUG, logging.INFO)
    """
    logging.getLogger("num_agents").setLevel(level)

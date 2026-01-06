"""Logging configuration and utilities."""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_file: str = "logs/app.log",
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "1 week",
) -> None:
    """
    Configure the logger with file and console output.

    Args:
        log_file: Path to the log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate the log file
        retention: How long to keep old log files
    """
    # Remove default logger
    logger.remove()

    # Add console handler with custom format
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # Add file handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation=rotation,
        retention=retention,
        compression="zip",
    )

    logger.info(f"Logger initialized - Level: {level}, Output: {log_file}")


def get_logger(name: str):
    """
    Get a logger instance with the specified name.

    Args:
        name: Name of the logger (usually __name__)

    Returns:
        Logger instance
    """
    return logger.bind(name=name)

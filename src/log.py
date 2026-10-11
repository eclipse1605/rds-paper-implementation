"""Console and file logging for a training run."""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER_NAME = "rds"


def get_logger() -> logging.Logger:
    """Return the project logger, with a console handler if none is attached."""

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(_is_console(handler) for handler in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console)
    return logger


def configure(path: str | Path) -> logging.Logger:
    """Append this run to `path` as well as the console."""

    logger = get_logger()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.handlers = [handler for handler in logger.handlers if not isinstance(handler, logging.FileHandler)]
    log_file = logging.FileHandler(destination, mode="a", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log_file.setFormatter(formatter)
    for handler in logger.handlers:
        handler.setFormatter(formatter)
    logger.addHandler(log_file)
    logger.info("logging to %s", destination)
    return logger


def _is_console(handler: logging.Handler) -> bool:
    return isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)

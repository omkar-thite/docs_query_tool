import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

def setup_logging(log_file: str = "app.log", level: int = logging.INFO):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)  # creates logs/ if missing

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Rotating file — caps at 5MB, keeps last 3 files
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
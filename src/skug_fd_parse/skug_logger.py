import logging
import os
import sys

from skug_fd_parse import constants as const


def init_logger() -> logging.Logger:
    """
    Initialize a logger with both console and file handlers.
    File logs are saved in a 'logs' folder in the parent directory of the module.

    :return: logging.Logger object
    """
    # Get the path to the parent directory of the module
    os.path.join(os.path.dirname(__file__), "logs")
    log_file_name: str = "skug_fd_parse"

    # Create console and file handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    file_handler: logging.FileHandler = logging.FileHandler(
        f"{log_file_name}{const.LOG_FILE_EXT}", mode="w", encoding="utf8"
    )
    file_handler.setLevel(logging.DEBUG)

    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter: logging.Formatter = logging.Formatter(log_format)

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger: logging.Logger = logging.getLogger()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)

    return logger


log = init_logger()

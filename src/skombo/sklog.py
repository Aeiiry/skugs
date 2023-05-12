import logging
import os
import sys

from skombo import const as const
from skombo import file_man as fm


def init_logger() -> logging.Logger:
    """
    Initialize a logger with both console and file handlers.
    File logs are saved in a 'logs' folder in the parent directory of the module.

    :return: logging.Logger object
    """
    # Get the path to the parent directory of the module
    os.path.join(os.path.dirname(__file__), "logs")
    log_file_name: str = "skombo"

    # Create console and file handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    file_handler: logging.FileHandler = logging.FileHandler(
        f"{log_file_name}{const.LOG_FILE_EXT}", mode="w", encoding="utf8"
    )
    file_handler.setLevel(logging.DEBUG)

    log_file_format = (
        "%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
    )
    date_format = "%H:%M:%S"

    console_format = "%(message)s"

    log_formatter: logging.Formatter = logging.Formatter(log_file_format, date_format)
    console_formatter: logging.Formatter = logging.Formatter(console_format)

    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(log_formatter)

    # Add handlers to logger
    logger: logging.Logger = logging.getLogger()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)
    # Redirect stdout and stderr to the logger
    # sys.stdout = StreamToLogger(logger, logging.INFO)  # type: ignore
    sys.stderr = StreamToLogger(logger, logging.ERROR)  # type: ignore

    return logger


def get_logger() -> logging.Logger:
    if not logging.getLogger().hasHandlers():
        init_logger()

    return logging.getLogger(fm.MODULE_NAME)


class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger: logging.Logger, log_level: int = logging.INFO) -> None:
        self.logger: logging.Logger = logger
        self.log_level: int = log_level
        self.linebuf: str = ""

    def write(self, buf: str) -> None:
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(
        self,
    ) -> None:  # Needed as we are redirecting stdout and stderr to the logger
        pass

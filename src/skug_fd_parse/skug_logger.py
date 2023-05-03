import logging
import os
import sys


# noinspection SpellCheckingInspection
def init_logger() -> logging.Logger:
    file_name = "skug_stats"
    file_folder = "logs"
    # logs should be in a folder called logs one level up from the current working directory
    log_file_path = os.path.join(os.getcwd(), "../..", file_folder)
    if os.path.exists(log_file_path) is False:
        os.mkdir(log_file_path)
    file_name = os.path.join(log_file_path, file_name)
    # Create file handler
    log_file_handler = logging.FileHandler(f"{file_name}.log", mode="w")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    # Create root logger
    result: logging.Logger = logging.getLogger()
    result.setLevel(logging.DEBUG)
    # Add the console handler to the root logger
    result.addHandler(console)
    # Add the file handler to the root logger
    result.addHandler(log_file_handler)

    log_file_format = (
        "%(relativeCreated)d - %(levelname)s - %(filename)s - %(lineno)d - %(message)s"
    )
    log_file_handler.setFormatter(logging.Formatter(log_file_format))
    return result


log = init_logger()

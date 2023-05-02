import logging
import os
import sys


def init_logger() -> logging.Logger:
    file_name = "skug_stats"
    file_folder = "logs"
    # logs should be in a folder called logs one level up from the current working directory
    logfilepath = os.path.join(os.getcwd(), "..", file_folder)
    if os.path.exists(logfilepath) is False:
        os.mkdir(logfilepath)
    file_name = os.path.join(logfilepath, file_name)
    # Create file handler
    logfilehandler = logging.FileHandler(f"{file_name}.log", mode="w")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter("%(message)s")
    )
    # Create root logger
    result: logging.Logger = logging.getLogger()
    result.setLevel(logging.DEBUG)
    # Add the console handler to the root logger
    result.addHandler(console)
    # Add the file handler to the root logger
    result.addHandler(logfilehandler)

    logfileformat = (
        "%(relativeCreated)d - %(levelname)s - %(filename)s - %(lineno)d - %(message)s"
    )
    logfilehandler.setFormatter(logging.Formatter(logfileformat))
    return result


log = init_logger()


# if passed a print function, intercept it and log the output
def log_print(*args, **kwargs):
    # print(args, kwargs)
    if "file" in kwargs and kwargs["file"] == sys.stdout:
        log.info(*args)
        return
    log.info(*args)

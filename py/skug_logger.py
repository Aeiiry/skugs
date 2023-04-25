import logging
import os
import sys


def init_logger() -> logging.Logger:
    logfilename = "skug_stats"
    logfilefolder = "logs"
    # logs should be in a folder called logs one level up from the current working directory
    logfilepath = os.path.join(os.getcwd(), "..", logfilefolder)
    if os.path.exists(logfilepath) is False:
        os.mkdir(logfilepath)
    logfilename = os.path.join(logfilepath, logfilename)
    # Create file handler
    logfilehandler = logging.FileHandler(f"{logfilename}.log", mode="w")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    # Create root logger
    result: logging.Logger = logging.getLogger()
    result.setLevel(logging.DEBUG)
    # Add the console handler to the root logger
    result.addHandler(console)
    # Add the file handler to the root logger
    result.addHandler(logfilehandler)

    logfileformat = "%(relativeCreated)d - %(levelname)s - %(message)s"
    logfilehandler.setFormatter(logging.Formatter(logfileformat))
    return result


sklogger = init_logger()

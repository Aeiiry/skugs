import logging
import os
import re
import sys
import datetime
import time
import atexit

# get the start datetime
START_TIME = datetime.datetime.now()


NUMERIC_COLUMNS: list[str] = [
    "on_hit",
    "on_block",
    "on_pushblock",
]

NUMERIC_LIST_COLUMNS: list[str] = [
    "damage",
    "chip_damage",
    "startup",
    "active",
    "recovery",
    "hitstun",
    "blockstun",
    "hitstop",
]

UNDIZZY_DICT: dict[str, int] = {
    "LIGHT_NORMAL": 15,
    "MEDIUM_NORMAL": 20,
    "HEAVY_NORMAL": 30,
    "SPECIAL_MOVE": 20,
    "SUPER": 0,
    "THROW": 0,
    "AIR_THROW": 0,
    "TAUNT": 0,
}
"""Undizzy values for each hit type.
"""

LOG_FILE_SUFFIX = "skugLog"
LOG_FOLEDER_NAME = "logs"
LOG_FILE_EXT = ".log"
LOG_LEVEL_CONSOLE: int = logging.INFO
LOG_LEVEL_FILE: int = logging.DEBUG

RE_NORMAL_MOVE: re.Pattern[str] = re.compile(
    r"^j?\.?\d?.?([lmh])[pk]", flags=re.IGNORECASE
)
""" regex to find normal moves """
RE_IN_PAREN: re.Pattern[str] = re.compile(r"\((.*?)\)")
RE_X_N: re.Pattern[str] = re.compile(r"(\d*?\.?\d+\s?)([x*]\s?)(\d+)")
RE_BRACKETS_X_N: re.Pattern[str] = re.compile(r"(\[[\d,\s]*?]\s?)([x*])(\d+)")

PLUS_MINUS_COLS: list[str] = [
    "on_hit",
    "on_block",
    "startup",
    "active",
    "hitstun",
    "on_pushblock",
]

REMOVE_NEWLINE_COLS: list[str] = [
    "guard",
    "properties",
    "damage",
    "meter",
    "on_hit",
    "on_block",
    "startup",
    "active",
    "recovery",
    "hitstun",
    "hitstop",
    "on_pushblock",
    "footer",
]

UNIVERSAL_MOVE_CATEGORIES: dict[str, str] = {
    "TAG IN": "TAG",
    "SNAPBACK": "SNAP",
    "THROW": "THROW",
    "AIR THROW": "AIR_THROW",
    "TAUNT": "TAUNT",
    "ASSIST RECOVERY": "ASSIST_RECOVERY",
}

NORMAL_STRENGTHS: dict[str, str] = {
    "L": "LIGHT",
    "M": "MEDIUM",
    "H": "HEAVY",
}

IGNORED_MOVES: list[str] = ["RESTAND", "KARA", "ADC", "AD","OTG", "RE STAND", "RE-STAND"]

FRAME_VALUE_REPLACEMENTS: list[str] = ["+", "±"]
FRAME_VALUE_REPLACEMENTS_STR: str = f"[{''.join(FRAME_VALUE_REPLACEMENTS)}]"
FRAME_VALUE_REPLACEMENTS_RE: re.Pattern[str] = re.compile(FRAME_VALUE_REPLACEMENTS_STR)

##################################################################################################
#                                       COMBO CONSTANTS                                          #
##################################################################################################

UNDIZZY_MAX = 240

SCALING_FACTOR = 0.875

SCALING_MIN_1K = 0.275
SCALING_MIN = 0.2

SCALING_DHC = 0.7

SCALING_ASSIST_START = 0.66
SCALING_ASSIST_START_COUNTER = 0.9

SCALING_START = 1.0


##################################################################################################
#                                        CHARACTER NAMES                                         #
##################################################################################################


class Characters:
    AN = "ANNIE"
    BE = "BEOWULF"
    BB = "BIG BAND"
    BD = "BLACK DAHLIA"
    CE = "CEREBELLA"
    DB = "DOUBLE"
    EL = "ELIZA"
    FI = "FILIA"
    FU = "FUKUA"
    MF = "MS. FORTUNE"
    PW = "PAINWHEEL"
    PA = "PARASOUL"
    PE = "PEACOCK"
    RF = "ROBO-FORTUNE"
    SQ = "SQUIGLY"
    UM = "UMBRELLA"
    VA = "VALENTINE"


CHARS = Characters()
##################################################################################################
#                                      FILE PATH CONSTANTS                                       #
##################################################################################################


FD_BOT_FILE_PREFIX = "SG2E - Frame Data Bot Data - "
DATA_NAME = "data"
CSVS = "csvs"
GAME_DATA = "game_data"

if getattr(sys, "frozen", False):
    ABS_PATH: str = os.path.dirname(sys.executable)
else:
    ABS_PATH = os.path.abspath(os.path.dirname(__file__))

MODULE_NAME: str = os.path.basename(ABS_PATH)

DATA_PATH: str = os.path.join(ABS_PATH, DATA_NAME)
CSV_PATH: str = os.path.join(DATA_PATH, CSVS)
GAME_DATA_PATH: str = os.path.join(CSV_PATH, GAME_DATA)

CHARACTER_DATA_PATH: str = os.path.join(
    GAME_DATA_PATH, f"{FD_BOT_FILE_PREFIX}Characters.csv"
)
FRAME_DATA_PATH: str = os.path.join(GAME_DATA_PATH, f"{FD_BOT_FILE_PREFIX}Moves.csv")
MOVE_NAME_ALIASES_PATH: str = os.path.join(
    GAME_DATA_PATH, f"{FD_BOT_FILE_PREFIX}Macros.csv"
)

TEST_DATA_FOLDER = "test_data"
TEST_COMBOS_SUFFIX = "_test_combos.csv"

LOG_DIR: str = os.path.join(ABS_PATH, "logs")


##################################################################################################
#                                      LOGGING CONSTANTS                                         #
##################################################################################################


def init_logger(name=__name__) -> logging.Logger:
    """
    Initialize a logger with both console and file handlers.
    File logs are saved in a 'logs' folder in the parent directory of the module.

    :return: logging.Logger object
    """
    # Get the path to the parent directory of the module

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    log_file_name = os.path.join(LOG_DIR, name)
    # Create console and file handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    file_handler: logging.FileHandler = logging.FileHandler(
        f"{log_file_name}{LOG_FILE_EXT}", mode="w", encoding="utf8"
    )
    file_handler.setLevel(logging.DEBUG)
    # log format: datetime [ms since start] [log level] [file name:line number] [message]
    log_file_format = (
        "[%(asctime)s] [%(module)s.%(funcName)s] [%(levelname)s] %(message)s"
    )

    datetime_format = "%Y-%m-%d %H:%M:%S"

    console_format = "%(message)s"

    log_formatter: logging.Formatter = logging.Formatter(
        log_file_format, datetime_format
    )
    console_formatter: logging.Formatter = logging.Formatter(console_format)

    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(log_formatter)

    # Add handlers to logger
    logger: logging.Logger = logging.getLogger(name)
    # Redirect stdout and stderr to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)
    # sys.stderr = StreamToLogger(logger, logging.ERROR)  # type: ignore
    # sys.stdout = StreamToLogger(logger, logging.INFO)  # type: ignore

    return logger


def get_logger(name=__name__) -> logging.Logger:
    if not logging.getLogger(name).hasHandlers():
        init_logger(name)

    return logging.getLogger(name)


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
            self.logger.log(self.log_level, line.rstrip()) if line.rstrip() not in [
                "",
                "\n",
                "^",
                "[",
                "]",
                "~",
            ] else None

    def flush(
        self,
    ) -> None:  # Needed as we are redirecting stdout and stderr to the logger
        pass

    def isatty(self) -> None:
        pass


log: logging.Logger = get_logger(__name__)

log.info("Constants initialized")

log.info(f"ABS_PATH: {ABS_PATH}")
log.info(f"MODULE_NAME: {MODULE_NAME}")
log.info(f"DATA_PATH: {DATA_PATH}")
log.info(f"CSV_PATH: {CSV_PATH}")
log.info(f"GAME_DATA_PATH: {GAME_DATA_PATH}")
log.info(f"CHARACTER_DATA_PATH: {CHARACTER_DATA_PATH}")
log.info(f"FRAME_DATA_PATH: {FRAME_DATA_PATH}")
log.info(f"MOVE_NAME_ALIASES_PATH: {MOVE_NAME_ALIASES_PATH}")
log.info(f"TEST_DATA_FOLDER: {TEST_DATA_FOLDER}")
log.info(f"TEST_COMBOS_SUFFIX: {TEST_COMBOS_SUFFIX}")

log.info("Logger initialized")


def exit_handler() -> None:
    # get the end datetime
    END_TIME = datetime.datetime.now()

    # get execution time
    elapsed_time = END_TIME - START_TIME
    log.info(f"Execution time: {elapsed_time} seconds 🤠")


atexit.register(exit_handler)

"""init for skombo package.
primarily contains constants used throughout the package. also contains the logger."""
import atexit
import datetime

START_TIME = datetime.datetime.now()

import logging
import os
import re
import sys
from dataclasses import dataclass


@dataclass
class Columns:
    char: str = "character"
    m_name: str = "move_name"
    a_names: str = "alt_names"
    guard: str = "guard"
    props: str = "properties"
    dmg: str = "damage"
    chip: str = "chip_damage"
    meter: str = "meter_on_hit"
    meter_whiff: str = "meter_on_whiff"
    onhit: str = "on_hit_adv"
    onhit_eff: str = "on_hit_effect"
    onblock: str = "on_block_adv"
    startup: str = "startup"
    active: str = "active"
    recovery: str = "recovery"
    hitstun: str = "hitstun"
    blockstun: str = "blockstun"
    hitstop: str = "hitstop"
    blockstop: str = "blockstop"
    super_hitstop: str = "super_hitstop"
    onpb: str = "on_pushblock"
    footer: str = "footer"
    thumb_url: str = "thumbnail_url"
    footer_url: str = "footer_url"
    move_cat: str = "move_category"
    undizzy: str = "undizzy"
    scaling: str = "scaling_effect"
    hit_scaling: str = "scaling_for_hit"
    mod_scaling: str = "scaling_after_modifiers"


COLS = Columns()

COL_TYPES: dict[str, str | tuple[str, str]] = {
    COLS.char: "str",
    COLS.m_name: "str",
    COLS.a_names: ("list", "str"),
    COLS.guard: ("list", "str"),
    COLS.props: ("list", "str"),
    COLS.dmg: ("list", "int"),
    COLS.scaling: ("dict", "int"),
    COLS.chip: ("list", "int"),
    COLS.meter: ("list", "int"),
    COLS.meter_whiff: ("list", "int"),
    COLS.onhit: "int",
    COLS.onhit_eff: "str",
    COLS.onblock: "int",
    COLS.startup: ("list", "int"),
    COLS.active: ("list", "int"),
    COLS.recovery: ("list", "int"),
    COLS.hitstun: ("list", "int"),
    COLS.blockstun: ("list", "int"),
    COLS.hitstop: ("list", "int"),
    COLS.blockstop: ("list", "int"),
    COLS.super_hitstop: "int",
    COLS.onpb: ("list", "int"),
    COLS.footer: "str",
    COLS.thumb_url: "str",
    COLS.footer_url: "str",
    COLS.move_cat: "str",
    COLS.undizzy: "int",
}


@dataclass
class ColumnClassification:
    REMOVE_NEWLINE_COLS = [
        COLS.guard,
        COLS.props,
        COLS.dmg,
        COLS.meter,
        COLS.onhit,
        COLS.onblock,
        COLS.startup,
        COLS.active,
        COLS.recovery,
        COLS.hitstun,
        COLS.hitstop,
        COLS.onpb,
        COLS.footer,
    ]

    PLUS_MINUS_COLS = [
        COLS.onhit,
        COLS.onblock,
        COLS.startup,
        COLS.active,
        COLS.hitstun,
        COLS.onpb,
    ]

    NUMERIC_COLUMNS = [
        COLS.onhit,
        COLS.onblock,
        COLS.onpb,
    ]

    LIST_COLUMNS = [
        COLS.dmg,
        COLS.chip,
        COLS.active,
        COLS.hitstun,
        COLS.blockstun,
        COLS.hitstop,
        COLS.blockstop,
        COLS.super_hitstop,
        COLS.undizzy,
        COLS.startup,
        COLS.recovery,
        COLS.meter,
        COLS.meter_whiff,
        COLS.props,
        COLS.guard,
    ]

    XN_COLS = [
        COLS.dmg,
        COLS.hitstun,
        COLS.blockstun,
        COLS.hitstop,
        COLS.meter,
        COLS.active,
    ]

    COMBO_COLS = [
        COLS.char,
        COLS.m_name,
        COLS.dmg,
        COLS.scaling,
        COLS.hit_scaling,
        COLS.mod_scaling,
        "scaled_damage",
        "summed_damage",
    ]


COLS_CLASSES = ColumnClassification()
# get the start datetime


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

IGNORED_MOVES: list[str] = [
    "RESTAND",
    "KARA",
    "ADC",
    "AD",
    "OTG",
    "RE STAND",
    "RE-STAND",
]

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


@dataclass
class Characters:
    """Class to hold character names as constants."""

    AN: str = "ANNIE"
    BE: str = "BEOWULF"
    BB: str = "BIG BAND"
    BD: str = "BLACK DAHLIA"
    CE: str = "CEREBELLA"
    DB: str = "DOUBLE"
    EL: str = "ELIZA"
    FI: str = "FILIA"
    FU: str = "FUKUA"
    MF: str = "MS. FORTUNE"
    PW: str = "PAINWHEEL"
    PA: str = "PARASOUL"
    PE: str = "PEACOCK"
    RF: str = "ROBO-FORTUNE"
    SQ: str = "SQUIGLY"
    UM: str = "UMBRELLA"
    VA: str = "VALENTINE"


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
    # log format: [minutes:seconds:ms] [log level] [filename:line number] [message
    log_file_format = (
        # get ms
        "[%(asctime)s:%(msecs)03d] %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
    )
    datetime_format = "%M:%S"

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
        pass  # Needed as we are redirecting stdout and stderr to the logger


LOG: logging.Logger = get_logger(__name__)

LOG.info("Constants initialized")

LOG.info(f"ABS_PATH: {ABS_PATH}")
LOG.info(f"MODULE_NAME: {MODULE_NAME}")

LOG.info("Logger initialized")


def exit_handler() -> None:
    # get the end datetime
    END_TIME = datetime.datetime.now()

    # get execution time
    elapsed_time = END_TIME - START_TIME
    LOG.info(f"Execution time: {elapsed_time} seconds")


atexit.register(exit_handler)

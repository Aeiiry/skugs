"""init for skombo package.
primarily contains constants used throughout the package. also contains the logger."""
import atexit
import datetime
import os
import re
import sys
from dataclasses import dataclass

from loguru import logger as log

START_TIME = datetime.datetime.now()


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


@dataclass
class ComboInputColumns:
    name: str = "name"
    character: str = "character"
    assist_1: str = "assist_1"
    assist_2: str = "assist_2"
    own_team_size: str = "own_team_size"
    opponent_team_size: str = "opponent_team_size"
    notation: str = "notation"

    counter_hit: str = "counter_hit"
    undizzy: str = "undizzy"

    damage: str = "damage"
    meter: str = "meter"


COMBO_INPUT_COLS = ComboInputColumns()

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

LOG_FILE_SUFFIX = "log"
LOG_FOLEDER_NAME = "logs"
LOG_FILE_EXT = ".log"

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

TESTS = "tests"
FD_BOT_FILE_PREFIX = "SG2E - Frame Data Bot Data - "
DATA_NAME = "data"
CSVS = "csvs"
GAME_DATA = "game_data"
TEST_DATA_FOLDER = "test_data"
TEST_COMBOS_SUFFIX = "_test_combos.csv"

if getattr(sys, "frozen", False):
    MODULE_PATH: str = os.path.dirname(sys.executable)
else:
    MODULE_PATH = os.path.abspath(os.path.dirname(__file__))

MODULE_NAME: str = os.path.basename(MODULE_PATH)

SRC_FOLDER_PATH: str = os.path.dirname(MODULE_PATH)
TOP_LEVEL_FOLDER_PATH: str = os.path.dirname(SRC_FOLDER_PATH)

DATA_PATH: str = os.path.join(MODULE_PATH, DATA_NAME)
CSV_PATH: str = os.path.join(DATA_PATH, CSVS)
GAME_DATA_PATH: str = os.path.join(CSV_PATH, GAME_DATA)

TESTS_PATH: str = os.path.join(TOP_LEVEL_FOLDER_PATH, TESTS)
TESTS_DATA_PATH: str = os.path.join(TESTS_PATH, TEST_DATA_FOLDER)

# filter CHARS dictionary values to only include strings
char_values = [val for val in CHARS.__dict__.values() if isinstance(val, str)]
# use a regular variable assignment instead of an assignment expression
TEST_COMBO_CSVS = [
    os.path.join(TESTS_DATA_PATH, csv_name)
    for char in char_values
    if (csv_name := f"{char.lower()}{TEST_COMBOS_SUFFIX}")
    in os.listdir(TESTS_DATA_PATH)
]

LOG_DIR: str = os.path.join(MODULE_PATH, "logs")


##################################################################################################
#                                    SET UP LOGGING                                              #
##################################################################################################

# remove the default logger


def config_logger() -> None:
    log.remove()

    levels = [
        {"name": "TRACE", "color": "<white>", "icon": "🔍"},
        {"name": "DEBUG", "color": "<cyan>", "icon": "🐛"},
        {"name": "INFO", "color": "<white>", "icon": "🌟"},
        {"name": "WARNING", "color": "<yellow>", "icon": "🔥"},
        {"name": "ERROR", "color": "<red>", "icon": "❌"},
        {"name": "CRITICAL", "color": "<magenta>", "icon": "💀"},
    ]
    # set up custom levels
    for level in levels:
        log.level(**level)

    ##########################################

    # formatting components

    format_components = {
        # time in gray italics
        "time": "<fg #999999><italic>{time:mm:ss:SSS}</italic></fg #999999>|",
        "level_icon": "{level.icon}|",
        "level": "<level>{level: <8}</level>",
        # format module and line with left aligned space for both as if they were one string
        "module_line": "|<fg #999999>{module: <10}:{line: <3}</fg #999999>",
        "message": "| <level>{message}</level>",
    }

    ##########################################

    log_file_path = os.path.join(LOG_DIR, f"{MODULE_NAME}{LOG_FILE_EXT}")
    file_components = [
        "time",
        "level_icon",
        "level",
        "module_line",
        "message",
    ]
    console_components = ["level_icon", "level", "message"]

    log.add(
        sink=log_file_path,
        format="".join(format_components[component] for component in file_components),
        level="DEBUG",
        rotation="1 hour",
        retention="4 weeks",
        enqueue=True,
        backtrace=True,
        diagnose=True,
        catch=True,
    )
    log.add(
        sink=sys.stderr,
        format="".join(
            format_components[component] for component in console_components
        ),
        level="INFO",
        enqueue=True,
    )


config_logger()


@atexit.register
def exit_handler() -> None:
    # get the end datetime
    END_TIME = datetime.datetime.now()

    # get execution time
    elapsed_time = END_TIME - START_TIME
    log.info(f"Execution time: {elapsed_time} seconds 🤠")


# test_funct()

log.info("Constants initialized")

log.info(f"ABS_PATH: {MODULE_PATH}")
log.info(f"MODULE_NAME: {MODULE_NAME}")

log.info("Logger initialized")

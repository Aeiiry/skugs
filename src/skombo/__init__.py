"""init for skombo package.
primarily contains constants used throughout the package. also contains the logger."""

import os
import re
import sys
from dataclasses import dataclass

from loguru import logger as log


@dataclass
class FdColumns:
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


FD_COLS = FdColumns()


@dataclass
class CharCols:
    char = "character"
    short_names = "character_start"
    color = "color"


CHAR_COLS = CharCols()

COL_TYPES: dict[str, str | tuple[str, str]] = {
    FD_COLS.char: "str",
    FD_COLS.m_name: "str",
    FD_COLS.a_names: ("list", "str"),
    FD_COLS.guard: ("list", "str"),
    FD_COLS.props: ("list", "str"),
    FD_COLS.dmg: ("list", "int"),
    FD_COLS.scaling: ("dict", "int"),
    FD_COLS.chip: ("list", "int"),
    FD_COLS.meter: ("list", "int"),
    FD_COLS.meter_whiff: ("list", "int"),
    FD_COLS.onhit: "int",
    FD_COLS.onhit_eff: "str",
    FD_COLS.onblock: "int",
    FD_COLS.startup: ("list", "int"),
    FD_COLS.active: ("list", "int"),
    FD_COLS.recovery: ("list", "int"),
    FD_COLS.hitstun: ("list", "int"),
    FD_COLS.blockstun: ("list", "int"),
    FD_COLS.hitstop: ("list", "int"),
    FD_COLS.blockstop: ("list", "int"),
    FD_COLS.super_hitstop: "int",
    FD_COLS.onpb: ("list", "int"),
    FD_COLS.footer: "str",
    FD_COLS.thumb_url: "str",
    FD_COLS.footer_url: "str",
    FD_COLS.move_cat: "str",
    FD_COLS.undizzy: "int",
}


@dataclass
class ColumnClassification:
    REMOVE_NEWLINE_COLS = [
        FD_COLS.guard,
        FD_COLS.props,
        FD_COLS.dmg,
        FD_COLS.meter,
        FD_COLS.onhit,
        FD_COLS.onblock,
        FD_COLS.startup,
        FD_COLS.active,
        FD_COLS.recovery,
        FD_COLS.hitstun,
        FD_COLS.hitstop,
        FD_COLS.onpb,
        FD_COLS.footer,
    ]

    PLUS_MINUS_COLS = [
        FD_COLS.onhit,
        FD_COLS.onblock,
        FD_COLS.startup,
        FD_COLS.active,
        FD_COLS.hitstun,
        FD_COLS.onpb,
    ]

    NUMERIC_COLUMNS = [
        FD_COLS.onhit,
        FD_COLS.onblock,
        FD_COLS.onpb,
    ]

    LIST_COLUMNS = [
        FD_COLS.dmg,
        FD_COLS.chip,
        FD_COLS.active,
        FD_COLS.hitstun,
        FD_COLS.blockstun,
        FD_COLS.hitstop,
        FD_COLS.blockstop,
        FD_COLS.super_hitstop,
        FD_COLS.undizzy,
        FD_COLS.startup,
        FD_COLS.recovery,
        FD_COLS.meter,
        FD_COLS.meter_whiff,
        FD_COLS.props,
        FD_COLS.guard,
    ]

    XN_COLS = [
        FD_COLS.dmg,
        FD_COLS.hitstun,
        FD_COLS.blockstun,
        FD_COLS.hitstop,
        FD_COLS.meter,
        FD_COLS.active,
    ]

    COMBO_COLS = [
        FD_COLS.char,
        FD_COLS.m_name,
        FD_COLS.dmg,
        FD_COLS.scaling,
        FD_COLS.hit_scaling,
        FD_COLS.mod_scaling,
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

    expected_damage: str = "expected_damage"
    meter: str = "meter"


COMBO_INPUT_COLS = ComboInputColumns()
C_COLS = COMBO_INPUT_COLS

COMBO_INPUT_COLS_DTYPES = {
    C_COLS.name: "string",
    C_COLS.character: "string",
    C_COLS.assist_1: "string",
    C_COLS.assist_2: "string",
    C_COLS.own_team_size: "Int8",
    C_COLS.opponent_team_size: "Int8",
    C_COLS.notation: "string",
    C_COLS.counter_hit: "boolean",
    C_COLS.undizzy: "Int8",
    C_COLS.expected_damage: "Int16",
    C_COLS.meter: "Int8",
}

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
    "TAG": 20,
    "SNAP": 20,
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

FONT_STYLE_SUBSTRINGS = {
    "bold": ["bold", "bld", "bd"],
    "italic": ["italic", "ital", "itl", "it"],
    "bold_italic": [
        "bold_italic",
        "bolditalic",
        "bold-italic",
        "bolditalic",
        "bldital",
        "bd-it",
        "bd it",
    ],
}
"""Organised as:
{
    style: [style, substrings, ...],
}"""

from pathlib import Path

TESTS = "tests"
FD_BOT_FILE_PREFIX = "SG2E - Frame Data Bot Data - "
DATA_NAME = "data"
CSVS = "csvs"
GAME_DATA = "game_data"
TEST_DATA_FOLDER = "test_data"
TEST_COMBOS_SUFFIX = "_test_combos.csv"

MODULE_PATH: Path = Path(__file__).resolve().parent
MODULE_NAME: str = MODULE_PATH.name

SRC_FOLDER_PATH: Path = MODULE_PATH.parent
TOP_LEVEL_FOLDER_PATH: Path = SRC_FOLDER_PATH.parent

DATA_PATH: Path = MODULE_PATH / DATA_NAME
CSV_PATH: Path = DATA_PATH / CSVS
GAME_DATA_PATH: Path = CSV_PATH / GAME_DATA

TESTS_PATH: Path = TOP_LEVEL_FOLDER_PATH / TESTS
TESTS_DATA_PATH: Path = TESTS_PATH / TEST_DATA_FOLDER

# filter CHARS dictionary values to only include strings
char_values = [val for val in CHARS.__dict__.values() if isinstance(val, str)]
# use a regular variable assignment instead of an assignment expression
TEST_COMBO_CSVS = [
    TESTS_DATA_PATH / f"{char.lower()}{TEST_COMBOS_SUFFIX}"
    for char in char_values
    if (TESTS_DATA_PATH / f"{char.lower()}{TEST_COMBOS_SUFFIX}").exists()
]

LOG_DIR: Path = MODULE_PATH / "logs"


##################################################################################################
#                                    SET UP LOGGING                                              #
##################################################################################################

# remove the default logger


def config_logger() -> None:
    log.remove()

    log.level(name="TRACE", color="<white>", icon="🔍")
    log.level(name="DEBUG", color="<cyan>", icon="🐛")
    log.level(name="INFO", color="<white>", icon="🌟")
    log.level(name="WARNING", color="<yellow>", icon="🔥")
    log.level(name="ERROR", color="<red>", icon="❌")
    log.level(name="CRITICAL", color="<magenta>", icon="💀")
    log.level(name="HEADING", color="<bold><white>", no=50)

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
        format="<level>{message}</level>",
        level="INFO",
        enqueue=True,
    )


config_logger()

# test_funct()
log.info(
    f"\n#=====================================================#\n‖{'Skombo time 🤠':^52}‖\n#=====================================================#\n",
)

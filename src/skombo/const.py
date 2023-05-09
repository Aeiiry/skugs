"""Skug combo constants."""
import logging
import re

NUMERIC_COLUMNS: list[str] = [
    "damage",
    "chip_damage",
    "meter_on_hit",
    "meter_on_whiff",
    "on_hit",
    "on_block",
    "startup",
    "active",
    "recovery",
    "hitstun",
    "blockstun",
    "hitstop",
    "on_pushblock",
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


CHARACTERS_TO_REMOVE: list[str] = ["+", "\n", "±"]
# Put the list of characters to remove in the regex below in a character class ([]), escaping any special characters
RE_CHARACTERS_TO_REMOVE: re.Pattern[str] = re.compile(
    r"[" + re.escape("".join(CHARACTERS_TO_REMOVE)) + r"]"
)


UNIVERSAL_MOVE_CATEGORIES = {
    "TAG IN": "tag",
    "SNAPBACK": "snap",
    "THROW": "throw",
    "AIR THROW": "air_throw",
    "TAUNT": "taunt",
    "ASSIST RECOVERY": "assist_recovery",
}


FRAME_VALUE_REPLACEMENTS: list[str] = ["+", "±"]
FRAME_VALUE_REPLACEMENTS_STR = f"[{''.join(FRAME_VALUE_REPLACEMENTS)}]"
FRAME_VALUE_REPLACEMENTS_RE: re.Pattern[str] = re.compile(FRAME_VALUE_REPLACEMENTS_STR)

UNDIZZY_MAX = 240

SCALING_FACTOR = 0.875

SCALING_MIN_1K = 0.275
SCALING_MIN = 0.2

SCALING_DHC = 0.7

SCALING_ASSIST_START = 0.66
SCALING_ASSIST_START_COUNTER = 0.9

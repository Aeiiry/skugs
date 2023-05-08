"""Skug combo constants."""
import logging
import re
from typing import Literal, Type

# flake8: noqa: E501


ALT_PREFIX: Literal["alt_"] = "alt_"
"""Prefix for alternate move data"""

# Column names
CHARACTER_NAME: Literal["Character"] = "Character"
MOVE_NAME: Literal["MoveName"] = "MoveName"
ALT_NAMES: Literal["AltNames"] = "AltNames"
DAMAGE: Literal["Damage"] = "Damage"

EXPECTED_DAMAGE: Literal["ExpectedDamage"] = "ExpectedDamage"

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

# Column names for combo data
HIT_NUMBER: Literal["HitNumber"] = "HitNumber"
DAMAGE_SCALING: Literal["DamageScaling"] = "DamageScaling"
SCALED_DAMAGE: Literal["ScaledDamage"] = "ScaledDamage"
UNDIZZY: Literal["Undizzy"] = "Undizzy"
TOTAL_DAMAGE_FOR_MOVE: Literal["TotalDamageForMove"] = "TotalDamageForMove"
TOTAL_DAMAGE_FOR_COMBO: Literal["TotalDamageForCombo"] = "TotalDamageForCombo"

UNDIZZY_DICT: dict[str, int] = {
    "Light": 15,
    "Medium": 30,
    "Heavy": 40,
    "Special": 30,
    "Throws+Supers": 0,
}
"""Undizzy values for each hit type.
"""

# Move names to automatically ignore
IGNORED_MOVES: list[str] = [
    "adc",
    "air dash cancel",
    "air dash",
    "delay",
    "delayed",
    "delaying",
    "jc",
    "jump cancel",
    "jump",
    "otg",
    "dash",
    "66",
    "restand",
]
""" List of move names to ignore when searching for moves in the combo data.
"""

SEARCH_STATES: dict[str, bool] = {
    "character_specific": False,
    "repeat": False,
    "start": False,
    "follow_up": False,
    "alias": False,
    "generic": False,
    "no_strength": False,
    "not_found": False,
}

ANNIE_DIVEKICK: str = "RE ENTRY"

HATRED_INSTALL: str = "(during hi)"

LOG_FILE_SUFFIX = "skugLog"
LOG_FOLEDER_NAME = "logs"
LOG_FILE_EXT = ".log"
LOG_LEVEL_CONSOLE: int = logging.INFO
LOG_LEVEL_FILE: int = logging.DEBUG

RE_IN_BRACKETS: re.Pattern[str] = re.compile(r"\[[^\[]*]")
RE_IN_PAREN: re.Pattern[str] = re.compile(r"\((.*?)\)")
RE_X_N: re.Pattern[str] = re.compile(r"(\d*?\.?\d+\s?)([x*]\s?)(\d+)")
RE_BRACKETS_X_N: re.Pattern[str] = re.compile(r"(\[[\d,\s]*?]\s?)([x*])(\d+)")
RE_ANY: re.Pattern[str] = re.compile(r".*")

CHARACTERS_TO_REMOVE: list[str] = ["+", "\n", "±"]
# Put the list of characters to remove in the regex below in a character class ([]), escaping any special characters
RE_CHARACTERS_TO_REMOVE: re.Pattern[str] = re.compile(
    r"[" + re.escape("".join(CHARACTERS_TO_REMOVE)) + r"]"
)


RE_STR_BETWEEN_WHITESPACE: re.Pattern[str] = re.compile(r"^\s*(.*?)\s*$")

FD_COLUMNS_TO_MOVE_ATTR_DICT: dict[str, str] = {
    "character": "character",
    "move_name": "name",
    "alt_names": "alt_names",
    "guard": "guard",
    "properties": "properties",
    "startup": "startup",
    "active": "active",
    "recovery": "recovery",
    "on_block": "on_block",
    "on_hit": "on_hit",
    "footer": "notes",
    "damage": "hits_str",
    "meter": "meter_gain_loss",
    "hitstun": "hitstun",
    "blockstun": "blockstun",
    "hitstop": "hitstop",
}

UNIVERSAL_MOVE_CATEGORIES = {
    "TAG IN": "tag",
    "SNAPBACK": "snap",
    "THROW": "throw",
    "AIR THROW": "air_throw",
    "TAUNT": "taunt",
    "ASSIST RECOVERY": "assist_recovery",
}
MOVE_PROPERTY_IDEAL_TYPES: dict[str, list[Type]] = {  # type: ignore
    "character": [str],
    "name": [str],
    "alt_names": [list[str]],
    "guard": [str, list[str]],
    "properties": [str, list[str]],
    "damage": [list[int]],
    "chip_damage": [list[int]],
    "on_hit": [int],
    "on_block": [int],
    "startup": [int],
    "active": [int],
    "recovery": [int],
    "hitstun": [int],
    "blockstun": [int],
    "hitstop": [int],
    "meter_gain_loss": [int],
    "notes": [str],
    "hits_str": [str],
    "hits": [list],
    "hits_alt": [dict],
    "category": [str],
}

FRAME_VALUE_FIELDS = [
    "startup",
    "active",
    "recovery",
    "on_block",
    "on_hit",
    "hitstop",
    "meter_gain_loss",
]

FRAME_VALUE_REPLACEMENTS: list[str] = ["+", "±"]
FRAME_VALUE_REPLACEMENTS_STR = f"[{''.join(FRAME_VALUE_REPLACEMENTS)}]"
FRAME_VALUE_REPLACEMENTS_RE: re.Pattern[str] = re.compile(FRAME_VALUE_REPLACEMENTS_STR)

UNDIZZY_VALUES: dict[str, int] = {
    "light": 15,
    "medium": 20,
    "special": 20,
    "heavy": 30,
}

UNDIZZY_MAX = 240

SCALING_FACTOR = 0.875

SCALING_MIN_1K = 0.275
SCALING_MIN = 0.2

SCALING_DHC = 0.7

SCALING_ASSIST_START = 0.66
SCALING_ASSIST_START_COUNTER = 0.9

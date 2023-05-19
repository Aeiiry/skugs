from typing import Any
from skombo import LOG
import pandas as pd
import functools
import skombo
import re

def filter_dict(
    dict: dict[str, Any], colfilter: str | list[str], filter_values: bool = False
) -> dict[str, Any]:
    """
    Return a dict excluding the keys in the filter param\n
    Retains order so isn't exceptionally fast\n
    filter_values = True will filter values instead, only supports string values\n
    """

    colfilter = list(colfilter)
    if filter_values:
        return {
            k: v for k, v in dict.items() if not isinstance(v, str) or v not in colfilter
        }
    else:
        return {k: v for k, v in dict.items() if k not in colfilter}


def format_column_headings(df: pd.DataFrame) -> pd.DataFrame:
    """Format column headings to lowercase with underscores"""
    df.columns = [col.replace(" ", "_").lower() for col in df.columns]
    return df


@functools.cache
def attempt_to_int(value: str | int) -> str | int:
    return int(value) if isinstance(value, str) and value.isnumeric() else value


@functools.cache
def remove_spaces(string: str) -> str:
    return string.replace(" ", "") if " " in string else string


def split_on_char(string: str, char: str, strip: bool = True) -> list[str]:
    if char not in string:
        return [string]
    split_string: list[str] = string.split(char)
    if strip:
        split_string = [remove_spaces(x) for x in split_string]

    return split_string


@functools.cache
def expand_all_x_n(damage: str) -> str:
    if isinstance(damage, str):
        if " " in damage:
            damage = damage.replace(" ", "")
        while (x_n_match := skombo.RE_X_N.search(damage)) or (
            x_n_match := skombo.RE_BRACKETS_X_N.search(damage)
        ):
            damage = expand_x_n(x_n_match)

    return damage


@functools.cache
def expand_x_n(match: re.Match[str]) -> str:
    num = int(match.group(3))
    damage: str = match.group(1).strip()
    original_damage: str = match.group(0)
    if "[" in original_damage:
        damage = re.sub(r"[\[\]]", "", original_damage).replace(" ", "")
        expanded_list: list[str] = damage.split(",") * num
        expanded_damage: str = ",".join(expanded_list)
    else:
        expanded_damage = ",".join([damage] * num)
    return (
        match.string[: match.start()] + expanded_damage + match.string[match.end() :]
        if match.end()
        else match.string[: match.start()] + expanded_damage
    )

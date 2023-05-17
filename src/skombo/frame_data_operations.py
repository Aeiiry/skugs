# sourcery skip: lambdas-should-be-short
"""Main module for frame data operations."""
import functools
import os
import re
from collections import abc
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas import Index

import skombo
from skombo import CHARS

log = skombo.log

DataFrame = pd.DataFrame

global fd


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


COLS = Columns()


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

    NUMERIC_LIST_COLUMNS = [
        COLS.dmg,
        COLS.chip,
        COLS.startup,
        COLS.active,
        COLS.recovery,
        COLS.hitstun,
        COLS.blockstun,
        COLS.hitstop,
        COLS.blockstop,
        COLS.super_hitstop,
        COLS.undizzy,
    ]


COL_CLASS = ColumnClassification()


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


def apply_to_columns(
        frame_data: DataFrame,
        func: abc.Callable,  # type: ignore
        columns: list[str],
        non_nan: bool = False,
) -> DataFrame:
    if non_nan:
        # apply function to non-nan cells in specified columns
        frame_data[columns] = frame_data[columns].applymap(
            lambda x: func(x) if pd.notna(x) else x
        )
    else:
        # apply function to all cells in specified columns
        frame_data[columns] = frame_data[columns].applymap(func)
    return frame_data


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
        match.string[: match.start()] + expanded_damage + match.string[match.end():]
        if match.end()
        else match.string[: match.start()] + expanded_damage
    )


def separate_meter(frame_data: DataFrame) -> DataFrame:
    """Separate meter into on_hit and on_whiff"""
    fd_meter = frame_data.loc[:, COLS.meter]
    # fd_meter_old = fd_meter.copy()

    on_hit, on_whiff = zip(*fd_meter.apply(split_meter))
    # on_hit and on_whiff are tuples, convert to lists
    on_hit = list(on_hit)
    on_whiff = list(on_whiff)
    # Split on_hit and on_whiff's values on commas if they are strings
    on_hit = [split_on_char(x, ",") if isinstance(x, str) else x for x in on_hit]
    on_whiff = [split_on_char(x, ",") if isinstance(x, str) else x for x in on_whiff]

    # Strip whitespace from on_hit and on_whiff and remove empty strings
    on_hit = [
        [remove_spaces(y) for y in x if y != "" and y is not None]
        if isinstance(x, list)
        else x
        for x in on_hit
    ]

    # Insert new columns into frame_data to the right of meter

    # Assign on_hit and on_whiff to the new columns
    frame_data[COLS.meter] = on_hit
    frame_data[COLS.meter_whiff] = on_whiff

    return frame_data


def split_meter(meter: str) -> tuple[str | None, str | None]:
    if isinstance(meter, str) and (search := skombo.RE_IN_PAREN.search(meter)):
        on_whiff: str | Any = search.group(1)
    else:
        on_whiff = None
    on_hit: str | None = (
        meter.replace(f"({on_whiff})", "") if isinstance(meter, str) else None
    )
    return on_hit, on_whiff


def separate_on_hit(frame_data: DataFrame) -> DataFrame:
    frame_data[COLS.onhit_eff] = frame_data[COLS.onhit].copy()
    frame_data = apply_to_columns(
        frame_data,
        lambda x: x
        if (isinstance(x, str) and x.isnumeric()) or isinstance(x, int)
        else None,
        [COLS.onhit],
    )
    frame_data = apply_to_columns(
        frame_data,
        lambda x: x if isinstance(x, str) and not x.isnumeric() else None,
        [COLS.onhit_eff],
    )

    return frame_data


def categorise_moves(df: DataFrame) -> DataFrame:
    """Categorise moves into different types"""
    # Dict of move names that each character has 1 of
    universal_move_categories: dict[str, str] = skombo.UNIVERSAL_MOVE_CATEGORIES
    df[COLS.move_cat] = df.index.get_level_values(1).map(universal_move_categories)

    re_normal_move: re.Pattern[str] = skombo.RE_NORMAL_MOVE

    normal_strengths: dict[str, str] = skombo.NORMAL_STRENGTHS
    # Normal moves are where move_name matches the regex and the move_category is None, we can use the regex to find the strength of the move by the first group

    # Make a mask of the rows that are normal moves, by checking against df.index.get_level_values(1)

    mask: Index = df.index.get_level_values(1).map(
        lambda x: isinstance(x, str) and re_normal_move.search(x) is not None
    )
    # Assign the move_category to the strength of the move
    df.loc[mask, COLS.move_cat] = (  # type: ignore
        df.loc[mask]
        .index.get_level_values(1)
        .map(lambda x: normal_strengths[re_normal_move.search(x).group(1)] + "_NORMAL")  # type: ignore
    )

    # Supers are where meter_on_hit has length 1 and meter_on_hit[0] is  -100 or less
    df.loc[
        df.astype(str)[COLS.meter].str.contains(r"-\d\d", regex=True, na=False)
        & df[COLS.move_cat].isna(),
        COLS.move_cat,
    ] = "SUPER"
    # For now, assume everything else is a special
    # TODO: Add more special move categories for things like double's level 5 projectiles, annie taunt riposte etc

    df.loc[df[COLS.move_cat].isna(), COLS.move_cat] = "SPECIAL_MOVE"

    return df


def insert_alt_name_aliases(frame_data: DataFrame) -> DataFrame:
    aliases: DataFrame = pd.read_csv(skombo.MOVE_NAME_ALIASES_PATH).dropna()

    # create dictionary from aliases dataframe
    alias_dict = dict(zip(aliases["Key"], aliases["Value"].str.replace("\n", ",")))

    # create a pandas Series from alt_names column using the map method
    alt_names_series = frame_data["alt_names"].map(
        lambda x: re.sub(
            rf"\b({'|'.join(alias_dict.keys())})\b",  # type: ignore
            lambda m: alias_dict.get(m.group(0)),  # type: ignore
            x,  # type: ignore
        )
        if isinstance(x, str)
        else x
    )

    # replace alt_names column with the new pandas Series
    frame_data["alt_names"] = alt_names_series

    return frame_data


def add_undizzy_values(df: DataFrame) -> DataFrame:
    """Add undizzy values to the dataframe"""

    undizzy_dict: dict[str, int] = skombo.UNDIZZY_DICT

    # Create a new column for undizzy values
    df["undizzy"] = df[COLS.move_cat].map(undizzy_dict)

    return df


@functools.cache
def str_to_int(x: Any) -> int | Any:
    if isinstance(x, str):
        x = x.strip()
        if "-" in x:
            return 1 - int(numx) if (numx := x.replace("-", "")).isnumeric() else x
        else:
            return int(x) if x.isnumeric() else x
    return x


def clean_frame_data(frame_data: DataFrame) -> DataFrame:
    """
    Args:
        frame_data: DataFrame containing the data to be cleaned

    Returns:
        DataFrame containing the cleaned data in a more readable format
    """

    log.info("========== Cleaning frame data ==========")
    frame_data = insert_alt_name_aliases(frame_data)
    log.info("Inserted alt name aliases from macros .csv")

    log.info("=== Initial string cleaning ===")
    frame_data = initial_string_cleaning(frame_data)

    frame_data = separate_annie_stars(frame_data)

    frame_data = separate_damage_chip_damage(frame_data)

    frame_data = separate_meter(frame_data)
    log.info(
        "Separated [meter] column into [meter_on_hit] and [meter_on_whiff] columns"
    )
    numeric_columns = COL_CLASS.NUMERIC_COLUMNS
    frame_data[numeric_columns] = frame_data[numeric_columns].applymap(str_to_int)
    log.info(f"Converted numeric columns to integers: {numeric_columns}")

    numeric_list_columns = COL_CLASS.NUMERIC_LIST_COLUMNS
    frame_data[numeric_list_columns] = frame_data[numeric_list_columns].applymap(
        lambda x: [str_to_int(y) for y in x.split(",")] if pd.notnull(x) else x
    )
    log.info(
        f"Converted numeric list columns to lists of integers: {numeric_list_columns}"
    )

    frame_data = separate_on_hit(frame_data)
    log.info(
        "Separated [on_hit] column into [on_hit_advantage] and [on_hit_effect] columns"
    )

    frame_data = categorise_moves(frame_data)
    log.info("Added categories to moves")

    frame_data = add_undizzy_values(frame_data)
    log.info("Adding undizzy values to moves")
    return frame_data


def initial_string_cleaning(frame_data: DataFrame) -> DataFrame:
    """
    Initial string operations to clean the data.

    Args:
        frame_data: DataFrame with columns to clean

    Returns:
        DataFrame with columns that are cleaned after being sent to Son
    """
    # Replace any individual cell that only contain "-" with np.nan
    frame_data = frame_data.replace("-", np.nan)
    log.info("Replaced [-] and [ ] with [NaN]")
    frame_data = frame_data.replace("", np.nan)
    log.info("Replaced empty strings with [NaN]")
    # Remove characters from columns that are not needed

    # Remove newlines from relevant columns
    remove_newline_cols = COL_CLASS.REMOVE_NEWLINE_COLS
    frame_data.loc[:, remove_newline_cols] = frame_data.loc[
                                             :, remove_newline_cols
                                             ].replace("\n", "")

    log.info(f"Removed newlines from columns: {remove_newline_cols}")

    # Remove + and ± from relevant columns (\u00B1 is the unicode for ±)
    plus_minus_cols = COL_CLASS.PLUS_MINUS_COLS
    re_plus_plusminus = r"\+|\u00B1"
    frame_data.loc[:, plus_minus_cols] = frame_data.loc[:, plus_minus_cols].replace(
        re_plus_plusminus, "", regex=True
    )

    log.info(f"Removed [+] and [±] from columns: {plus_minus_cols}")

    # Remove percentage symbol from meter column
    frame_data[COLS.meter] = frame_data[COLS.meter].str.replace("%", "")
    log.info("Removed [%] from [meter] column")

    # Split alt_names by newline
    frame_data["alt_names"] = frame_data["alt_names"].str.replace("\n", ",")
    log.info("Split [alt_names] by [\\n]")

    # Split footer by dash, drop empty strings, and strip whitespace
    frame_data["footer"] = frame_data["footer"].str.findall(r"(?:-\s)([^-]+)")
    log.info("Split [footer] by [-]")

    # Expand all x_n notation in damage, hitstun, blockstun, hitstop, meter, and active columns
    x_n_cols = [
        COLS.dmg,
        COLS.hitstun,
        COLS.blockstun,
        COLS.hitstop,
        COLS.meter,
        COLS.active,
    ]

    frame_data[x_n_cols] = frame_data[x_n_cols].applymap(
        lambda x: expand_all_x_n(x) if "x" in str(x) else x
    )
    log.info(f"Expanded [x_n] notation in columns: {x_n_cols}")
    return frame_data


def separate_damage_chip_damage(frame_data: DataFrame) -> DataFrame:
    """
    Separate damage and chip_damage columns into one column

    Args:
        frame_data: Dataframe to be processed by function

    Returns:
        New dataframe with chip_damage column seper
    """
    # Extract values in parentheses from damage column and put them in chip_damage column
    frame_data.loc[:, COLS.chip] = frame_data.loc[:, COLS.dmg].str.extract(r"\((.*)\)")[
        0
    ]
    # Remove the values in parentheses from damage column
    frame_data.loc[:, COLS.dmg] = frame_data.loc[:, COLS.dmg].str.replace(
        r"\(.*\)", "", regex=True
    )

    frame_data.loc[:, COLS.dmg] = (
        frame_data.loc[:, COLS.dmg].str.strip().replace(r",,", ",", regex=True)
    )
    frame_data.loc[:, COLS.dmg] = (
        frame_data.loc[:, COLS.dmg]
        .str.strip()
        .replace(r"^,|,$", "", regex=True)
        # .str.split(",")
    )

    frame_data.loc[:, COLS.chip] = (
        frame_data.loc[:, COLS.chip].str.strip().replace(r",,", ",", regex=True)
    )
    frame_data.loc[:, COLS.chip] = (
        frame_data.loc[:, COLS.chip]
        .str.strip()
        .replace(r"^,|,$", "", regex=True)
        # .str.split(",")
    )

    "]'"

    log.info("Separated [damage] column into [damage] and [chip_damage] columns")
    return frame_data


def separate_annie_stars(frame_data: DataFrame) -> DataFrame:
    """
    Separate annie stars into one or more sets of damage and on_block

    Args:
        frame_data: Dataframe to be analysed.

    Returns:
        A DataFrame with a column for each row in the data
    """

    # First select rows of index ANNIE and just the  damage and on_block
    star_rows: DataFrame = frame_data.loc[(CHARS.AN, slice(None)), [COLS.dmg, COLS.onblock]]  # type: ignore
    log.info("Splitting Annie's normals into star power and original versions")

    # Filter out rows without a "[" in damage or on_block
    star_rows = star_rows[
        (star_rows[COLS.dmg].notna() & star_rows[COLS.dmg].str.contains(r"\["))
        | (
                star_rows[COLS.onblock].notna()
                & star_rows[COLS.onblock].str.contains(r"\[")
        )
        ]

    orig_rows: DataFrame = star_rows.copy()

    # Remove the content inside the brackets for the original rows
    orig_rows.loc[:, COLS.dmg] = orig_rows[COLS.dmg].str.replace(
        r"\[.*\]", "", regex=True
    )
    orig_rows.loc[:, COLS.onblock] = orig_rows[COLS.onblock].str.replace(
        r"\[.*\]", "", regex=True
    )

    # Remove brackets for star power damage
    star_rows.loc[:, COLS.dmg] = star_rows[COLS.dmg].str.replace(
        r"\[|\]", "", regex=True
    )
    # Extract the content inside the brackets for star power on_block
    star_rows.loc[:, COLS.onblock] = star_rows[COLS.onblock].str.extract(
        r"\[(.*)\]", expand=False
    )

    # convert star rows to have the same columns as frame_data
    star_rows = star_rows.reindex(columns=frame_data.columns)

    # Fill the nan values with corresponding values from the original rows
    star_rows = star_rows.fillna(frame_data.loc[orig_rows.index])
    # Add _star_power to index 2 to differentiate between the original and star power rows
    star_rows.index = star_rows.index.set_levels(  # type: ignore
        star_rows.index.levels[1] + "_STAR_POWER", level=1  # type: ignore
    )

    # log.info(f"Star power extracted and converted for {tabulate(star_rows.index)}")

    # Modify original rows to have the orig_rows damage and on_block values
    frame_data.loc[orig_rows.index, orig_rows.columns] = orig_rows

    # Add the star power rows to the original frame_data, interleave the rows with the original rows
    frame_data = pd.concat([frame_data, star_rows])  # type: ignore

    return frame_data


def format_column_headings(df: DataFrame) -> DataFrame:
    """
    Formats column headings to make it easier to read.

    Args:
        df: The DataFrame to be formatted. Should be of type DataFrame

    Returns:
        The DataFrame with formatted
    """
    log.debug("Formatting column headings")
    log.debug(f"Original column headings: {df.columns}")
    df_lower_cols: list[str] = [col.replace(" ", "_").lower() for col in df.columns]
    df.columns = df_lower_cols  # type: ignore
    log.debug(f"Formatted column headings: {df.columns}")
    return df


@functools.cache
def get_fd_bot_data() -> DataFrame:
    log.info("========== Getting frame data bot data ==========")
    log.info("Loading csvs into dataframes")
    log.info(f"Currect working directory: {os.getcwd()}")
    fd: DataFrame = extract_fd_from_csv()
    return fd


@functools.cache
def extract_fd_from_csv() -> DataFrame:
    log.info("========== Extracting frame data from fd bot csv ==========")
    # We don't need existing index column
    with open(skombo.CHARACTER_DATA_PATH, "r", encoding="utf8") as characters_file:
        characters_df: DataFrame = format_column_headings(
            pd.read_csv(characters_file, encoding="utf8")
        )

    with open(skombo.FRAME_DATA_PATH, "r", encoding="utf8") as frame_file:
        frame_data: DataFrame = format_column_headings(
            pd.read_csv(frame_file, encoding="utf8")
        )

    log.info("Loaded csvs into dataframes")

    # == Clean up move_name column before using it as an index ==
    frame_data[COLS.m_name] = (
        frame_data[COLS.m_name].str.strip().replace(r"\n", "", regex=True)
    )
    # == Set the index to character and move_name ==

    frame_data = frame_data.set_index([COLS.char, COLS.m_name], verify_integrity=True)
    # Name the index
    frame_data.index.names = [COLS.char, COLS.m_name]

    rename_cols = {
        "on_block": COLS.onblock,
        "meter": COLS.meter,
        "on_hit": COLS.onhit,
    }
    frame_data.rename(columns=rename_cols, inplace=True)
    cols_minus_index: list[str] = list(COLS.__dict__.values())[2:]
    frame_data = pd.DataFrame(data=frame_data, columns=cols_minus_index)

    frame_data = clean_frame_data(frame_data)

    # Export as csv

    frame_data.to_csv("fd_cleaned.csv")
    log.info("Exported cleaned frame data to csv: [fd_cleaned.csv]")
    log.info("========== Finished extracting frame data from fd bot csv ==========")

    return frame_data

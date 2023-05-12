# sourcery skip: lambdas-should-be-short
"""Main module for frame data operations."""
import atexit
import functools
import os
import re
from collections import abc
from typing import Any

import numpy as np
import pandas as pd
from pandas import Index
from tabulate import tabulate

import skombo.const as const
import skombo.file_man as fm
from skombo import sklog as sklog

log = sklog.get_logger()

DataFrame = pd.DataFrame
global fd

fd = None


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
        while (x_n_match := const.RE_X_N.search(damage)) or (
            x_n_match := const.RE_BRACKETS_X_N.search(damage)
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
        match.string[: match.start()] + expanded_damage + match.string[match.end() :]
        if match.end()
        else match.string[: match.start()] + expanded_damage
    )


def separate_meter(frame_data: DataFrame) -> DataFrame:
    """Separate meter into on_hit and on_whiff"""
    fd_meter = frame_data.loc[:, "meter"]
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
    frame_data = add_new_columns_at_column(
        frame_data, "meter", ["meter_on_hit", "meter_on_whiff"], copy_values=True
    )
    # Assign on_hit and on_whiff to the new columns
    frame_data["meter_on_hit"] = on_hit
    frame_data["meter_on_whiff"] = on_whiff

    return frame_data


def split_meter(meter: str) -> tuple[str | None, str | None]:
    if isinstance(meter, str) and (search := const.RE_IN_PAREN.search(meter)):
        on_whiff: str | Any = search.group(1)
    else:
        on_whiff = None
    on_hit: str | None = (
        meter.replace(f"({on_whiff})", "") if isinstance(meter, str) else None
    )
    return on_hit, on_whiff


def separate_on_hit(frame_data: DataFrame) -> DataFrame:
    frame_data = add_new_columns_at_column(
        frame_data, "on_hit", ["on_hit_advantage", "on_hit_effect"], copy_values=True
    )
    frame_data["on_hit_effect"] = frame_data["on_hit_advantage"].copy()
    frame_data = apply_to_columns(
        frame_data,
        lambda x: x
        if (isinstance(x, str) and x.isnumeric()) or isinstance(x, int)
        else None,
        ["on_hit_advantage"],
    )
    frame_data = apply_to_columns(
        frame_data,
        lambda x: x if isinstance(x, str) and not x.isnumeric() else None,
        ["on_hit_effect"],
    )

    return frame_data


def categorise_moves(df: DataFrame) -> DataFrame:
    """Categorise moves into different types"""
    # Dict of move names that each character has 1 of
    log.debug("Categorising moves")
    universal_move_categories: dict[str, str] = const.UNIVERSAL_MOVE_CATEGORIES
    df["move_category"] = df.index.get_level_values(1).map(universal_move_categories)

    re_normal_move = const.RE_NORMAL_MOVE

    normal_strengths: dict[str, str] = {
        "l": "LIGHT",
        "m": "MEDIUM",
        "h": "HEAVY",
    }
    # Normal moves are where move_name matches the regex and the move_category is None, we can use the regex to find the strength of the move by the first group

    df.loc[:, "move_category"] = df.loc[:, "move_name"].apply(
        lambda x: normal_strengths[search.group(1).lower()] + "_NORMAL"
        if isinstance(x, str)
        and (search := re_normal_move.search(x))
        and search.groups().__len__() > 0
        else np.nan
    )

    # Supers are where meter_on_hit has length 1 and meter_on_hit[0] is  -100 or less
    df.loc[
        df["meter_on_hit"].apply(
            # meter_on_hit contains a lot of float values represented as strings
            lambda x: isinstance(x, list)
            and len(x) >= 1
            and x[0][0] == "-"
            and int(x[0]) <= -100
        ),
        "move_category",
    ] = "SUPER"

    # For now, assume everything else is a special
    # TODO: Add more special move categories for things like double's level 5 projectiles, annie taunt riposte etc

    df.loc[df["move_category"].isna(), "move_category"] = "SPECIAL_MOVE"

    return df


def insert_alt_name_aliases(frame_data: DataFrame) -> DataFrame:
    aliases: DataFrame = pd.read_csv(fm.MOVE_NAME_ALIASES_PATH).dropna()

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

    undizzy_dict: dict[str, int] = const.UNDIZZY_DICT

    # Create a new column for undizzy values
    df["undizzy"] = df["move_category"].map(undizzy_dict)

    return df


@functools.cache
def convert_numeric(x: Any) -> int | Any:
    return int(x) if isinstance(x, str) and x.isnumeric() else x


def clean_frame_data(frame_data: DataFrame) -> DataFrame:
    """
    Args:
        frame_data: DataFrame containing the data to be cleaned

    Returns:
        DataFrame containing the cleaned data in a more readable format
    """

    log.debug("Cleaning frame data")
    log.debug("Inserting alt name aliases")
    frame_data = insert_alt_name_aliases(frame_data)

    log.debug("Initial string cleaning")
    frame_data = initial_string_cleaning(frame_data)

    # Turn empty strings into np.nan
    frame_data = frame_data.replace("", np.nan)

    log.debug("Separating Annie stars")
    frame_data = separate_annie_stars(frame_data)

    log.debug("Separating damage and chip damage")
    frame_data = separate_damage_chip_damage(frame_data)

    log.debug("Separating meter into on_hit and on_whiff")
    frame_data = separate_meter(frame_data)

    log.debug("Converting negative numbers to int")
    numeric_columns = const.NUMERIC_COLUMNS

    log.debug("Converting numbers in strings to int")

    frame_data[numeric_columns] = frame_data[numeric_columns].convert_dtypes()

    log.debug("Separating on_hit into on_hit_advantage and on_hit_effect")
    frame_data = separate_on_hit(frame_data)

    log.debug("Adding summed damage columns")
    frame_data["summed_damage"] = frame_data["damage"].apply(
        lambda x: sum(x)
        if isinstance(x, list) and all(isinstance(d, int) for d in x)
        else x
    )
    frame_data["summed_chip_damage"] = frame_data["chip_damage"].apply(
        lambda x: sum(x)
        if isinstance(x, list) and all(isinstance(d, int) for d in x)
        else x
    )

    log.debug("Categorising moves")
    frame_data = categorise_moves(frame_data)

    log.debug("Adding undizzy values")
    frame_data = add_undizzy_values(frame_data)
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

    # Remove characters from columns that are not needed
    columns_to_remove_chars: list[str] = frame_data.columns.tolist()
    columns_to_remove_chars.remove("alt_names")
    # Remove characters from all string columns
    frame_data[columns_to_remove_chars] = frame_data[columns_to_remove_chars].apply(
        lambda x: x.str.replace(const.RE_CHARACTERS_TO_REMOVE, "", regex=True)
        if x.dtype == "object"
        else x
    )

    # Remove percentage symbol from meter column
    frame_data["meter"] = frame_data["meter"].str.replace("%", "")

    # Split alt_names by newline
    frame_data["alt_names"] = frame_data["alt_names"].str.replace("\n", ",")

    # Split footer by dash, drop empty strings, and strip whitespace
    frame_data["footer"] = frame_data["footer"].str.findall(r"(?:-\s)([^-]+)")

    # Expand all x_n notation in damage, hitstun, blockstun, hitstop, meter, and active columns
    x_n_cols = ["damage", "hitstun", "blockstun", "hitstop", "meter", "active"]
    frame_data[x_n_cols] = frame_data[x_n_cols].applymap(
        lambda x: expand_all_x_n(x) if "x" in str(x) else x
    )

    # find if there are any lists in the damage column that are not equal in length to the meter_on_hit column for that row

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
    frame_data.loc[:, "chip_damage"] = frame_data.loc[:, "damage"].str.extract(
        r"\((.*)\)"
    )[0]
    # Remove the values in parentheses from damage column
    frame_data.loc[:, "damage"] = frame_data.loc[:, "damage"].str.replace(
        r"\(.*\)", "", regex=True
    )
    log.debug(f'\n{frame_data.loc[frame_data.loc[:, "chip_damage"].notna(),["damage","chip_damage"]]}')

    "]'"
    log.info(f"========== # of rows with chip damage: {frame_data['chip_damage'].count()} ==========")
    # log dtypes


    return frame_data


def add_new_columns_at_column(
    frame_data: DataFrame,
    old_columns: str | list[str],
    new_columns: str | list[str],
    offset: int = 1,
    copy_values: bool = False,
) -> DataFrame:
    """
    Add new columns to a DataFrame in place of old columns, leaving values in the old columns if any of the names match the new columns names

    Args:
        frame_data: DataFrame to add new columns to
        old_columns: old columns to be used as reference for index and/or data
        new_columns: names of new columns
        offset: Offset to insert the new columns
        copy_values: Copy values from old columns to new columns, if there are more new columns than old columns, the values will be copied to next empty column (1 old, 2 new -> 2 new with values from old in first column)

    Returns:
        DataFrame with new columns added to its reference columns ( index
    """
    if isinstance(old_columns, str):
        old_columns_list: list[str] = [old_columns]
    else:
        old_columns_list = old_columns  # type: ignore
    if isinstance(new_columns, str):
        new_columns_list: list[str] = [new_columns]
    else:
        new_columns_list = new_columns  # type: ignore

    # Don't update old_columns if any of the new columns are in the old columns
    # Otherwise, the old columns will be overwritten

    new_columns_index: int = (
        frame_data.columns.tolist().index(old_columns_list[-1]) + offset
    )

    # Insert the new columns at the last index of the old columns
    # This is to ensure that the new columns are added after the old columns

    for i, new_column in enumerate(new_columns_list):
        log.debug(f"Adding new column {new_column} to dataframe")
        if new_column not in old_columns_list:
            # Copy values from old columns to new columns if copy_values is True and values exist in the old columns for the current index
            new_column_values: pd.Series[Any] | float = (
                frame_data[old_columns_list[i]]
                if copy_values and i < len(old_columns_list)
                else np.nan
            )
            frame_data.insert(new_columns_index, new_column, new_column_values)
            new_columns_index += 1

    for old_column in old_columns_list:
        frame_data.drop(
            columns=old_column, inplace=True
        ) if old_column not in new_columns_list else None
    log.debug(f"Columns after adding new columns: {frame_data.columns.tolist()}")
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
    star_rows: DataFrame = frame_data.loc[("ANNIE", slice(None)), ["damage", "on_block"]]  # type: ignore
    log.info("Splitting Annie's normals into star power and original versions")

    # Filter out rows without a "[" in damage or on_block
    star_rows = star_rows[
        (star_rows["damage"].notna() & star_rows["damage"].str.contains(r"\["))
        | (star_rows["on_block"].notna() & star_rows["on_block"].str.contains(r"\["))
    ]

    orig_rows: DataFrame = star_rows.copy()

    # Remove the content inside the brackets for the original rows
    orig_rows.loc[:, "damage"] = orig_rows["damage"].str.replace(
        r"\[.*\]", "", regex=True
    )
    orig_rows.loc[:, "on_block"] = orig_rows["on_block"].str.replace(
        r"\[.*\]", "", regex=True
    )

    # Remove brackets for star power damage
    star_rows.loc[:, "damage"] = star_rows["damage"].str.replace(
        r"\[|\]", "", regex=True
    )
    # Extract the content inside the brackets for star power on_block
    star_rows.loc[:, "on_block"] = star_rows["on_block"].str.extract(
        r"\[(.*)\]", expand=False
    )

    # convert star rows to have the same columns as frame_data
    star_rows = star_rows.reindex(columns=frame_data.columns)

    # Fill the nan values with corresponding values from the original rows
    star_rows = star_rows.fillna(frame_data.loc[orig_rows.index])
    # Add _star_power to index 2 to differentiate between the original and star power rows
    star_rows.index = star_rows.index.set_levels(  # type: ignore
        star_rows.index.levels[1] + "_star_power", level=1  # type: ignore
    )

    log.info(
        f"Star power extracted and converted for {tabulate(star_rows.index)}"
    )

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
    with open(fm.CHARACTER_DATA_PATH, "r", encoding="utf8") as characters_file:
        characters_df: DataFrame = format_column_headings(
            pd.read_csv(characters_file, encoding="utf8").convert_dtypes()
        )

    with open(fm.FRAME_DATA_PATH, "r", encoding="utf8") as frame_file:
        frame_data: DataFrame = format_column_headings(
            pd.read_csv(frame_file, encoding="utf8").convert_dtypes()
        )
    atexit.register(frame_data.to_csv, "frame_data.csv")

    log.info("Loaded csvs into dataframes")

    frame_data = frame_data.set_index(["character", "move_name"], verify_integrity=True)
    # Name the index
    frame_data.index.names = ["character", "move_name"]

    frame_data = add_new_columns_at_column(
        frame_data, "damage", ["damage", "chip_damage"]
    )

    frame_data.info(verbose=True, memory_usage="deep")
    frame_data = clean_frame_data(frame_data)

    frame_data.info(buf=log.debug, verbose=True, memory_usage="deep")

    log.info("========== Data extracted and cleaned ==========")

    """     # Get some stats about the data
    log.debug(f"Number of rows in frame_data: {frame_data.shape[0]}")
    log.debug(f"Number of columns in frame_data: {frame_data.shape[1]}")

    # Value counts for eachc column
    for column in frame_data.columns:
        log.debug(f"Value counts for column {column}:")
        log.debug(f"\n\n{frame_data[column].value_counts(dropna=False)}\n\n") """

    # Export as csv on program exit

    return frame_data

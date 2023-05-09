# sourcery skip: lambdas-should-be-short
"""Main module for frame data operations."""
import cProfile
import os
import pstats
import re
from collections import abc
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

import skug_fd_parse.constants as const
import skug_fd_parse.file_management as fm
from skug_fd_parse.skug_logger import log

DataFrame = pd.DataFrame
global frame_data

frame_data = None


def attempt_to_int(value: str | int) -> str | int:
    return int(value) if isinstance(value, str) and value.isnumeric() else value


def remove_spaces(string: str) -> str:
    return string.replace(" ", "") if " " in string else string


def split_on_char(string: str, char: str, strip: bool = True) -> list[str]:
    if char not in string:
        return [string]
    split_string: list[str] = string.split(char)
    if strip:
        split_string = [remove_spaces(x) for x in split_string]

    return split_string


def expand_all_x_n(damage: str) -> str:
    if isinstance(damage, str):
        if " " in damage:
            damage = damage.replace(" ", "")
        while re.search(r"\d+\s?[x*]\s?\d+", damage):
            if x_n_match := const.RE_X_N.search(damage):
                damage = expand_x_n(x_n_match)
            elif x_n_brackets_matches := const.RE_BRACKETS_X_N.search(damage):
                damage = expand_x_n(x_n_brackets_matches)
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
    df["move_category"] = df["move_name"].map(universal_move_categories)

    re_normal_move: re.Pattern[str] = re.compile(
        r"^j?\.?\d?.?([lmh])[pk]", flags=re.IGNORECASE
    )
    """ regex to find normal moves """

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

    # replace substrings using map method with dictionary
    frame_data["alt_names"] = frame_data["alt_names"].map(
        lambda x: re.sub(
            rf"\b({'|'.join(alias_dict.keys())})\b",  # type: ignore
            lambda m: alias_dict.get(m.group(0)),  # type: ignore
            x,  # type: ignore
        )
        if isinstance(x, str)
        else x
    )

    return frame_data


def add_undizzy_values(df: DataFrame) -> DataFrame:
    """Add undizzy values to the dataframe"""

    undizzy_dict: dict[str, int] = const.UNDIZZY_DICT

    # Create a new column for undizzy values
    df["undizzy"] = df["move_category"].map(undizzy_dict)

    return df


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

    frame_data[numeric_columns] = frame_data[numeric_columns].applymap(convert_numeric)

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
    frame_data["alt_names"] = frame_data["alt_names"].str.split("\n|,")

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
    # Create a new column for chip damage
    # If the damage column is a string, get the value between the parentheses, this is the chip damage

    # apply the function to the 'damage' column and assign the result to a new 'chip_damage' column

    # Create a boolean mask of rows containing parentheses in 'damage' column
    paren_mask = frame_data["damage"].str.contains(r"\(").fillna(False)

    # Extract the content inside parentheses using regex
    chip_damage = frame_data.loc[paren_mask, "damage"].str.extract(r"\((\d+)\)")

    # Fill NaN values with 0
    chip_damage = chip_damage.fillna(0).astype(int)

    # Assign extracted values to 'chip_damage' column
    frame_data.loc[paren_mask, "chip_damage"] = chip_damage

    # Create a dictionary of functions to apply to the damage and chip_damage columns
    function_column_dict: dict[abc.Callable, list[str]] = {  # type: ignore
        lambda d: d[: d.find("(")]
        if isinstance(d, str) and "(" in d
        else d: ["damage"],
        lambda x: [
            int(d.strip()) if d.strip().isnumeric() else d
            for d in (x.split(","))
            if d.strip() not in ["", ","]
        ]
        if isinstance(x, str)
        else x: ["damage", "chip_damage"],
    }
    for func, columns in function_column_dict.items():
        frame_data = apply_to_columns(frame_data, func, columns)

    # log dtypes
    log.debug("Data types after separating damage and chip damage")
    log.debug(frame_data.dtypes)

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
    # Locate all rows that have a star power, star power is annie exclusive and is in []
    # These rows will have a damage and on_block value that is a list
    star_power_annie_rows = frame_data[
        (
            (frame_data["damage"].apply(lambda x: isinstance(x, str) and "[" in x))
            | (frame_data["on_block"].apply(lambda x: isinstance(x, str) and "[" in x))
        )
        & (frame_data["character"] == "Annie")
    ]
    # Insert copies of the rows that have star power
    # original_rows_copy: DataFrame = star_power_annie_rows.copy()

    # log the rows that have star power, just move_name, damage and on_block
    log.debug("////////// Rows that have star power //////////")
    log.debug(f"\n{star_power_annie_rows[['move_name', 'damage', 'on_block']]}")

    # Probably isn't too slow to iterate through unique move names to get pairs of rows
    # Get unique move names from star_power_annie_rows
    move_names = star_power_annie_rows["move_name"].unique()

    # Get the single string damage and on_block values for each move name
    damage_values = star_power_annie_rows[
        star_power_annie_rows["move_name"].isin(move_names)
    ]["damage"]
    star_damage_values = damage_values.str.replace(r"[\[\]]", "", regex=True)
    on_block_values = star_power_annie_rows[
        star_power_annie_rows["move_name"].isin(move_names)
    ]["on_block"]
    star_on_block_values = on_block_values.str.extract(r"\[(.*)]").fillna("").iloc[:, 0]

    # Remove stars from original damage and on_block values
    damage_values = damage_values.str.replace(r"\[.*]", "", regex=True)
    on_block_values = on_block_values.str.replace(r"\[.*]", "", regex=True)

    # Update damage and on_block values for original rows
    frame_data.loc[frame_data["move_name"].isin(move_names), "damage"] = damage_values
    frame_data.loc[
        frame_data["move_name"].isin(move_names), "on_block"
    ] = on_block_values

    # Update damage and on_block values for new rows
    star_power_annie_rows.loc[
        star_power_annie_rows["move_name"].isin(move_names), "damage"
    ] = star_damage_values
    star_power_annie_rows.loc[
        star_power_annie_rows["move_name"].isin(move_names), "on_block"
    ] = star_on_block_values

    # Update move_name for new rows
    star_power_annie_rows.loc[
        star_power_annie_rows["move_name"].isin(move_names), "move_name"
    ] = (move_names + "_STAR_POWER")

    frame_data = pd.concat([frame_data, star_power_annie_rows], ignore_index=True)

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


def capitalise_words(name: str) -> str:
    """
    Capitalise the first letter of each word in a string
    """
    return (
        " ".join([word.capitalize() for word in name.split(" ")])
        if pd.notnull(name)
        else name
    )


def get_fd_bot_data() -> DataFrame:
    log.info("========== Starting skug_stats ==========")
    log.info("Loading csvs into dataframes")
    log.info(f"Currect working directory: {os.getcwd()}")
    global frame_data
    if frame_data is None:
        frame_data = extract_fd_from_csv()

    return frame_data


# TODO Rename this here and in `get_fd_bot_data`
def extract_fd_from_csv() -> DataFrame:
    log.info("========== Extracting frame data from fd bot csv ==========")

    with open(fm.CHARACTER_DATA_PATH, "r", encoding="utf8") as characters_file:
        characters_df: DataFrame = format_column_headings(
            pd.read_csv(characters_file, encoding="utf8")
        )

    with open(fm.FRAME_DATA_PATH, "r", encoding="utf8") as frame_file:
        result: DataFrame = format_column_headings(
            pd.read_csv(frame_file, encoding="utf8")
        )

    log.info("Loaded csvs into dataframes")

    characters_df["character"] = characters_df["character"].apply(capitalise_words)
    result["character"] = result["character"].apply(capitalise_words)

    global sg_characters
    sg_characters = characters_df["character"].tolist()  # type: ignore
    result = add_new_columns_at_column(result, "damage", ["damage", "chip_damage"])

    result = clean_frame_data(result)

    log.info("========== Data extracted and cleaned ==========")

    # Get some stats about the data
    log.debug(f"Number of rows in frame_data: {result.shape[0]}")
    log.debug(f"Number of columns in frame_data: {result.shape[1]}")

    # Value counts for eachc column
    for column in result.columns:
        log.debug(f"Value counts for column {column}:")
        log.debug(f"\n\n{result[column].value_counts(dropna=False)}\n\n")

        # Export as csv
    try:
        result.to_csv("output.csv", index=False)
    except PermissionError:
        log.error("Could not export to csv, ensure output.csv is not open")
    else:
        log.info("Exported to csv")
    return result

# sourcery skip: lambdas-should-be-short
"""Main module for frame data operations."""
import cProfile
import os
import pstats
import re
from collections import abc
from typing import Any, Literal
import numpy as np

import pandas as pd
import strictly_typed_pandas
from strictly_typed_pandas.dataset import DataSet
from tabulate import tabulate

import skug_fd_parse.constants as const
import skug_fd_parse.file_management as fm
from skug_fd_parse.skug_logger import log


class FrameDataSchema:
    """Initial Schema for frame data.
    Eventually schema will be more strict, only str and int for now"""

    character: str
    move_name: str
    alt_names: str
    guard: str
    properties: str
    damage: str
    chip_damage: str
    # meter: int
    on_hit: int
    on_block: int
    startup: int
    active: int
    recovery: int
    hitstun: int
    blockstun: int
    hitstop: int
    on_pushblock: int
    footer: str
    thumbnail_url: str
    footer_url: str
    meter_on_hit: float | int | None
    meter_on_whiff: float | int | None


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


def expand_all_x_n(damage: Any) -> Any:
    original_damage = damage
    if isinstance(damage, str):
        while True:
            if x_n_match := const.RE_X_N.search(damage):
                damage = expand_x_n(x_n_match)
            elif x_n_brackets_matches := const.RE_BRACKETS_X_N.search(damage):
                damage = expand_x_n(x_n_brackets_matches)
            else:
                break
    if damage != original_damage and pd.notna(damage):
        log.debug(f"Expanded [{original_damage}] to [{damage}]")
    return damage


def apply_to_columns(
    frame_data: pd.DataFrame, func: abc.Callable, columns: list[str]  # type: ignore
) -> pd.DataFrame:
    for column in columns:
        frame_data[column] = frame_data[column].apply(func)
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


def separate_meter(frame_data: pd.DataFrame) -> pd.DataFrame:
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


def check_frame_data_types(df: pd.DataFrame) -> None:
    try:
        DataSet[FrameDataSchema](df)
    except TypeError as e:
        log.warning("Frame data types are incorrect")
        log.warning(e)
    else:
        log.debug("Frame data types are correct")


def clean_frame_data(frame_data: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        frame_data: pd.DataFrame containing the data to be cleaned

    Returns:
        pd.DataFrame containing the cleaned data in a more readable format
    """

    log.debug("Cleaning frame data")

    log.debug("Initial string cleaning")
    frame_data = initial_string_cleaning(frame_data)

    log.debug("Separating Annie stars")
    frame_data = separate_annie_stars(frame_data)

    log.debug("Separating damage and chip damage")
    frame_data = separate_damage_chip_damage(frame_data)

    log.debug("Separating meter into on_hit and on_whiff")
    frame_data = separate_meter(frame_data)

    log.debug("Converting negative numbers to int")
    numeric_columns: list[str] = [
        "damage",
        "chip_damage",
        "meter_on_hit",
        "meter_on_whiff",
    ]

    frame_data = apply_to_columns(
        frame_data,
        lambda x: int(x) if isinstance(x, str) and "-" in x and x.isnumeric() else x,
        numeric_columns,
    )

    damage_meter_on_hit = frame_data.loc[
        frame_data["damage"].apply(lambda x: isinstance(x, list))
        & frame_data["meter_on_hit"].apply(lambda x: isinstance(x, list))
        & frame_data["meter_on_hit"].apply(
            lambda x: isinstance(x, list)
            and (
                False
                not in [isinstance(d, str) and len(d) > 0 and d[0] != "-" for d in x]
            )
        )
        & (~frame_data["move_name"].str.contains("STAR_POWER"))
    ]

    damage_meter_on_hit = damage_meter_on_hit.loc[
        damage_meter_on_hit["damage"].apply(lambda x: len(x))
        != damage_meter_on_hit["meter_on_hit"].apply(lambda x: len(x))
    ]
    # Drop all columns except move_name, character, damage, meter_on_hit, and append a column to display the difference
    damage_meter_on_hit = damage_meter_on_hit[
        [
            "move_name",
            "character",
            "damage",
            "meter_on_hit",
        ]
    ]

    log.debug("Checking frame data types")

    check_frame_data_types(frame_data)

    return frame_data


def initial_string_cleaning(frame_data: pd.DataFrame) -> pd.DataFrame:
    """
    Initial string operations to clean the data.

    Args:
        frame_data: pd.DataFrame with columns to clean

    Returns:
        pd.DataFrame with columns that are cleaned after being sent to Son
    """
    columns_to_remove_chars: list[str] = frame_data.columns.tolist()
    if "alt_names" in columns_to_remove_chars:
        columns_to_remove_chars.remove("alt_names")
        columns_to_remove_chars.remove("damage")
        columns_to_remove_chars.remove("chip_damage")

    function_column_dict: dict[abc.Callable, list[str]] = {  # type: ignore
        lambda x: const.RE_CHARACTERS_TO_REMOVE.sub("", x)
        if isinstance(x, str)
        else x: columns_to_remove_chars,
        lambda x: x.replace("%", "") if isinstance(x, str) else x: ["meter"],
        lambda x: x.split("\n") if isinstance(x, str) else x: ["alt_names"],
        lambda x: split_on_char(x, "- ", False)[1:]
        if isinstance(x, str)
        else x: ["footer"],
        expand_all_x_n: ["damage", "meter"],
    }
    for func, columns in function_column_dict.items():
        frame_data = apply_to_columns(frame_data, func, columns)

    # find if there are any lists in the damage column that are not equal in length to the meter_on_hit column for that row

    return frame_data


def separate_damage_chip_damage(frame_data: pd.DataFrame) -> pd.DataFrame:
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
    frame_data["chip_damage"] = (
        frame_data["damage"]
        .where(frame_data["damage"].apply(lambda x: isinstance(x, str) and "(" in x))
        .apply(lambda x: x[x.find("(") + 1 : x.find(")")] if isinstance(x, str) else x)
    )

    # Create a dictionary of functions to apply to the damage and chip_damage columns
    function_column_dict: dict[abc.Callable, list[str]] = {  # type: ignore
        # Similar to above, but replace the value with the value before the parentheses
        lambda d: d[: d.find("(")]
        if isinstance(d, str) and "(" in d
        else d: ["damage"],
        # Split the value by commas, and convert each value to an integer if possible
        lambda x: [
            int(d.strip()) if d.strip().isnumeric() else d for d in (x.split(","))
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
    frame_data: pd.DataFrame,
    old_columns: str | list[str],
    new_columns: str | list[str],
    offset: int = 1,
    copy_values: bool = False,
) -> pd.DataFrame:
    """
    Add new columns to a pd.DataFrame in place of old columns, leaving values in the old columns if any of the names match the new columns names

    Args:
        frame_data: pd.DataFrame to add new columns to
        new_columns: Dictionary of reference columns to add to
        offset: Offset to insert the new columns
        copy_values: Copy values from old columns to new columns, if there are more new columns than old columns, the values will be copied to next empty column (1 old, 2 new -> 2 new with values from old in first column)

    Returns:
        pd.DataFrame with new columns added to its reference columns ( index
    """
    if isinstance(old_columns, str):
        old_columns_list: list[str] = [old_columns]
    else:
        old_columns_list = old_columns  # type: ignore
    if isinstance(new_columns, str):
        new_columns_list: list[str] = [new_columns]
    else:
        new_columns_list = new_columns  # type: ignore

    log.info(
        f"Adding new {new_columns_list} to dataframe in place of {old_columns_list}"
    )
    log.info(f"Columns before adding new columns: {frame_data.columns.tolist()}")

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


def separate_annie_stars(frame_data: pd.DataFrame) -> pd.DataFrame:
    """
    Separate annie stars into one or more sets of damage and on_block

    Args:
        frame_data: Dataframe to be analysed.

    Returns:
        A pd.DataFrame with a column for each row in the data
    """
    # Locate all rows that have a star power, star power is annie exclusive and is in []
    # These rows will have a damage and on_block value that is a list
    star_power_annie_rows: pd.DataFrame = frame_data[
        (
            frame_data["damage"].apply(lambda x: isinstance(x, str) and "[" in x)
            | frame_data["on_block"].apply(lambda x: isinstance(x, str) and "[" in x)
        )
        & (frame_data["character"] == "Annie")
    ]  # type: ignore

    # log the rows that have star power, just move_name, damage and on_block
    log.debug("////////// Rows that have star power //////////")
    log.debug(f"\n{star_power_annie_rows[['move_name', 'damage', 'on_block']]}")

    # Create a copy of the rows that have star power
    original_annie_rows: pd.DataFrame = star_power_annie_rows.copy()

    re_stars: re.Pattern[str] = const.RE_ANNIE_STARS
    re_any: re.Pattern[str] = const.RE_ANY

    # Search for the star power in the damage and on_block columns, this will return a list of matches.
    # re_any is there to avoid a type error when the value is not a string or not a match
    star_damage = original_annie_rows["damage"].apply(
        lambda x: re_stars.search(x) or re_any.search(x)  # type: ignore
    )
    star_on_block = original_annie_rows["on_block"].apply(
        lambda x: re_stars.search(x) or re_any.search(x)  # type: ignore
    )
    # Replace the damage and on_block columns with the values before the star power
    original_annie_rows.loc[:, "damage"] = original_annie_rows.loc[:, "damage"].where(
        pd.Series(not bool(match) for match in star_damage),
        pd.Series(
            match.group(1) + match.group(4)
            if match and match.groups().__len__() > 3
            else match.group(1)
            if match and match.groups().__len__() > 0
            else match.string
            for match in star_damage
        ),
    )
    original_annie_rows.loc[:, "on_block"] = original_annie_rows.loc[
        :, "on_block"
    ].where(
        pd.Series((not bool(match)) for match in star_on_block),
        pd.Series(
            match.group(1) if match.groups().__len__() > 0 else match.string
            for match in star_on_block
        ),
    )
    # Same as above, but replace the values with the star power values
    star_power_annie_rows.loc[:, "on_block"] = star_power_annie_rows.loc[
        :, "on_block"
    ].where(
        pd.Series((not bool(match)) for match in star_on_block),
        pd.Series(
            match.group(3) if match.groups().__len__() > 2 else match.string
            for match in star_on_block
        ),
    )

    star_power_annie_rows.loc[:, "damage"] = star_power_annie_rows.loc[
        :, "damage"
    ].where(
        pd.Series(not bool(match) for match in star_damage),
        pd.Series(
            "".join(match.groups()) if match.groups() else match.string
            for match in star_damage
        ),
    )

    # Modify the move name to include the star power for the star power rows
    star_power_annie_rows.loc[:, "move_name"] = star_power_annie_rows.loc[
        :, "move_name"
    ].apply(lambda name: name + "_STAR_POWER" if isinstance(name, str) else name)

    # Reset the index of the original and star power rows
    original_annie_rows = original_annie_rows.reset_index(drop=True)
    star_power_annie_rows = star_power_annie_rows.reset_index(drop=True)

    # logging
    log.debug("////////// Separated star power rows //////////")
    log.debug(f"\n{star_power_annie_rows[['move_name', 'damage', 'on_block']]}")
    log.debug("////////// Separated original rows //////////")
    log.debug(f"\n{original_annie_rows[['move_name', 'damage', 'on_block']]}")

    # Combine the original and star power rows
    combined_annie: pd.DataFrame = pd.concat(
        [original_annie_rows, star_power_annie_rows]
    ).sort_index()

    # Drop the original rows from the frame_data and replace them with the combined rows
    frame_data = frame_data.drop(original_annie_rows.index)

    frame_data = pd.concat([combined_annie, frame_data]).sort_index()

    return frame_data


def format_column_headings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats column headings to make it easier to read.

    Args:
        df: The pd.DataFrame to be formatted. Should be of type pd.DataFrame

    Returns:
        The pd.DataFrame with formatted
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


def main() -> Literal[1, 0]:
    """Main"""
    log.info("========== Starting skug_stats ==========")
    log.info("Loading csvs into dataframes")
    log.info(f"Currect working directory: {os.getcwd()}")

    with open(fm.CHARACTER_DATA_PATH, "r", encoding="utf8") as characters_file:
        characters_df = format_column_headings(
            pd.read_csv(characters_file, encoding="utf8")
        )

    with open(fm.FRAME_DATA_PATH, "r", encoding="utf8") as frame_file:
        frame_data: pd.DataFrame = format_column_headings(
            pd.read_csv(frame_file, encoding="utf8")
        )

    log.info("Loaded csvs into dataframes")

    characters_df["character"] = characters_df["character"].apply(capitalise_words)
    frame_data["character"] = frame_data["character"].apply(capitalise_words)

    frame_data = add_new_columns_at_column(
        frame_data, "damage", ["damage", "chip_damage"]
    )

    frame_data = clean_frame_data(frame_data)

    # Get some stats about the data
    log.debug(f"Number of rows in frame_data: {frame_data.shape[0]}")
    log.debug(f"Number of columns in frame_data: {frame_data.shape[1]}")

    # Value counts for eachc column
    for column in frame_data.columns:
        log.debug(f"Value counts for column {column}:")
        log.debug(f"\n\n{frame_data[column].value_counts(dropna=False)}\n\n")

    # Export as csv
    try:
        frame_data.to_csv("output.csv", index=False)
    except PermissionError:
        log.error("Could not export to csv, ensure output.csv is not open")
        return 1
    else:
        log.info("Exported to csv")
    log.info("========== Finished skug_stats ==========")
    stats = pstats.Stats(pr)
    stats.dump_stats("skug_stats.prof")
    # stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(25)
    return 0


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()

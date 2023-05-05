# sourcery skip: lambdas-should-be-short
"""Main module for frame data operations."""
import cProfile
import os
import pstats
import re
from collections import abc
from typing import Any, Literal

import pandas as pd
import strictly_typed_pandas
from strictly_typed_pandas.dataset import DataSet

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
    meter: int
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


def attempt_to_int(value: str | int) -> str | int:
    """
    Attempts to convert a string to an integer. If the string is numeric it is returned as - is.

    Args:
        value: The value to attempt to convert.

    Returns:
        The converted value or the original value if it couldn't be converted
    """
    return int(value) if isinstance(value, str) and value.isnumeric() else value


def remove_spaces(string: str) -> str:
    """
    Removes spaces from a string. This is useful for stripping spaces from text that is sent to Sphinx.

    Args:
        string: The string to remove spaces from. Can be a string or anything that can be converted to a string.

    Returns:
        The string with spaces removed or the original string if nothing was
    """
    return string.replace(" ", "") if " " in string else string


def split_on_char(string: str, char: str, strip: bool = True) -> str | list[str]:
    if char not in string:
        return string
    split_string: list[str] = string.split(char)
    if strip:
        split_string = [remove_spaces(x) for x in split_string]

    return split_string


def expand_all_x_n(damage: str) -> str:
    while True:
        if x_n_match := const.RE_X_N.search(damage):
            log.debug(f"Expanding all x_n in '{damage}'")
            damage = expand_x_n(x_n_match)
        elif x_n_brackets_matches := const.RE_BRACKETS_X_N.search(damage):
            log.debug(f"Expanding all x_n in '{damage}'")
            damage = expand_x_n(x_n_brackets_matches)
        else:
            break
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
        expanded_damage = ",".join(expanded_list)
    else:
        expanded_damage = ",".join([damage] * num)
    return (
        match.string[: match.start()] + expanded_damage + match.string[match.end() :]
        if match.end()
        else match.string[: match.start()] + expanded_damage
    )


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

    log.debug("Converting negative numbers to int")

    numberic_columns: list[str] = ["damage", "chip_damage", "meter"]

    frame_data = apply_to_columns(
        frame_data,
        lambda x: int(x) if isinstance(x, str) and "-" in x and x.isnumeric() else x,
        numberic_columns,
    )

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
        columns_to_remove_chars.remove("footer")
        columns_to_remove_chars.remove("damage")
        columns_to_remove_chars.remove("chip_damage")

    function_column_dict: dict[abc.Callable, list[str]] = {  # type: ignore
        lambda x: x.split("-"): ["footer"],
        lambda x: const.RE_CHARACTERS_TO_REMOVE.sub("", x)
        if isinstance(x, str)
        else x: columns_to_remove_chars,
        lambda x: x.split("\n"): ["alt_names"],
        lambda x: split_on_char(x, "- ", False): ["properties"],
        expand_all_x_n: ["damage", "meter"],
    }
    for func, columns in function_column_dict.items():
        frame_data = apply_to_columns(frame_data, func, columns)

    return frame_data


def separate_damage_chip_damage(frame_data: pd.DataFrame) -> pd.DataFrame:
    """
    Separate damage and chip_damage columns into one column

    Args:
        frame_data: Dataframe to be processed by function

    Returns:
        New dataframe with chip_damage column seper
    """
    frame_data["chip_damage"] = frame_data["damage"].apply(
        lambda d: d[d.find("(") + 1 : d.find(")")] if isinstance(d, str) else d
    )
    function_column_dict: dict[abc.Callable, list[str]] = {  # type: ignore
        lambda d: d[: d.find("(")] if isinstance(d, str) else d: ["damage"],
        lambda x: [
            int(d.strip()) if d.strip().isnumeric() else d for d in (x.split(","))
        ]: ["damage", "chip_damage"],
    }
    for func, columns in function_column_dict.items():
        frame_data = apply_to_columns(frame_data, func, columns)

    # log dtypes
    log.debug("Data types after separating damage and chip damage")
    log.debug(frame_data.dtypes)

    return frame_data


def add_new_columns(
    frame_data: pd.DataFrame, new_columns: dict[str, str], offset: int = 1
) -> pd.DataFrame:
    """
    Add new columns to a pd.DataFrame. This is a helper function to make it easier to use in conjunction with pd.DataFrame. reindex

    Args:
        frame_data: pd.DataFrame to add new columns to
        new_columns: Dictionary of reference columns to add to
        offset: Offset to insert the new columns

    Returns:
        pd.DataFrame with new columns added to its reference columns ( index
    """
    log.debug(f"Adding new columns {new_columns} to pd.DataFrame")
    for reference_column, new_column in new_columns.items():
        frame_data_columns: list[str] = frame_data.columns.tolist()

        old_index = frame_data_columns.index(reference_column)
        frame_data_columns.insert(old_index + offset, new_column)

        frame_data = frame_data.reindex(columns=frame_data_columns, fill_value=None)

    return frame_data


def separate_annie_stars(frame_data: pd.DataFrame) -> pd.DataFrame:
    """
    Separate annie stars into one or more sets of damage and on_block

    Args:
        frame_data: Dataframe to be analysed.

    Returns:
        A pd.DataFrame with a column for each row in the data
    """
    star_power_annie_rows: pd.DataFrame = frame_data[
        (
            frame_data["damage"].apply(lambda x: isinstance(x, str) and "[" in x)
            | frame_data["on_block"].apply(lambda x: isinstance(x, str) and "[" in x)
        )
        & (frame_data["character"] == "Annie")
    ]  # type: ignore

    original_annie_rows: pd.DataFrame = star_power_annie_rows.copy()
    row: pd.Series[Any]
    re_stars = const.RE_ANNIE_STARS
    re_any = const.RE_ANY
    star_damage = original_annie_rows["damage"].apply(
        lambda x: re_stars.search(x) or re_any.search(x)  # type: ignore
    )
    star_on_block = original_annie_rows["on_block"].apply(
        lambda x: re_stars.search(x) or re_any.search(x)  # type: ignore
    )

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
    star_power_annie_rows.loc[:, "move_name"] = star_power_annie_rows.loc[
        :, "move_name"
    ].apply(lambda name: name + "_STAR_POWER" if isinstance(name, str) else name)

    original_annie_rows = original_annie_rows.reset_index(drop=True)
    star_power_annie_rows = star_power_annie_rows.reset_index(drop=True)

    combined_annie: pd.DataFrame = pd.concat(
        [original_annie_rows, star_power_annie_rows]
    ).sort_index()

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
    log.debug(f"Capitalising name {name}")
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

    new_columns_dict: dict[str, str] = {"damage": "chip_damage"}

    frame_data = add_new_columns(frame_data, new_columns_dict)

    # convert columns to string
    frame_data = frame_data.astype(str).fillna("-")

    frame_data = clean_frame_data(frame_data)

    # Get some stats about the data
    log.debug(f"Number of rows in frame_data: {frame_data.shape[0]}")
    log.debug(f"Number of columns in frame_data: {frame_data.shape[1]}")

    # Value counts
    log.debug("Value counts")
    for column in frame_data.columns:
        log.debug(f"Column: {column}")
        log.debug(frame_data[column].value_counts())

    # Export as csv
    try:
        frame_data.to_csv("output.csv", index=False)
    except PermissionError:
        log.error("Could not export to csv, ensure output.csv is not open")
        return 1
    else:
        log.info("Exported to csv")
    log.info("========== Finished skug_stats ==========")
    profiler.dump_stats("stats.prof")
    stats = pstats.Stats(profiler)

    stats.sort_stats("tottime").print_stats(10)
    stats.sort_stats("cumtime").print_stats(10)
    return 0


if __name__ == "__main__":
    with cProfile.Profile() as profiler:
        main()

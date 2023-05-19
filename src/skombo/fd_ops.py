# sourcery skip: lambdas-should-be-short
"""Main module for frame data operations."""
import fnmatch
import functools
import os
import re
from collections import abc
from dataclasses import dataclass
from typing import Any, Self
from skombo.utils import (
    filter_dict,
    remove_spaces,
    split_on_char,
    expand_all_x_n,
    remove_spaces,
    format_column_headings,
)
import numpy as np
import pandas as pd
from pandas import Index

import skombo
from skombo import CHARS, LOG


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


def separate_meter(frame_data: pd.DataFrame) -> pd.DataFrame:
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


def separate_on_hit(frame_data: pd.DataFrame) -> pd.DataFrame:
    frame_data[COLS.onhit_eff] = frame_data[COLS.onhit].copy()
    frame_data[COLS.onhit] = frame_data[COLS.onhit].apply(
        lambda x: x  # type: ignore
        if (isinstance(x, str) and x.isnumeric()) or isinstance(x, int)
        else None
    )

    frame_data[COLS.onhit_eff] = frame_data[COLS.onhit_eff].apply(
        lambda x: x if isinstance(x, str) and not x.isnumeric() else None  # type: ignore
    )

    return frame_data


def categorise_moves(df: pd.DataFrame) -> pd.DataFrame:
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


def add_undizzy_values(df: pd.DataFrame) -> pd.DataFrame:
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


def clean_frame_data(frame_data: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        frame_data: DataFrame containing the data to be cleaned

    Returns:
        DataFrame containing the cleaned data in a more readable format
    """

    LOG.info("========== Cleaning frame data ==========")
    LOG.info("Inserted alt name aliases from macros .csv")

    LOG.info("=== Initial string cleaning ===")
    frame_data = initial_string_cleaning(frame_data)

    frame_data = separate_annie_stars(frame_data)

    frame_data = separate_damage_chip_damage(frame_data)

    frame_data = separate_meter(frame_data)
    LOG.info(
        "Separated [meter] column into [meter_on_hit] and [meter_on_whiff] columns"
    )
    numeric_columns = COL_CLASS.NUMERIC_COLUMNS
    frame_data[numeric_columns] = frame_data[numeric_columns].applymap(str_to_int)
    LOG.info(f"Converted numeric columns to integers: {numeric_columns}")

    numeric_list_columns = COL_CLASS.NUMERIC_LIST_COLUMNS
    frame_data[numeric_list_columns] = frame_data[numeric_list_columns].applymap(
        lambda x: [str_to_int(y) for y in x.split(",")] if pd.notnull(x) else x
    )
    LOG.info(
        f"Converted numeric list columns to lists of integers: {numeric_list_columns}"
    )

    frame_data = separate_on_hit(frame_data)
    LOG.info(
        "Separated [on_hit] column into [on_hit_advantage] and [on_hit_effect] columns"
    )

    frame_data = categorise_moves(frame_data)
    LOG.info("Added categories to moves")

    frame_data = add_undizzy_values(frame_data)
    LOG.info("Adding undizzy values to moves")
    return frame_data


def initial_string_cleaning(frame_data: pd.DataFrame) -> pd.DataFrame:
    """
    Initial string operations to clean the data.

    Args:
        frame_data: DataFrame with columns to clean

    Returns:
        DataFrame with columns that are cleaned after being sent to Son
    """
    # Replace any individual cell that only contain "-" with np.nan
    frame_data = frame_data.replace("-", np.nan)
    LOG.info("Replaced [-] and [ ] with [NaN]")
    frame_data = frame_data.replace("", np.nan)
    LOG.info("Replaced empty strings with [NaN]")
    # Remove characters from columns that are not needed

    # Remove newlines from relevant columns
    remove_newline_cols = COL_CLASS.REMOVE_NEWLINE_COLS
    frame_data.loc[:, remove_newline_cols] = frame_data.loc[
        :, remove_newline_cols
    ].replace("\n", "")

    LOG.info(f"Removed newlines from columns: {remove_newline_cols}")

    # Remove + and ± from relevant columns (\u00B1 is the unicode for ±)
    plus_minus_cols = COL_CLASS.PLUS_MINUS_COLS
    re_plus_plusminus = r"\+|\u00B1"
    frame_data.loc[:, plus_minus_cols] = frame_data.loc[:, plus_minus_cols].replace(
        re_plus_plusminus, "", regex=True
    )

    LOG.info(f"Removed [+] and [±] from columns: {plus_minus_cols}")

    # Remove percentage symbol from meter column
    frame_data[COLS.meter] = frame_data[COLS.meter].str.replace("%", "")
    LOG.info("Removed [%] from [meter] column")

    # Split alt_names by newline
    frame_data["alt_names"] = frame_data["alt_names"].str.replace("\n", ",")
    LOG.info("Split [alt_names] by [\\n]")

    # Split footer by dash, drop empty strings, and strip whitespace
    frame_data["footer"] = frame_data["footer"].str.findall(r"(?:-\s)([^-]+)")
    LOG.info("Split [footer] by [-]")

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
    LOG.info(f"Expanded [x_n] notation in columns: {x_n_cols}")
    return frame_data


def separate_damage_chip_damage(frame_data: pd.DataFrame) -> pd.DataFrame:
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

    LOG.info("Separated [damage] column into [damage] and [chip_damage] columns")
    return frame_data


def separate_annie_stars(frame_data: pd.DataFrame) -> pd.DataFrame:
    """
    Separate annie stars into one or more sets of damage and on_block

    Args:
        frame_data: Dataframe to be analysed.

    Returns:
        A DataFrame with a column for each row in the data
    """

    # First select rows of index ANNIE and just the  damage and on_block
    star_rows: pd.DataFrame = frame_data.loc[(CHARS.AN, slice(None)), [COLS.dmg, COLS.onblock]]  # type: ignore
    LOG.info("Splitting Annie's normals into star power and original versions")

    # Filter out rows without a "[" in damage or on_block
    star_rows = star_rows[
        (star_rows[COLS.dmg].notna() & star_rows[COLS.dmg].str.contains(r"\["))
        | (
            star_rows[COLS.onblock].notna()
            & star_rows[COLS.onblock].str.contains(r"\[")
        )
    ]

    orig_rows: pd.DataFrame = star_rows.copy()

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


@functools.cache
def get_fd_bot_data() -> pd.DataFrame:
    LOG.info("========== Getting frame data bot data ==========")
    LOG.info("Loading csvs into dataframes")
    LOG.info(f"Currect working directory: {os.getcwd()}")
    fd: pd.DataFrame = extract_fd_from_csv()
    return fd


class CsvManager:
    """Class to manage CSV files"""

    def __init__(self, path: str, file_keys: dict[str, str]) -> None:
        self.path: str = path
        self.file_keys: dict[str, str] = file_keys
        self.dataframes: dict[str, pd.DataFrame] = {}
        """Raw dataframes from csvs before they are modified"""
        for key in self.file_keys:
            self.dataframes[key] = self.open_csv(key)

    def open_csv(self, file_key: str) -> pd.DataFrame:
        """Open a CSV file and return it as a DataFrame"""
        file_ends_with: str = self.file_keys[file_key] + ".csv"

        file_found: list[str] = fnmatch.filter(
            os.listdir(self.path), f"*{file_ends_with}"
        )

        if not file_found:
            raise FileNotFoundError(
                f"Could not find {file_key} CSV in {self.path} with ending {file_ends_with}"
            )

        file_path: str = os.path.join(self.path, file_found[0])

        if file_found[0].lower() != file_ends_with:
            os.rename(file_path, os.path.join(self.path, file_ends_with))
            file_path = os.path.join(self.path, file_ends_with)

        with open(file_path, "r", encoding="utf8") as file:
            df: pd.DataFrame = pd.read_csv(file, encoding="utf8")
            return format_column_headings(df)


class FdBotCsvManager(CsvManager):
    """Class to manage the frame data bot CSV files"""

    file_keys: dict[str, str] = {
        "characters": "characters",
        "aliases": "macros",
        "frame_data": "moves",
    }

    def __init__(self, path: str = skombo.GAME_DATA_PATH) -> None:
        """Initialise the class"""

        super().__init__(path, self.file_keys)


FD_BOT_CSV_MANAGER = FdBotCsvManager()


class FrameData(pd.DataFrame):
    """DataFrame subclass for frame data"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def re_index(self) -> Self:
        self[COLS.m_name] = self[COLS.m_name].str.strip().replace(r"\n", "", regex=True)
        self.set_index([COLS.char, COLS.m_name], verify_integrity=True, inplace=True)
        self.index.names = [COLS.char, COLS.m_name]
        return self

    def rename_cols(self):
        rename_cols: dict[str, str] = {
            "on_block": COLS.onblock,
            "meter": COLS.meter,
            "on_hit": COLS.onhit,
        }
        self.rename(columns=rename_cols, inplace=True)

        cols_minus_index: list[str] = list(
            filter_dict(COLS.__dict__, self.index.names, filter_values=True).values()
        )
        self = FrameData(data=self, columns=cols_minus_index)
        self.fillna(np.nan, inplace=True)
        return self


FD = FrameData(FD_BOT_CSV_MANAGER.dataframes["frame_data"]).re_index().rename_cols()


@functools.cache
def extract_fd_from_csv() -> pd.DataFrame:
    LOG.info("========== Extracting frame data from fd bot csv ==========")
    # We don't need existing index column
    frame_data = FD
    LOG.info("Copied frame data from fd bot csv")

    LOG.info("Loaded csvs into dataframes")

    frame_data = clean_frame_data(frame_data)

    frame_data.to_csv("fd_cleaned.csv")

    LOG.info("Exported cleaned frame data to csv: [fd_cleaned.csv]")
    LOG.info("========== Finished extracting frame data from fd bot csv ==========")
    return frame_data
"""Main module for frame data operations."""
import fnmatch
import os
import re
from typing import Self

import numpy as np
import pandas as pd
from pandas import Index

import skombo
from skombo import CHARS, COLS, COLS_CLASSES, LOG
from skombo.utils import (
    expand_all_x_n,
    extract_blockstop,
    filter_dict,
    format_column_headings,
    split_meter,
    timer_func,
)
from skombo.utils import for_all_methods


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


@for_all_methods(timer_func)
class FrameData(pd.DataFrame):
    """DataFrame subclass for frame data"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def re_index(self) -> Self:
        LOG.debug("Re-indexing frame data...")

        self[COLS.m_name] = self[COLS.m_name].str.strip().replace(r"\n", "", regex=True)
        self.set_index([COLS.char, COLS.m_name], verify_integrity=True, inplace=True)
        self.index.names = [COLS.char, COLS.m_name]
        LOG.debug(f"Frame data re-indexed, new index: {self.index.names}")
        return self

    def rename_cols(
        self,
        rename_cols: dict[str, str],
    ) -> Self:  # sourcery skip: default-mutable-arg
        LOG.debug(f"Renaming columns: {rename_cols} ...")
        self.rename(columns=rename_cols, inplace=True)

        cols_minus_index: list[str] = list(
            filter_dict(COLS.__dict__, self.index.names, filter_values=True).values()
        )
        self = FrameData(data=self, columns=cols_minus_index)  # type:ignore
        self.fillna(np.nan, inplace=True)
        return self

    def remove_chars_from_cols(
        self, chars: str | list[str], cols: str | list[str]
    ) -> Self:
        LOG.debug(f"Removing {chars.__repr__()} from {cols}...")
        for col in cols if isinstance(cols, list) else [cols]:
            chars_re = "|".join(re.escape(char) for char in chars)
            self[col] = self[col].str.replace(chars_re, "", regex=True)
        return self

    def bulk_remove_chars_from_cols(
        self, chars_cols: list[tuple[str | list[str], str | list[str]]]
    ) -> Self:
        for chars, cols in chars_cols:
            self.remove_chars_from_cols(chars, cols)

        return self

    def strings_to_nan(self, strings: list[str]) -> Self:
        LOG.debug(f"Replacing {strings} with NaN...")
        self.replace(strings, np.nan, inplace=True)
        self.fillna(np.nan, inplace=True)
        return self

    def col_str_replace(self, col: str, old: str, new: str) -> Self:
        LOG.debug(f"Replacing {old.__repr__()} with {new.__repr__()} in {col}...")
        self[col] = self[col].str.replace(old, new)
        return self

    def expand_xn_cols(self, xn_cols: list[str]) -> Self:
        LOG.debug(f"Expanding {xn_cols}...")
        self[xn_cols] = self[xn_cols].applymap(lambda x: expand_all_x_n(x))

        return self

    def separate_annie_stars(self) -> Self:
        LOG.debug("Separating Annie stars...")
        star_rows = self.loc[(CHARS.AN, slice(None)), [COLS.dmg, COLS.onblock]]

        star_rows = star_rows[
            (star_rows[COLS.dmg].notna() & star_rows[COLS.dmg].str.contains(r"\["))
            | (
                star_rows[COLS.onblock].notna()
                & star_rows[COLS.onblock].str.contains(r"\[")
            )
        ]

        orig_rows: pd.DataFrame = star_rows.copy()

        for col in [COLS.dmg, COLS.onblock]:
            orig_rows[col] = orig_rows[col].str.replace(r"\[.*\]", "", regex=True)

        star_rows[COLS.dmg] = star_rows[COLS.dmg].str.replace(r"\[|\]", "", regex=True)

        star_rows[COLS.onblock] = star_rows[COLS.onblock].str.extract(
            r"\[(.*)\]", expand=False
        )

        # Fill in missing values with original values from the original rows

        # Modify orignal rows to have the orig_rows damage and on_block values
        self.loc[orig_rows.index, orig_rows.columns] = orig_rows

        star_rows = pd.DataFrame(star_rows, columns=self.columns)
        star_rows.fillna(self, inplace=True)

        star_rows.index = star_rows.index.set_levels(  # type: ignore
            star_rows.index.levels[1] + "_STAR_POWER", level=1  # type: ignore
        )
        self = FrameData(pd.concat([self, star_rows]))

        return self

    def separate_damage_chip_damage(self) -> Self:
        LOG.debug("Separating damage and chip damage...")
        # Extract values in parentheses from damage column and put them in chip_damage column
        self[COLS.chip] = self[COLS.dmg].str.extract(r"\((.*)\)")[0]
        self[COLS.dmg] = self[COLS.dmg].str.replace(r"\(.*\)", "", regex=True)

        # Clean up damage and chip_damage columns
        for col in [COLS.dmg, COLS.chip]:
            self[col] = (
                self[col]
                .str.strip()
                .str.replace(r",,", ",", regex=True)
                .str.replace(r"^,|,$", "", regex=True)
            )
        return self

    def separate_meter(self) -> Self:
        LOG.debug("Separating meter...")
        # Get the meter column from the dataframe
        meter_col = self[COLS.meter]

        # Apply the split_meter function to each value in the meter column and store the results in separate lists
        on_hit, on_whiff = zip(*meter_col.apply(split_meter))

        # Update the dataframe with the new columns and return the updated dataframe
        self[COLS.meter], self[COLS.meter_whiff] = on_hit, on_whiff
        return self

    def split_cols_on_comma(self, cols: list[str]) -> Self:
        LOG.debug(f"Splitting {cols} on comma...")
        for col in cols:
            LOG.debug(f"\tSplitting {col.__repr__()}...")

            # only split if the column is a string
            if self[col].dtype in ["object", "string"]:
                self[col] = self[col].str.split(",")
        return self

    def separate_on_hit(self) -> Self:
        LOG.debug("Separating on hit...")
        self[COLS.onhit_eff] = self[COLS.onhit].copy()
        self[COLS.onhit] = self[COLS.onhit].apply(
            lambda x: x  # type: ignore
            if (isinstance(x, str) and x.strip("-").isnumeric()) or isinstance(x, int)
            else None
        )

        self[COLS.onhit_eff] = self[COLS.onhit_eff].apply(
            lambda x: x if isinstance(x, str) and not x.strip("-").isnumeric() else np.NaN  # type: ignore
        )
        return self

    def categorise_moves(self) -> Self:
        LOG.debug("Categorising moves...")
        universal_move_categories: dict[str, str] = skombo.UNIVERSAL_MOVE_CATEGORIES
        self[COLS.move_cat] = self.index.get_level_values(1).map(
            universal_move_categories
        )

        re_normal_move: re.Pattern[str] = skombo.RE_NORMAL_MOVE

        normal_strengths: dict[str, str] = skombo.NORMAL_STRENGTHS
        # Normal moves are where move_name matches the regex and the move_category is None, we can use the regex to find the strength of the move by the first group

        # Make a mask of the rows that are normal moves, by checking against self.index.get_level_values(1)

        mask: Index = self.index.get_level_values(1).map(
            lambda x: isinstance(x, str) and re_normal_move.search(x) is not None
        )
        # Assign the move_category to the strength of the move
        self.loc[mask, COLS.move_cat] = (  # type: ignore
            self.loc[mask]
            .index.get_level_values(1)
            .map(lambda x: normal_strengths[re_normal_move.search(x).group(1)] + "_NORMAL")  # type: ignore
        )

        # Supers are where meter_on_hit has length 1 and meter_on_hit[0] is  -100 or less
        self.loc[
            self.astype(str)[COLS.meter].str.contains(r"-\d\d", regex=True, na=False)
            & self[COLS.move_cat].isna(),
            COLS.move_cat,
        ] = "SUPER"
        # For now, assume everything else is a special
        # TODO: Add more special move categories for things like double's level 5 projectiles, annie taunt riposte etc

        self.loc[self[COLS.move_cat].isna(), COLS.move_cat] = "SPECIAL_MOVE"
        return self

    def add_undizzy_values(self) -> Self:
        LOG.debug("Adding undizzy values...")
        self[COLS.undizzy] = self[COLS.move_cat].map(skombo.UNDIZZY_DICT)
        return self

    def split_on_pushblock(self) -> Self:
        LOG.debug("Splitting on pushblock...")
        self[COLS.onpb] = self[COLS.onpb].str.split(" to ")
        return self

    def sep_hitstop_blockstop(self) -> Self:
        self[COLS.blockstop] = self[COLS.hitstop].apply(
            # big lambda to extract the blockstop value from the hitstop column
            # is messy but it is really convenient 🤷‍♀️
            lambda x: bs
            if isinstance(x, list)  # only split list values
            and (
                bs := [
                    ex_i.strip()
                    for i in x  # iterate over each value in the list
                    if isinstance(i, str)
                    if (ex_i := extract_blockstop(i))
                    is not None  # extract the blockstop value
                ]
            ).__len__()
            > 0  # only return the list if it has values
            else pd.NA  # type: ignore
        )

        self[COLS.hitstop] = self[COLS.hitstop].apply(
            # similar but just remove the blockstop values from the hitstop column
            lambda x: ex_hs
            if isinstance(x, list)
            and (
                ex_hs := [
                    i.replace(
                        ex_i, ""
                    ).strip()  # remove the blockstop value from the hitstop value
                    for i in x  # iterate over each value in the list
                    if isinstance(i, str)
                    if (ex_i := extract_blockstop(i, in_paren=False))
                    is not None  # extract the blockstop value
                ]
            ).__len__()
            > 0
            else x
        )

        return self


cols_to_rename: dict[str, str] = {
    "on_block": COLS.onblock,
    "meter": COLS.meter,
    "on_hit": COLS.onhit,
}
remove_chars_from_cols: list[tuple[str | list[str], str | list[str]]] = [
    ("\n", COLS_CLASSES.REMOVE_NEWLINE_COLS),
    (["+", "±"], COLS_CLASSES.PLUS_MINUS_COLS),
    ("%", COLS.meter),
]

string_to_nan: list[str] = ["-", ""]
global FD


def get_fd() -> FrameData:
    FD_BOT_CSV_MANAGER = FdBotCsvManager()
    global FD
    FD = FrameData(FD_BOT_CSV_MANAGER.dataframes["frame_data"].convert_dtypes())
    return FD


def clean_fd() -> FrameData:
    global FD
    LOG.debug("Cleaning frame data...")
    FD = (
        FD.re_index()  # Reindex the dataframe to have character and move_name as the index
        .rename_cols(
            rename_cols=cols_to_rename
        )  # Rename columns as specified in cols_to_rename
        .bulk_remove_chars_from_cols(
            remove_chars_from_cols
        )  # Remove characters from columns as specified in remove_chars_from_cols
        .strings_to_nan(
            string_to_nan
        )  # Replace strings with np.nan as specified in string_to_nan
        .col_str_replace(
            COLS.a_names, "\n", ","
        )  # Replace newlines in alt_names with commas
        .expand_xn_cols(COLS_CLASSES.XN_COLS)  # Expand all xN columns
        .separate_annie_stars()  # Separate Annie's star power moves into separate rows
        .separate_damage_chip_damage()  # Separate damage and chip damage into separate columns
        .separate_meter()  # Separate meter into on_hit and on_whiff
        .split_cols_on_comma(COLS_CLASSES.LIST_COLUMNS)  # Split numeric list cols
        .separate_on_hit()  # Separate on_hit and on_hit_eff
        .categorise_moves()  # Categorise moves
        .add_undizzy_values()  # Add undizzy values
        .split_on_pushblock()  # Split on_pushblock into lists
        .sep_hitstop_blockstop()  # Separate hitstop and blockstop
    )
    return FD


get_fd()
clean_fd()

FD.to_csv("fd_cleaned.csv")

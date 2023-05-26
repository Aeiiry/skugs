"""Main module for frame data operations."""
import fnmatch
import functools
import os
import re
from typing import Self

import numpy as np
import pandas as pd
from loguru import logger as log
from pandas import Index

import skombo
from skombo import CHAR_COLS, CHARS, COLS_CLASSES, FD_COLS
from skombo.utils import (
    expand_all_x_n,
    extract_blockstop,
    filter_dict,
    for_all_methods,
    format_column_headings,
    split_meter,
    timer_func,
)


class CsvManager:
    """Class to manage CSV files"""

    def __init__(self, path: str, file_keys: dict[str, str]) -> None:
        log.debug(f"Initialising CSV manager for {path}...")
        self.path: str = path
        self.file_keys: dict[str, str] = file_keys
        self.dataframes: dict[str, pd.DataFrame] = {}
        """Raw dataframes from csvs before they are modified"""
        for key in self.file_keys:
            self.dataframes[key] = self.open_csv(key).dropna(axis=0, how="all")
            log.debug(f"Opened {key} CSV")

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


@for_all_methods(timer_func)
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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def re_index(self) -> Self:
        log.debug("Re-indexing frame data...")

        self[FD_COLS.m_name] = (
            self[FD_COLS.m_name].str.strip().replace(r"\n", "", regex=True)
        )
        self.set_index(
            [FD_COLS.char, FD_COLS.m_name], verify_integrity=True, inplace=True
        )
        self.index.names = [FD_COLS.char, FD_COLS.m_name]
        log.debug(f"Frame data re-indexed, new index: {self.index.names}")
        return self

    def rename_cols(
        self,
        rename_cols: dict[str, str],
    ) -> Self:  # sourcery skip: default-mutable-arg, lambdas-should-be-short
        log.debug(f"Renaming columns: {rename_cols} ...")
        self.rename(columns=rename_cols, inplace=True)

        cols_minus_index: list[str] = list(
            filter_dict(FD_COLS.__dict__, self.index.names, filter_values=True).values()
        )
        # noinspection PyMethodFirstArgAssignment
        self = FrameData(data=self, columns=cols_minus_index)
        self.fillna(np.nan, inplace=True)
        return self  # type: ignore

    def remove_chars_from_cols(
        self, chars: str | list[str], cols: str | list[str]
    ) -> Self:
        log.debug(f"Removing {chars.__repr__()} from {cols}...")
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
        log.debug(f"Replacing {strings} with NaN...")
        self.replace(strings, np.nan, inplace=True)
        self.fillna(np.nan, inplace=True)
        return self

    def col_str_replace(self, col: str, old: str, new: str) -> Self:
        log.debug(f"Replacing {old.__repr__()} with {new.__repr__()} in {col}...")
        self[col] = self[col].str.replace(old, new)
        return self

    def expand_xn_cols(self, xn_cols: list[str]) -> Self:
        log.debug(f"Expanding {xn_cols}...")
        self[xn_cols] = self[xn_cols].applymap(lambda x: expand_all_x_n(x))

        return self

    def separate_annie_stars(self) -> Self:
        log.debug("Separating Annie stars...")
        star_rows = self.loc[(CHARS.AN, slice(None)), [FD_COLS.dmg, FD_COLS.onblock]]

        star_rows = star_rows[
            (
                star_rows[FD_COLS.dmg].notna()
                & star_rows[FD_COLS.dmg].str.contains(r"\[")
            )
            | (
                star_rows[FD_COLS.onblock].notna()
                & star_rows[FD_COLS.onblock].str.contains(r"\[")
            )
        ]

        orig_rows: pd.DataFrame = star_rows.copy()

        for col in [FD_COLS.dmg, FD_COLS.onblock]:
            orig_rows[col] = orig_rows[col].str.replace(r"\[.*\]", "", regex=True)

        star_rows[FD_COLS.dmg] = star_rows[FD_COLS.dmg].str.replace(
            r"\[|\]", "", regex=True
        )

        star_rows[FD_COLS.onblock] = star_rows[FD_COLS.onblock].str.extract(
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
        # Expand self to include the new rows
        # Keep FrameData type
        # noinspection PyMethodFirstArgAssignment
        self = FrameData(pd.concat([self, star_rows]))  # type: ignore

        return self  # type: ignore

    def separate_damage_chip_damage(self) -> Self:
        log.debug("Separating damage and chip damage...")
        # Extract values in parentheses from damage column and put them in chip_damage column
        self[FD_COLS.chip] = self[FD_COLS.dmg].str.extract(r"\((.*)\)")[0]
        self[FD_COLS.dmg] = self[FD_COLS.dmg].str.replace(r"\(.*\)", "", regex=True)

        # Clean up damage and chip_damage columns
        for col in [FD_COLS.dmg, FD_COLS.chip]:
            self[col] = (
                self[col]
                .str.strip()
                .str.replace(r",,", ",", regex=True)
                .str.replace(r"^,|,$", "", regex=True)
            )
        return self

    def separate_meter(self) -> Self:
        log.debug("Separating meter...")
        # Get the meter column from the dataframe
        meter_col = self[FD_COLS.meter]

        # Apply the split_meter function to each value in the meter column and store the results in separate lists
        on_hit, on_whiff = zip(*meter_col.apply(split_meter))

        # Update the dataframe with the new columns and return the updated dataframe
        self[FD_COLS.meter], self[FD_COLS.meter_whiff] = on_hit, on_whiff
        return self

    def split_cols_on_comma(self, cols: list[str]) -> Self:
        log.debug(f"Splitting {cols} on comma...")
        for col in cols:
            log.debug(f"\tSplitting {col.__repr__()}...")

            # only split if the column is a string
            if self[col].dtype in ["object", "string"]:
                self[col] = self[col].str.split(",")
        return self

    def separate_on_hit(self) -> Self:
        log.debug("Separating on hit...")
        self[FD_COLS.onhit_eff] = self[FD_COLS.onhit].copy()
        self[FD_COLS.onhit] = self[FD_COLS.onhit].apply(  # type: ignore
            lambda x: x
            if (isinstance(x, str) and x.strip("-").isnumeric()) or isinstance(x, int)
            else None
        )

        self[FD_COLS.onhit_eff] = self[FD_COLS.onhit_eff].apply(
            lambda x: x if isinstance(x, str) and not x.strip("-").isnumeric() else np.NaN  # type: ignore
        )
        return self

    def categorise_moves(self) -> Self:
        log.debug("Categorising moves...")
        universal_move_categories: dict[str, str] = skombo.UNIVERSAL_MOVE_CATEGORIES
        self[FD_COLS.move_cat] = self.index.get_level_values(1).map(
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
        self.loc[mask, FD_COLS.move_cat] = (  # type: ignore
            self.loc[mask]
            .index.get_level_values(1)
            .map(lambda x: normal_strengths[re_normal_move.search(x).group(1)] + "_NORMAL")  # type: ignore
        )

        # Supers are where meter_on_hit has length 1 and meter_on_hit[0] is  -100 or less
        self.loc[
            self.astype(str)[FD_COLS.meter].str.contains(r"-\d\d", regex=True, na=False)
            & self[FD_COLS.move_cat].isna(),
            FD_COLS.move_cat,
        ] = "SUPER"
        # For now, assume everything else is a special
        # TODO: Add more special move categories for things like double's level 5 projectiles, annie taunt riposte etc

        self.loc[self[FD_COLS.move_cat].isna(), FD_COLS.move_cat] = "SPECIAL_MOVE"
        return self

    def add_undizzy_values(self) -> Self:
        log.debug("Adding undizzy values...")
        self[FD_COLS.undizzy] = self[FD_COLS.move_cat].map(skombo.UNDIZZY_DICT)
        return self

    def split_on_pushblock(self) -> Self:
        log.debug("Splitting on pushblock...")
        self[FD_COLS.onpb] = self[FD_COLS.onpb].str.split(" to ")
        return self

    # todo finish this with no scaling, scaling notes and whatever else
    def extract_damage_scaling(self) -> Self:
        replacement_refs = {}
        scaling_rows = self.loc[
            self[FD_COLS.props]
            .str.contains("scaling", flags=re.IGNORECASE)
            .fillna(False)
        ]
        # Regex patterns for finding the scaling values in the props column
        forced_scaling_re = re.compile(r"(\d+)%\s?damage scaling", flags=re.IGNORECASE)
        min_scaling_re = re.compile(r"(\d+)%\s?min\.?\s? scaling", flags=re.IGNORECASE)

        for idx, row in scaling_rows.iterrows():
            props = row[FD_COLS.props]
            # Search for forced scaling and min scaling in the props column
            forced_scaling = forced_scaling_re.search(props)
            min_scaling = min_scaling_re.search(props)
            # Store any match in a dictionary
            scaling_dict = (
                {"forced_scaling": int(forced_scaling[1])} if forced_scaling else {}
            )
            if min_scaling:
                scaling_dict["min_scaling"] = int(min_scaling[1])

            replacement_refs[idx] = scaling_dict
        # Update the DataFrame all at once instead of iterating through it
        self.loc[scaling_rows.index, FD_COLS.scaling] = pd.Series(data=replacement_refs)

        return self

    def separate_hitstop_blockstop(self) -> Self:
        # Split hitstop and blockstop into separate columns
        log.debug("Separating hitstop and blockstop...")

        # Make a mask of the rows that have a blockstop value
        block_mask: pd.Series[bool] = (
            self[FD_COLS.hitstop].str.contains("on block").fillna(False)
        )

        block_rows: pd.DataFrame = self.loc[
            block_mask, [FD_COLS.hitstop, FD_COLS.blockstop]
        ]

        for i, row in block_rows.iterrows():
            # Extract the blockstop value from the hitstop column, returns a re match object
            ext_block = extract_blockstop(row[FD_COLS.hitstop])
            if ext_block is not None:
                # TLDR: blockstop = value before "on block", hitstop has the parenthesised substring removed
                blockstop, hitstop = (
                    ext_block[1].strip(),
                    row[FD_COLS.hitstop].replace(ext_block[0].strip(), "").strip(),
                )
                (
                    block_rows.at[i, FD_COLS.blockstop],
                    block_rows.at[i, FD_COLS.hitstop],
                ) = (
                    blockstop,
                    hitstop,
                )

        self.loc[block_mask, [FD_COLS.hitstop, FD_COLS.blockstop]] = block_rows
        return self

    def clean_fd(self) -> Self:
        self = (
            self.re_index()  # Reindex the dataframe to have character and move_name as the index
            .rename_cols(
                rename_cols=cols_to_rename
            )  # Rename columns as specified in cols_to_rename
            .bulk_remove_chars_from_cols(
                remove_chars_from_cols
            )  # Remove characters from columns as specified in remove_chars_from_cols
            .expand_xn_cols(COLS_CLASSES.XN_COLS)  # Expand all xN columns
            .separate_annie_stars()  # Separate Annie's star power moves into separate rows
            .separate_damage_chip_damage()  # Separate damage and chip damage into separate columns
            .separate_meter()  # Separate meter into on_hit and on_whiff
            .separate_on_hit()  # Separate on_hit and on_hit_eff
            .categorise_moves()  # Categorise moves
            .add_undizzy_values()  # Add undizzy values
            .split_on_pushblock()  # Split on_pushblock into lists
            .separate_hitstop_blockstop()  # Separate hitstop and blockstop
            .extract_damage_scaling()
            .split_cols_on_comma(COLS_CLASSES.LIST_COLUMNS)  # Split numeric list cols
            .strings_to_nan(
                string_to_nan
            )  # Replace strings with np.nan as specified in string_to_nan
        )
        return self


@for_all_methods(timer_func)
class Character:
    """Class for an individual character. Containing identifying attributes alongside fast ways to search their move-lists"""

    def __init__(self, input_series: pd.Series, character_moves: pd.DataFrame):
        self.name = input_series[CHAR_COLS.char]
        self.short_names: list[str] = input_series[CHAR_COLS.short_names].split("\n")
        self.color = input_series[CHAR_COLS.color]
        # We can remove the first index of the multi-index as it is the character name
        self.moves = character_moves.droplevel(0)
        self.normals = self.moves.loc[
            self.moves[FD_COLS.move_cat].str.endswith("NORMAL")
        ].reset_index()[FD_COLS.m_name]


@for_all_methods(timer_func)
class CharacterManager:
    """Class for managing the characters"""

    def __init__(self, input_df: pd.DataFrame, frame_data: FrameData):
        for _, character_series in input_df.iterrows():
            # character is the first index of the multi-index
            character = character_series[CHAR_COLS.char]
            character_moves = frame_data.loc[
                frame_data.index.get_level_values(0) == character
            ]

            setattr(self, character, Character(character_series, character_moves))


cols_to_rename: dict[str, str] = {
    "on_block": FD_COLS.onblock,
    "meter": FD_COLS.meter,
    "on_hit": FD_COLS.onhit,
}
remove_chars_from_cols: list[tuple[str | list[str], str | list[str]]] = [
    ("\n", COLS_CLASSES.REMOVE_NEWLINE_COLS),
    (["+", "±"], COLS_CLASSES.PLUS_MINUS_COLS),
    ("%", FD_COLS.meter),
]

string_to_nan: list[str] = ["-", ""]


@functools.cache
def get_fd_bot_csv_manager() -> FdBotCsvManager:
    """Get the FDBotCSVManager instance"""
    return FdBotCsvManager()


@functools.cache
def get_fd_bot_frame_data() -> FrameData:
    """Get the FrameData instance"""
    return FrameData(get_fd_bot_csv_manager().dataframes["frame_data"])  # type: ignore


@functools.cache
def get_fd_bot_character_manager() -> CharacterManager:
    """Get the CharacterManager instance"""
    return CharacterManager(
        get_fd_bot_csv_manager().dataframes["characters"],
        get_fd_bot_frame_data().clean_fd(),
    )

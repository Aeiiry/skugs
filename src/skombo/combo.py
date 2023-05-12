"""Parsing combo from strings/csvs and calculating the combo's damage/properties
"""

import re
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from skombo import fd_clean as fdo
from skombo import sklog as sklog

log = sklog.get_logger()

global fd
import functools

DataFrame = pd.DataFrame


def parse_combos_from_csv(csv_path: str) -> pd.DataFrame:
    """Parse combo from csv file, needs to have columns of "character", "notation", "damage" for testing purposes"""
    combos_df: DataFrame = pd.read_csv(csv_path)
    return pd.concat(
        [
            parse_combo_from_string(
                combos_df["character"][index], combos_df["notation"][index]
            )
            for index in range(len(combos_df.index))
        ]
    )


def fd_rows_to_combo_calc_df(fd_rows: DataFrame, character_fd: DataFrame) -> DataFrame:
    """This is a function primarily to turn pure frame data rows into a dataframe that can be used for combo calculations
    Individual hits will be separated out, and the damage will be calculated for each hit alongside considerations for what current move the combo is on
    """

    # First we need to expand on moves that are technically multiple moves, e.g 5HPx2 -> 5HP, 5HPx2. 236K~K -> 236K, 236K~K

    # Case for moves that have a number in the name, e.g 5HPx2
    # There is a possibility of changing the number of rows in the dataframe, so we need to iterate over the dataframe in reverse
    re_xn_move_name: re.Pattern[str] = re.compile(r"[xX]\s?(\d)")
    for index, row in fd_rows[::-1].iterrows():
        if move_name_xn := re_xn_move_name.search(row["move_name"]):
            log.debug(f"Found move with possible xN repetition: {row['move_name']}")

            if move_name_xn[1].isnumeric():
                log.debug(f"Found move with xN repetition: {row['move_name']}")
                # Get the number of repetitions
                repetitions: int = int(move_name_xn[1])

                # See if a valid move exists for the move with the xN removed
                move_name: str = row["move_name"].replace(move_name_xn[0], "").strip()

                # look for the move name in the frame data
                move_name_df: DataFrame = character_fd.loc[
                    character_fd["move_name"] == move_name
                ]
                if len(move_name_df) == 1:
                    log.debug(f"Found move name {move_name} in frame data")

    return fd_rows


@functools.cache
def similar(a: Any, b: Any) -> float:
    return (
        ratio
        if (ratio := SequenceMatcher(None, a, b).real_quick_ratio()) > 0
        else SequenceMatcher(None, a, b).ratio()
    )


@functools.cache
def parse_combo_from_string(character: str, combo_string: str) -> pd.DataFrame:
    combo, character_moves, combo_moves = initial_combo_operations(
        character, combo_string
    )

    # Attempt to find each move in the combo string in the character's moves from the frame data
    # If a move is not found, check each item in the alt_names list for the move

    # Search character_moves DataFrame for move names or alternative names
    # found_moves = initial_search(character_moves, combo_moves)

    # Find the move with the highest similarity score if multiple matches exist
    # if len(found_moves) > 1:
    #   found_moves = best_match(combo_moves, found_moves)

    # Add the move to the combo
    # combo = pd.concat([combo, found_moves])

    # combo_calc_df: DataFrame = fd_rows_to_combo_calc_df(combo, character_moves)

    return combo


def best_match(original_move: str, found_moves: pd.Series) -> pd.Series:  # type: ignore
    move_name_similarity: pd.Series[Any] = found_moves.apply(
        lambda move_name: similar(original_move, move_name)
    )
    found_moves = found_moves.loc[move_name_similarity == move_name_similarity.max()]

    return found_moves


def move_name_search(
    character_moves: pd.DataFrame, combo_moves: pd.Series, move_name: str
) -> pd.Series:
    if isinstance(move_name, str):
        # Search for the move name in the frame data
        move_name_series = (
            character_moves["move_name"]
            .where(character_moves["move_name"] == move_name)
            .dropna()
            .reset_index(drop=True)
        )

        # If move name not found in move_names then search in alt_names
        if len(move_name_series) == 0:
            move_name_series = (
                character_moves["move_name"]
                .where(character_moves["alt_names"].str.contains(move_name))
                .dropna()
                .reset_index(drop=True)
            )

        # If there are multiple matches for move name, then log info message and select the best match
        if len(move_name_series) > 1:
            log.info(
                f"Multiple moves found for move name {move_name}, checking similarity to pick best match"
            )
            move_name_series = best_match(move_name, move_name_series)

        return move_name_series


def initial_search(character_moves: pd.DataFrame, combo_moves: pd.Series):
    # Initialize empty DataFrame with columns 'original_move_name' and 'move_name'
    move_name_series: pd.Series = pd.Series()
    combo_series: pd.Series = pd.Series()
    # Search for each move in the combo string in the frame data
    for move_name in combo_moves:
        # If move_name isn't a string then skip to next move in the list
        move_name_series = move_name_search(character_moves, combo_moves, move_name)

        # Add the move name to the move_name_series DataFrame
        if len(move_name_series) == 0:
            log.warning(f"Move name {move_name} not found in search")
            combo_series = pd.concat([combo_series, pd.Series()])
        else:
            log.debug(f"Move name {move_name} found in search")

            combo_series = pd.concat([combo_series, move_name_series])
    log.info(combo_series)
    return move_name_series


def initial_combo_operations(
    character: str, combo_string: str
) -> tuple[DataFrame, DataFrame, pd.Series]:
    combo: DataFrame = pd.DataFrame()
    character_moves: DataFrame = get_character_moves(character)

    combo_move_names = pd.Series(combo_string.strip().split(" "))

    combo_move_names = combo_move_names.str.lower()
    character_moves.loc[:, ["move_name"]] = character_moves["move_name"].str.lower()
    character_moves.loc[:, ["alt_names"]] = character_moves["alt_names"].str.lower()

    # log.info(f"combo_moves: {combo_move_names}")
    return combo, character_moves, combo_move_names


@functools.cache
def get_character_moves(character: str) -> pd.DataFrame:
    fd: DataFrame = fdo.get_fd_bot_data()
    character_moves: DataFrame = fd.loc[
        fd["character"].str.lower() == character.lower()
    ]

    log.info(f"Retreived {len(character_moves)} moves for {character}")
    return character_moves

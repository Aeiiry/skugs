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


def similar(a: Any, b: Any) -> float:
    return (
        ratio
        if (ratio := SequenceMatcher(None, a, b).real_quick_ratio()) > 0
        else SequenceMatcher(None, a, b).ratio()
    )


def parse_combo_from_string(character: str, combo_string: str) -> pd.DataFrame:
    combo: DataFrame
    character_moves: DataFrame
    combo_moves: list[str]

    combo, character_moves, combo_moves = initial_combo_operations(
        character, combo_string
    )

    # Attempt to find each move in the combo string in the character's moves from the frame data
    # If a move is not found, check each item in the alt_names list for the move

    # Search character_moves DataFrame for move names or alternative names
    found_moves: DataFrame = initial_search(character_moves, combo_moves)

    # Find the move with the highest similarity score if multiple matches exist
    # if len(found_moves) > 1:
    #   found_moves = best_match(combo_moves, found_moves)

    # Add the move to the combo
    combo = pd.concat([combo, found_moves])

    # combo_calc_df: DataFrame = fd_rows_to_combo_calc_df(combo, character_moves)

    return combo


def best_match(combo_moves: pd.Series, found_moves: DataFrame) -> DataFrame:  # type: ignore
    move_name_similarity: pd.Series[Any] = found_moves["move_name"].apply(
        lambda x: max(similar(x, y) for y in combo_moves)
    )
    found_moves = found_moves.loc[move_name_similarity == move_name_similarity.max()]

    return found_moves


def initial_search(character_moves: DataFrame, combo_moves: list[str]) -> DataFrame:
    # Initialize empty DataFrame with columns 'original_move_name' and 'move_name'
    found_moves: pd.DataFrame = pd.DataFrame(
        columns=["original_move_name", "move_name"]
    )

    found_moves["original_move_name"] = combo_moves

    # Search for each move in the combo string in the frame data
    for move_name in combo_moves:
        # Check if the move is in the frame data
        move_name_df: DataFrame = character_moves.loc[
            character_moves["move_name"] == move_name
        ]

        # If the move is not in the frame data, check the alternative names
        if len(move_name_df) == 0:
            move_name_df = character_moves.loc[
                character_moves["alt_names"].apply(lambda x: move_name in x)
            ]

        # If the move is in the frame data, add it to the found moves DataFrame
        if len(move_name_df) == 1:
            found_moves = pd.concat([found_moves, move_name_df])

    return found_moves


def initial_combo_operations(
    character: str, combo_string: str
) -> tuple[DataFrame, DataFrame, list[str]]:
    combo: DataFrame = pd.DataFrame()
    character_moves: DataFrame = get_character_moves(character)

    combo_move_names: list[str] = combo_string.strip().split(" ")
    log.info(f"combo_moves: {combo_move_names}")

    combo_move_names = [move.lower() for move in combo_move_names]
    character_moves.loc[:, ["move_name"]] = character_moves["move_name"].str.lower()
    character_moves.loc[:, ["alt_names"]] = character_moves.loc[
        :, ["alt_names"]
    ].applymap(lambda x: [alt_name.lower() for alt_name in x])

    return combo, character_moves, combo_move_names


@functools.cache
def get_character_moves(character: str) -> pd.DataFrame:
    fd: DataFrame = fdo.get_fd_bot_data()
    character_moves: DataFrame = fd[fd["character"].str.lower() == character.lower()]

    log.info(f"Retreived {len(character_moves)} moves for {character}")
    return character_moves

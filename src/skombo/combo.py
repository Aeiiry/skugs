"""Parsing combo from strings/csvs and calculating the combo's damage/properties
"""

import re
from difflib import SequenceMatcher
from typing import Any
from numpy import char, number

import pandas as pd
from pandas.core import frame
from skombo import const
from skombo import fd_clean as fdo
from skombo import sklog as sklog

log = sklog.get_logger()

global fd
import functools

DataFrame = pd.DataFrame


def get_combo_scaling_factor(fd: DataFrame) -> pd.Series:
    """Get the scaling factor for the combo"""
    fd = fd.reset_index(drop=True)
    scaling_series = pd.Series(dtype=float)

    # constants
    factor = const.SCALING_FACTOR

    min_1k = const.SCALING_MIN_1K
    min_other = const.SCALING_MIN
    start = const.SCALING_START

    # Scaling is 1.0 for first 3 hits
    # Then 0.875 less for each hit after that
    # Minimum scaling is 0.2
    # Minimum scaling for moves with over 1000 base damage is 0.275
    index: int
    skip_row: int = 0
    for index, row in fd.iterrows():  # type: ignore
        offset_index = index - skip_row
        if not isinstance(row["damage"], int) or row["damage"] == 0:
            scaling_series[offset_index] = scaling_series[offset_index - 1]
            skip_row += 1

            continue
        if offset_index < 3:
            scaling_series[offset_index] = start
        else:
            min_for_hit: float = min_1k if row["damage"] >= 1000 else min_other
            scaling_series[offset_index] = max(
                start * factor ** (offset_index - 2), min_for_hit
            )

    # Return the scaling factor for the combo for each hit

    return scaling_series


def naiive_damage_calc(moves: DataFrame):
    """Naiive damage calc for a series of moves"""
    moves["scaling_factor"] = get_combo_scaling_factor(moves)
    moves["damage"] = moves["damage"].apply(fdo.attempt_to_int)
    # Replace any non floats with 0
    moves["damage"] = moves["damage"].apply(lambda x: x if isinstance(x, int) else 0).astype(float)
    moves["scaled_damage"] = moves["damage"] * moves["scaling_factor"]

    return moves["scaled_damage"].sum(), moves


def parse_combos_from_csv(csv_path: str) -> list:
    """Parse combo from csv file, needs to have columns of "character", "notation", "damage" for testing purposes"""
    log.info(f"========== Parsing combos from csv [{csv_path}] ==========")
    combos_df: DataFrame = pd.read_csv(csv_path)
    log.info(f"Parsing [{len(combos_df.index)}] combos from csv")
    return [
        (
            parse_combo_from_string(
                combos_df["character"][index], combos_df["notation"][index]
            ),
            combos_df["damage"][index],
        )
        for index in range(len(combos_df.index))
    ]


def fd_to_combo_df(fd: DataFrame, character_fd: DataFrame) -> DataFrame:
    """This is a function primarily to turn pure frame data rows into a dataframe that can be used for combo calculations
    Individual hits will be separated out, and the damage will be calculated for each hit alongside considerations for what current move the combo is on
    """

    combo_df = fd.copy().drop(
        ["alt_names", "footer", "thumbnail_url", "footer_url"], axis=1
    )

    # Split damage into individual rows for each hit, damage column contains individual lists of damage for each hit of the move

    combo_df = combo_df.explode(["damage"]).convert_dtypes()

    return combo_df


@functools.cache
def similar(a: Any, b: Any) -> float:
    return (
        ratio
        if (ratio := SequenceMatcher(None, a, b).real_quick_ratio()) > 0
        else SequenceMatcher(None, a, b).ratio()
    )


@functools.cache
def parse_combo_from_string(character: str, combo_string: str) -> pd.DataFrame:
    # Get the frame data for the character

    character_moves: DataFrame = get_character_moves(character)

    combo_move_names = pd.Series(combo_string.strip().split(" "))
    log.info(f"Parsing combo for [{character}] : {combo_move_names.to_list()}")
    # Attempt to find each move in the combo string in the character's moves from the frame data
    # If a move is not found, check each item in the alt_names list for the move

    # Search character_moves DataFrame for move names or alternative names
    combo_df: DataFrame = find_combo_moves(character_moves, combo_move_names)

    # Find the move with the highest similarity score if multiple matches exist
    # if len(found_moves) > 1:
    # found_moves = best_match(combo_moves, found_moves)

    combo_calc_df: DataFrame = fd_to_combo_df(combo_df, character_moves)

    return combo_calc_df.reset_index(drop=True)


def best_match(original_move: str, found_moves: pd.Series) -> pd.Series:  # type: ignore
    move_name_similarity: pd.Series[Any] = found_moves.apply(
        lambda move_name: similar(original_move, move_name)
    )
    found_moves = found_moves.loc[move_name_similarity == move_name_similarity.max()]

    return found_moves


def character_specific_move_name_check(character: str, move_name: str) -> str:
    # sourcery skip: merge-nested-ifs
    if character == "ANNIE":
        # We don't actually need the move strength for annie divekicks
        if re.search(r"236[lmh]k", move_name, flags=re.IGNORECASE):
            log.debug(f"Removing move strength from annie divekick {move_name}")
            move_name = re.sub(r"[lmh]", "", move_name, flags=re.IGNORECASE)

    return move_name


def find_single_move(
    character_moves: pd.DataFrame, combo_moves: pd.Series, move_name: str
) -> pd.Series:  # sourcery skip: remove-pass-body
    move_name = move_name.upper()

    character = str(character_moves.index.get_level_values(0)[0])

    blank_move = pd.Series(index=character_moves.columns, name=(character, move_name))

    move_df = pd.DataFrame()

    if move_name in const.IGNORED_MOVES:
        return blank_move

    in_index = character_moves.index.get_level_values(1) == move_name

    move_name = character_specific_move_name_check(character, move_name)

    if in_index.max() == True:
        # log.info(f"Found move name {move_name} in frame data")
        move_df = character_moves[in_index]

    else:
        in_alt_names = character_moves["alt_names"].str.contains(move_name, regex=False)
        if in_alt_names.any():
            pass
        # log.info(f"Found move name {move_name} in alt_names")
        else:
            log.warning(f"!!! Could not find move name {move_name} in frame data !!!")
        move_df = character_moves[in_alt_names]

    blank_move = pd.Series(index=character_moves.columns, name=(character, move_name))
    return move_df.iloc[0] if move_df.__len__() > 0 else blank_move


def find_move_repeats_follow_ups(moves: pd.Series):
    moves_new: pd.Series = pd.Series()

    xn_re: re.Pattern[str] = re.compile(r"\s?X\s?(\d+)", re.IGNORECASE)

    for move in moves:
        if xn_match := xn_re.search(move):
            # log.info(f"Found move repeat {xn_match[1]} in {move}")

            xn = int(xn_match[1])

            for i in range(xn):
                move_name: str = (
                    move.replace(xn_match[0], f" X{str(i + 1)}")
                    if i > 0
                    else move.replace(xn_match[0], "")
                )

                moves_new = pd.concat([moves_new, pd.Series([move_name])])

        elif followup_match := re.search(r"[~]", move):
            followup_split: list[str] = followup_match.string.split("~")

            for i in range(followup_split.__len__()):
                moves_new = pd.concat(
                    [moves_new, pd.Series("~".join(followup_split[: i + 1]))]
                )

            # log.info(f"Found followup {followup_split[1]} in {move}")

        else:
            moves_new = pd.concat([moves_new, pd.Series([move])])

    # log.info(moves_new)

    return moves_new


def find_combo_moves(
    character_moves: pd.DataFrame, combo_moves: pd.Series
) -> DataFrame:
    # Initialize empty DataFrame with columns 'original_move_name' and 'move_name'
    move_series: pd.Series = pd.Series()
    # character mopves columns plus 'character' and 'move_name'
    combo_df_cols: list[str] = ["character", "move_name"] + list(
        character_moves.columns
    )
    combo_df: pd.DataFrame = pd.DataFrame(columns=combo_df_cols)

    # Interpret things such as 5HPx2 as 5hpx1, 5hpx2
    combo_moves = find_move_repeats_follow_ups(combo_moves)

    # Search for each move in the combo string in the frame data
    for i, move_name in enumerate(combo_moves):
        # If move_name isn't a string then skip to next move in the list
        move_series = find_single_move(character_moves, combo_moves, move_name).fillna(
            ""
        )

        # Add the move name to the move_name_series DataFrame
        combo_df.loc[i, :] = move_series
        combo_df.at[i, "character"] = move_series.name[0]  # type: ignore
        combo_df.at[i, "move_name"] = move_series.name[1]  # type: ignore

    return combo_df


@functools.cache
def get_character_moves(character: str) -> pd.DataFrame:
    fd: DataFrame = fdo.get_fd_bot_data()
    character_moves: DataFrame = fd.loc[
        fd.index.get_level_values(0) == character.upper()
    ]

    # log.info(f"Retreived {len(character_moves)} moves for {character}")
    return character_moves

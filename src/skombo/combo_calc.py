"""Parsing combo from strings/csvs and calculating the combo's damage/properties
"""

import functools
import os
import re
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd
from numpy import floor
from pandas import DataFrame, Series
import skombo
from skombo import CHARS, COLS, COLS_CLASSES
from skombo.fd_ops import FD, FD_BOT_CSV_MANAGER
from loguru import logger as log

def get_combo_scaling(combo: DataFrame) -> DataFrame:
    combo = combo.reset_index(drop=True)
    factor = skombo.SCALING_FACTOR
    min_1k = skombo.SCALING_MIN_1K
    min_other = skombo.SCALING_MIN
    start = skombo.SCALING_START

    combo[COLS.hit_scaling] = 0.0
    combo[COLS.mod_scaling] = 0.0
    for i, row in combo.iterrows():
        min_for_hit = min_other
        if i == 0 or row[COLS.dmg] != 0 and i < 3:  # type: ignore
            combo.loc[i, COLS.hit_scaling] = start  # type: ignore
        elif not isinstance(row[COLS.dmg], int) or row[COLS.dmg] == 0:
            combo.loc[i, COLS.hit_scaling] = combo.loc[i - 1, COLS.hit_scaling]  # type: ignore
        else:
            min_for_hit = min_1k if row[COLS.dmg] >= 1000 else min_other
            combo.loc[i, COLS.hit_scaling] = max(  # type: ignore
                combo.loc[i - 1, COLS.hit_scaling] * factor, min_other  # type: ignore
            )
        combo.loc[i, COLS.mod_scaling] = max(  # type: ignore
            min_for_hit, combo.loc[i, COLS.hit_scaling]  # type: ignore
        )

    return combo


def naiive_damage_calc(combo: DataFrame) -> DataFrame:
    """Naiive damage calc for a series of moves"""
    # if cull_columns:
    #  combo = combo[[COLS.m_name, COLS.dmg]]
    # Replace any non floats with 0
    combo[COLS.dmg] = combo[COLS.dmg].apply(
        lambda x: int(x) if isinstance(x, str) and x.isnumeric() else 0
    )

    combo = get_combo_scaling(combo)
    combo["scaled_damage"] = combo.apply(
        lambda row: floor(row[COLS.dmg] * row[COLS.mod_scaling]), axis=1
    )
    combo["summed_damage"] = combo.loc[:, "scaled_damage"].cumsum()
    # fill rows with nan if the move does not do damage
    no_damage_rows = combo[COLS.dmg] == 0
    # fill all columns but char and move name with nan

    no_damage_df = combo.drop(columns=[COLS.char, COLS.m_name]).loc[no_damage_rows]
    no_damage_df.loc[:, :] = np.nan
    combo.loc[no_damage_rows, no_damage_df.columns] = no_damage_df

    return combo


def fd_to_combo_df(fd: DataFrame) -> DataFrame:
    # Split damage into individual rows for each hit, damage column contains individual lists of damage for each hit of the move
    fd = fd.explode(COLS.dmg)
    # Pull the character and move name from the mutliindex
    fd[COLS.char] = fd.index.get_level_values(0)
    fd[COLS.m_name] = fd.index.get_level_values(1)
    # Drop the multiindex
    fd = fd.reset_index(drop=True)
    # Assume a move that is the row before the move name "kara" is a kara cancel and does not do damage
    fd.loc[fd[COLS.m_name].shift(-1) == "KARA", "damage"] = 0

    return DataFrame(data=fd, columns=COLS_CLASSES.COMBO_COLS)


@functools.cache
def similar(a: Any, b: Any) -> float:
    return (
        ratio
        if (ratio := SequenceMatcher(None, a, b).real_quick_ratio()) > 0
        else SequenceMatcher(None, a, b).ratio()
    )


@functools.cache
def parse_combo_from_string(character: str, combo_string: str) -> DataFrame:
    # Get the frame data for the character

    character_moves: DataFrame = get_character_moves(character)

    combo_move_names = Series(combo_string.strip().split(" "))
    log.info(f"Parsing combo for [{character}] : {combo_move_names.to_list()}")
    # Attempt to find each move in the combo string in the character's moves from the frame data
    # If a move is not found, check each item in the alt_names list for the move

    # Search character_moves DataFrame for move names or alternative names
    combo_df: DataFrame = find_combo_moves(character_moves, combo_move_names)

    # Find the move with the highest similarity score if multiple matches exist
    # if len(found_moves) > 1:
    # found_moves = best_match(combo_moves, found_moves)

    combo_calc_df: DataFrame = fd_to_combo_df(combo_df)

    return combo_calc_df.reset_index(drop=True)


def best_match(original_move: str, found_moves: Series) -> Series:  # type: ignore
    move_name_similarity: Series = found_moves.apply(  # type: ignore
        lambda move_name: similar(original_move, move_name)
    )
    found_moves = found_moves.loc[move_name_similarity == move_name_similarity.max()]

    return found_moves


@functools.cache
def character_specific_move_name_check(character: str, move_name: str) -> str:
    # sourcery skip: merge-nested-ifs
    if character == CHARS.AN:
        # We don't actually need the move strength for annie divekicks
        if re.search(r"236[lmh]k", move_name, flags=re.IGNORECASE):
            log.debug(f"Removing move strength from annie divekick {move_name}")
            move_name = re.sub(r"[lmh]", "", move_name, flags=re.IGNORECASE)

    return move_name


def get_fd_for_single_move(character_moves: DataFrame, move_name: str) -> Series:  # type: ignore
    """
    Find a single move by name.

    Args:
        character_moves: DataFrame of character's moves
        move_name: Name of the move to find e. g. "5HP"

    Returns:
        pandas. Series of ( character move
    """
    ALIAS_DF = FD_BOT_CSV_MANAGER.dataframes["aliases"]
    move_name = move_name.upper()
    character = str(character_moves.index.get_level_values(0)[0])

    blank_move = Series(index=character_moves.columns, name=(character, move_name))
    move_df = DataFrame(columns=character_moves.columns)

    # Return blank move if it is in the list of ignored moves
    if move_name in skombo.IGNORED_MOVES:
        return blank_move

    in_index = character_moves.index.get_level_values(1) == move_name

    # Find the move in frame data.
    if in_index.max():
        move_df = character_moves[in_index]

    else:
        in_alt_names = character_moves[COLS.a_names].apply(
            lambda move_names: move_name in move_names
        )
        # Check if move name is in alt_names
        name_between_re = re.compile(rf"[^\n]?{move_name}[\n$]", re.IGNORECASE)
        in_alt_names = character_moves["alt_names"].str.contains(
            name_between_re.pattern, regex=True
        )
        # Check if move name is in alt_names
        # log.info(f"Found move name {move_name} in alt_names")
        if not in_alt_names.any():
            # Search alias_df
            in_alias_keys = ALIAS_DF["value"].str.contains(
                name_between_re.pattern, regex=True
            )
            if in_alias_keys.any():
                move_df = character_moves.loc[
                    character_moves["alt_names"].str.startswith(
                        ALIAS_DF.loc[in_alias_keys, "key"].values[0]
                    )
                ]
        else:
            move_df = character_moves[in_alt_names]

    blank_move = Series(index=character_moves.columns, name=(character, move_name))
    return move_df.iloc[0] if move_df.__len__() > 0 else blank_move


def find_move_repeats_follow_ups(moves):
    moves_new: Series = Series()
    annie_divekick_count: int = 0
    xn_re: re.Pattern[str] = re.compile(r"\s?X\s?(\d+)", re.IGNORECASE)

    for move in moves:
        if xn_match := xn_re.search(move):
            xn = int(xn_match[1])

            if moves.name == "ANNIE" and "RE ENTRY" in move.upper():
                annie_divekick_count += 1

            for i in range(xn):
                i_offset = i + annie_divekick_count
                move_name: str = (
                    move.replace(xn_match[0], f" X{str(i_offset + 1)}")
                    if i_offset > 0
                    else move.replace(xn_match[0], "")
                )

                moves_new = pd.concat([moves_new, Series([move_name])])

        elif followup_match := re.search(r"~", move):
            followup_split: list[str] = followup_match.string.split("~")

            for i in range(followup_split.__len__()):
                moves_new = pd.concat(
                    [moves_new, Series("~".join(followup_split[: i + 1]))]
                )

        else:
            moves_new = pd.concat([moves_new, Series([move])])

    # LOG.debug(moves_new)

    return moves_new


def character_specific_operations(
    character: str, combo_df: DataFrame, character_fd: DataFrame
) -> DataFrame:
    if character == "ANNIE":
        annie_divekick = "RE ENTRY"
        if (
            divekick_index := Series(
                combo_df.index.get_level_values(1).str.contains(annie_divekick)
            )
        ).any():
            # Remove all but "True" from divekick_index

            divekick_index = divekick_index[
                divekick_index
            ].index.to_list()  # type: ignore

            divekick_count: int = divekick_index.__len__()

            for i in range(divekick_count):
                divekick_index_cur = divekick_index[i]
                divekick_str: str = (
                    f"{annie_divekick} X{i + 1}" if i > 0 else annie_divekick
                )

                if combo_df.iloc[divekick_index_cur].name[1] != divekick_str:
                    combo_df.iloc[divekick_index_cur] = character_fd.loc[  # type: ignore
                        (character, divekick_str)
                    ]

    return combo_df


def find_combo_moves(character_moves: DataFrame, combo_moves: Series) -> DataFrame:
    # character_moves: DataFrame, combo_moves: Series[Any]
    # Initialize empty DataFrame with columns 'original_move_name' and 'move_name'
    # character mopves columns plus 'character' and 'move_name'

    character: str = str(character_moves.index.get_level_values(0)[0])
    combo_df = DataFrame(columns=character_moves.columns)

    # Interpret things such as 5HPx2 as 5hpx1, 5hpx2
    combo_moves = combo_moves.apply(
        lambda x: character_specific_move_name_check(character, x)
    )
    combo_moves = find_move_repeats_follow_ups(combo_moves.rename(character))

    # Search for each move in the combo string in the frame data
    # Add the move name to the move_name_series DataFrame
    combo_df = pd.concat(
        get_fd_for_single_move(character_moves, move_name).to_frame().T
        for move_name in combo_moves
    )

    combo_df = character_specific_operations(character, combo_df, character_moves)
    return combo_df


@functools.cache
def get_character_moves(character: str) -> DataFrame:
    character_moves: DataFrame = FD.loc[
        FD.index.get_level_values(0) == character.upper()
    ]

    log.debug(f"Retreived {len(character_moves)} moves for {character}")
    return character_moves


def flatten_combo_df(combo: DataFrame) -> DataFrame:
    """Un-explode a combo dataframe, to get information on a per-move instead of per-hit basis"""

    flat_combo = combo.copy()
    # find each time a move changes, assign a new move number to each move
    flat_combo["move_id"] = (
        flat_combo[COLS.m_name].ne(flat_combo[COLS.m_name].shift()).cumsum() - 1
    )
    flat_combo = flat_combo.groupby(
        ["move_id", COLS.m_name, COLS.char], as_index=False
    ).agg(
        {
            COLS.dmg: lambda x: list(x) if any(pd.notna(x)) else np.nan,
            "scaled_damage": lambda x: sum(x) if any(pd.notna(x)) else np.nan,
            # max and min for the first and last hit of the move
            COLS.hit_scaling: lambda x: x.iloc[0],
            COLS.mod_scaling: lambda x: x.iloc[0],
            "summed_damage": max,
        }
    )
    # Get the number of hits for each move
    # Damage is a list of damage values for each hit of the move
    dmg_col = flat_combo[COLS.dmg]
    dmg_list_lens = dmg_col[
        dmg_col.apply(lambda x: isinstance(x, list) and (x.__len__() > 1 or x[0] != 0))
    ].apply(len)
    flat_combo["num_hits"] = dmg_list_lens

    return flat_combo


def parse_combos_from_csv(
    csv_path: str, calc_damage: bool = False
) -> tuple[list[DataFrame], list[int]]:
    """Parse combo from csv file, needs to have columns of COLS.char, "notation", COLS.dmg for testing purposes"""

    log.info(f"========== Parsing combos from csv [{csv_path}] ==========")

    combos_df: DataFrame = pd.read_csv(csv_path)
    log.info(f"Parsing [{len(combos_df.index)}] combos from csv")

    combo_dfs: list[DataFrame] = [
        (
            parse_combo_from_string(
                combos_df[COLS.char][index], combos_df["notation"][index]
            )
        )
        for index in range(len(combos_df.index))
    ]

    combos_expected_damage: list[int] = combos_df[COLS.dmg].tolist()

    if calc_damage:
        combo_dfs = [naiive_damage_calc(combo) for combo in combo_dfs]

    for i, combo in enumerate(combo_dfs):
        combo_output_path = os.path.join(
            skombo.LOG_DIR, f"{combos_df[COLS.char][i]}_{i}.csv"
        )
        combo.to_csv(combo_output_path)

    return combo_dfs, combos_expected_damage


import os

if __name__ == "__main__":
    test_combo_csv_path = skombo.TEST_COMBO_CSVS[0]
    combos, combo_damage = parse_combos_from_csv(test_combo_csv_path, calc_damage=True)

    test = flatten_combo_df(combos[7])

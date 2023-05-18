"""Parsing combo from strings/csvs and calculating the combo's damage/properties
"""

import re
from difflib import SequenceMatcher
from typing import Any

import pandas as pd
from numpy import floor

import skombo
from skombo.fd_ops import COLS, FdBotCsvManager, attempt_to_int, get_fd_bot_data

import functools
from skombo.fd_ops import FD_BOT_CSV_MANAGER

ALIAS_DF = FD_BOT_CSV_MANAGER.dataframes["aliases"]
_log = skombo.LOG


def get_combo_scaling(fd: pd.DataFrame) -> pd.DataFrame:
    """
    Get the scaling factor for the combo. It is used to determine the number of moves that will be taken in a round of damage to get to the next combat hit

    Args:
        fd: pd.DataFrame with hit data in column " damage "
    """
    """Get the scaling factor for the combo"""
    fd = fd.reset_index(drop=True)

    # Turn damage into int
    fd["damage"] = fd["damage"].apply(attempt_to_int)

    # constants
    factor = skombo.SCALING_FACTOR

    min_1k = skombo.SCALING_MIN_1K
    min_other = skombo.SCALING_MIN
    start = skombo.SCALING_START

    # Scaling is 1.0 for first 3 hits
    # Then 0.875 less for each hit after that
    # Minimum scaling is 0.2
    # Minimum scaling for moves with over 1000 base damage is 0.275
    fd["scaling_for_hit"] = 0.0
    fd["scaling_after_modifiers"] = 0.0
    for move in fd.itertuples():
        min_for_hit: float = min_other
        skip_row: int = 0
        index = move.Index
        name = move.move_name
        damage = move.damage

        if index == 0:
            fd.loc[index, "scaling_for_hit"] = start
        elif not isinstance(damage, int) or damage == 0:
            fd.loc[index, "scaling_for_hit"] = fd.iloc[index - 1]["scaling_for_hit"]
            skip_row += 1
        elif index - skip_row < 3:
            fd.loc[index, "scaling_for_hit"] = start
        else:
            min_for_hit = min_1k if damage >= 1000 else min_for_hit
            fd.loc[index, "scaling_for_hit"] = max(
                fd.iloc[index - 1]["scaling_for_hit"] * factor, min_other
            )
        fd.loc[index, "scaling_after_modifiers"] = max(
            min_for_hit, fd.loc[index, "scaling_for_hit"]  # type: ignore
        )
    # Return the scaling factor added to the frame data

    return fd


def naiive_damage_calc(combo: pd.DataFrame, cull_columns: bool = True) -> pd.DataFrame:
    """Naiive damage calc for a series of moves"""
    # if cull_columns:
    #  combo = combo[["move_name", "damage"]]
    combo = get_combo_scaling(combo)
    combo["damage"] = combo["damage"].apply(attempt_to_int)
    # Replace any non floats with 0
    combo["damage"] = (
        combo["damage"].apply(lambda x: x if isinstance(x, int) else 0).astype(float)
    )
    combo["scaled_damage"] = combo.apply(
        lambda row: floor(row["damage"] * row["scaling_after_modifiers"]), axis=1
    )
    combo["summed_damage"] = combo.loc[:, "scaled_damage"].cumsum()

    return combo


def parse_combos_from_csv(
    csv_path: str, calc_damage: bool = False
) -> tuple[list[pd.DataFrame], list[int]]:
    """Parse combo from csv file, needs to have columns of "character", "notation", "damage" for testing purposes"""

    _log.info(f"========== Parsing combos from csv [{csv_path}] ==========")

    combos_df: pd.DataFrame = pd.read_csv(csv_path)
    _log.info(f"Parsing [{len(combos_df.index)}] combos from csv")

    combo_dfs: list[pd.DataFrame] = [
        (
            parse_combo_from_string(
                combos_df["character"][index], combos_df["notation"][index]
            )
        )
        for index in range(len(combos_df.index))
    ]

    combos_expected_damage: list[int] = combos_df["damage"].tolist()

    if calc_damage:
        combo_dfs = [naiive_damage_calc(combo) for combo in combo_dfs]

    return combo_dfs, combos_expected_damage


def fd_to_combo_df(fd: pd.DataFrame) -> pd.DataFrame:
    """
    Takes pure frame data and character pd.DataFrame and turns it into a pd.DataFrame that can be used for combo calculations

    Args:
        fd: pure frame data to be converted
    Returns:
        converted pd.DataFrame
    """

    combo_df: pd.DataFrame = fd.copy().drop(
        ["alt_names", "footer", "thumbnail_url", "footer_url"], axis=1
    )

    # Split damage into individual rows for each hit, damage column contains individual lists of damage for each hit of the move

    combo_df = combo_df.explode("damage")

    # Assume a move that is the row before the move name "kara" is a kara cancel and does not do damage
    combo_df.loc[combo_df["move_name"].shift(-1) == "KARA", "damage"] = 0

    return combo_df


@functools.cache
def similar(a: Any, b: Any) -> float:
    """
    Compares two sequences and returns the ratio of similar elements. This is useful for comparing a set of elements to see if they are similar or not.

    Args:
        a: The first sequence to compare. Can be any type that can be converted to a : class : ` str `.
        b: The second sequence to compare. Can be any type that can be converted to a : class : ` str `.

    Returns:
        A float between 0 and 1 indicating the similarity of the two sequences. If a ratio cannot be determined, 0 is returned.
    """
    return (
        ratio
        if (ratio := SequenceMatcher(None, a, b).real_quick_ratio()) > 0
        else SequenceMatcher(None, a, b).ratio()
    )


@functools.cache
def parse_combo_from_string(character: str, combo_string: str) -> pd.DataFrame:
    # Get the frame data for the character

    character_moves: pd.DataFrame = get_character_moves(character)

    combo_move_names = pd.Series(combo_string.strip().split(" "))
    _log.info(f"Parsing combo for [{character}] : {combo_move_names.to_list()}")
    # Attempt to find each move in the combo string in the character's moves from the frame data
    # If a move is not found, check each item in the alt_names list for the move

    # Search character_moves pd.DataFrame for move names or alternative names
    combo_df: pd.DataFrame = find_combo_moves(character_moves, combo_move_names)

    # Find the move with the highest similarity score if multiple matches exist
    # if len(found_moves) > 1:
    # found_moves = best_match(combo_moves, found_moves)

    combo_calc_df: pd.DataFrame = fd_to_combo_df(combo_df)

    return combo_calc_df.reset_index(drop=True)


def best_match(original_move: str, found_moves: pd.Series) -> pd.Series:  # type: ignore
    move_name_similarity: pd.Series = found_moves.apply(
        lambda move_name: similar(original_move, move_name)
    )
    found_moves = found_moves.loc[move_name_similarity == move_name_similarity.max()]

    return found_moves


def character_specific_move_name_check(character: str, move_name: str) -> str:
    # sourcery skip: merge-nested-ifs
    if character == "ANNIE":
        # We don't actually need the move strength for annie divekicks
        if re.search(r"236[lmh]k", move_name, flags=re.IGNORECASE):
            _log.debug(f"Removing move strength from annie divekick {move_name}")
            move_name = re.sub(r"[lmh]", "", move_name, flags=re.IGNORECASE)

    return move_name


def get_fd_for_single_move(character_moves: pd.DataFrame, move_name: str) -> pd.Series:
    """
    Find a single move by name.

    Args:
        character_moves: pd.DataFrame of character's moves
        move_name: Name of the move to find e. g. "5HP"

    Returns:
        pandas. Series of ( character move
    """
    move_name = move_name.upper()
    character = str(character_moves.index.get_level_values(0)[0])

    blank_move = pd.Series(index=character_moves.columns, name=(character, move_name))

    # Return blank move if it is in the list of ignored moves
    if move_name in skombo.IGNORED_MOVES:
        return blank_move

    in_index = character_moves.index.get_level_values(1) == move_name

    # Find the move in frame data.
    if in_index.max():  # type: ignore
        # log.info(f"Found move name {move_name} in frame data")
        move_df = character_moves[in_index]

    else:
        name_between_re = re.compile(rf"[^,]?{move_name}[,$]", re.IGNORECASE)
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
        move_df = character_moves[in_alt_names]

    blank_move = pd.Series(index=character_moves.columns, name=(character, move_name))
    return move_df.iloc[0] if move_df.__len__() > 0 else blank_move


def find_move_repeats_follow_ups(moves: pd.Series):
    moves_new: pd.Series = pd.Series()
    annie_divekick_count: int = 0
    xn_re: re.Pattern[str] = re.compile(r"\s?X\s?(\d+)", re.IGNORECASE)

    for move in moves:
        if xn_match := xn_re.search(move):
            # log.info(f"Found move repeat {xn_match[1]} in {move}")

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


def character_specific_operations(
    character: str, combo_df: pd.DataFrame, character_fd: pd.DataFrame
) -> pd.DataFrame:
    if character == "ANNIE":
        annie_divekick = "RE ENTRY"
        if (divekick_index := combo_df["move_name"].str.contains(annie_divekick)).any():
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

                if combo_df.loc[divekick_index_cur, "move_name"] != divekick_str:
                    combo_df.loc[divekick_index_cur] = character_fd.loc[  # type: ignore
                        (character, divekick_str)
                    ]
                    combo_df.loc[divekick_index_cur, "move_name"] = divekick_str
                    combo_df.loc[divekick_index_cur, "character"] = character

    return combo_df


def find_combo_moves(
    character_moves: pd.DataFrame, combo_moves: pd.Series
) -> pd.DataFrame:
    # Initialize empty pd.DataFrame with columns 'original_move_name' and 'move_name'
    # character mopves columns plus 'character' and 'move_name'
    combo_df_cols: list[str] = ["character", "move_name"] + list(
        character_moves.columns
    )
    character: str = str(character_moves.index.get_level_values(0)[0])
    combo_df: pd.DataFrame = pd.DataFrame(columns=combo_df_cols)

    # Interpret things such as 5HPx2 as 5hpx1, 5hpx2
    combo_moves = combo_moves.apply(
        lambda x: character_specific_move_name_check(character, x)
    )
    combo_moves = find_move_repeats_follow_ups(combo_moves.rename(character))

    # Search for each move in the combo string in the frame data
    # Add the move name to the move_name_series pd.DataFrame
    for i, move_name in enumerate(combo_moves):
        # If move_name isn't a string then skip to next move in the list

        move_series = get_fd_for_single_move(character_moves, move_name).fillna("")

        # Add the move name to the move_name_series pd.DataFrame
        combo_df.loc[i, :] = move_series
        combo_df.at[i, "character"] = move_series.name[0]  # type: ignore
        combo_df.at[i, "move_name"] = move_series.name[1]  # type: ignore

    combo_df = character_specific_operations(character, combo_df, character_moves)
    return combo_df


@functools.cache
def get_character_moves(character: str) -> pd.DataFrame:
    fd: pd.DataFrame = get_fd_bot_data()
    character_moves: pd.DataFrame = fd.loc[
        fd.index.get_level_values(0) == character.upper()
    ]

    # log.info(f"Retreived {len(character_moves)} moves for {character}")
    return character_moves

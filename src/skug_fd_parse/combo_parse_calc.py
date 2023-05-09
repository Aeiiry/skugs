"""Parsing combo from strings/csvs and calculating the combo's damage/properties
"""
import pandas as pd

from skug_fd_parse import frame_data_operations as fdo
from skug_fd_parse.skug_logger import log

global fd


DataFrame = pd.DataFrame


def parse_combos_from_csv(csv_path: str) -> DataFrame:
    """Parse combo from csv file, needs to have columns of "character", "notation", "damage" for testing purposes"""
    combos = pd.DataFrame()
    combo_csv_df: pd.DataFrame = pd.read_csv(csv_path)

    # Each line is a combo
    for _, combo in combo_csv_df.iterrows():
        character: str = combo["character"]
        notation: str = combo["notation"]
        expected_damage: str = combo["damage"]

        parsed_combo: DataFrame = parse_combo_from_string(character, notation)
        pd.concat([combos, parsed_combo], ignore_index=True)
    return combos

def fd_rows_to_combo_calc_df(fd_rows: DataFrame) -> DataFrame:
    """This is a function primarily to turn pure frame data rows into a dataframe that can be used for combo calculations
    Individual hits will be separated out, and the damage will be calculated for each hit alongside considerations for what current move the combo is on"""

    # First we need to expand on moves that are technically multiple moves, e.g 5HPx2 -> 5HP, 5HPx2. 236K~K -> 236K, 236K~K

    



    return fd_rows
    
def parse_combo_from_string(character: str, combo_string: str) -> pd.DataFrame:
    combo: DataFrame = pd.DataFrame()
    character_moves: DataFrame = get_character_moves(character)

    combo_move_names: list[str] = combo_string.strip().split(" ")
    log.info(f"combo_moves: {combo_move_names}")

    combo_move_names = [move.lower() for move in combo_move_names]

    # Attempt to find each move in the combo string in the character's moves from the frame data
    # If a move is not found, check each item in the alt_names list for the move

    for move in combo_move_names:
        # Pandas vectorized string operations
        found_move: DataFrame = character_moves.loc[
            character_moves["move_name"].str.upper() == move.upper()
        ]
        if found_move.empty:
            found_move = character_moves.loc[
                character_moves["alt_names"].apply(
                    lambda x: move.upper() in (d.upper() for d in x)
                )
            ]
        if found_move.empty:
            log.warning(f"Could not find move: {move}")
            break
        move_name: str = found_move["move_name"].values[0]
        log.info(f"Found move: {move_name}")
        combo = pd.concat([combo, found_move])

    combo_calc_df = fd_rows_to_combo_calc_df(combo)
    
    return combo


def get_character_moves(character: str) -> pd.DataFrame:
    fd = fdo.get_fd_bot_data()
    character_moves: DataFrame = fd[fd["character"].str.lower() == character.lower()]

    log.info(f"Retreived {len(character_moves)} moves for {character}")
    return character_moves

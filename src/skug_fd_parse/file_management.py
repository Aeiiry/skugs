import os
from typing import Literal

DATA_FOLDER: Literal["data"] = "data"
MODULE_FOLDER = "skug_fd_parse"  # === WE ARE HERE ===

TOP_LEVEL_FOLDER: Literal["skugs"] = "skugs"

# Search for the top module folder
# Only search 10 levels up or down

FILE_PATH: str = os.path.abspath(__file__)
MODULE_FOLDER_PATH: str = os.path.dirname(FILE_PATH)
DATA_PATH: str = os.path.join(MODULE_FOLDER_PATH, r"..\..", DATA_FOLDER)


CSV_PATH: str = os.path.join(DATA_PATH, "csvs")

GAME_DATA_PATH: str = os.path.join(CSV_PATH, "game_data")

FD_BOT_FILE_PREFIX = "SG2E - Frame Data Bot Data - "

CHARACTER_DATA_PATH: str = os.path.join(
    GAME_DATA_PATH, f"{FD_BOT_FILE_PREFIX}Characters.csv"
)
FRAME_DATA_PATH: str = os.path.join(GAME_DATA_PATH, f"{FD_BOT_FILE_PREFIX}Moves.csv")
MOVE_NAME_ALIASES_PATH: str = os.path.join(
    GAME_DATA_PATH, f"{FD_BOT_FILE_PREFIX}Macros.csv"
)

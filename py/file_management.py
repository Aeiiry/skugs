import constants
import os
from typing import Literal

DATA_FOLDER: Literal["data"] = "data"
PY_FOLDER: Literal["py"] = "py"

TOP_LEVEL_FOLDER: Literal["skugs"] = "skugs"

# Generate paths
DATA_PATH: str = os.path.join(os.getcwd(), "..", DATA_FOLDER)
PY_PATH: str = os.path.join(os.getcwd(), "..", PY_FOLDER)

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

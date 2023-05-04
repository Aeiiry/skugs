import os

SRC_NAME = "src"
DATA_NAME = "data"
MODULE_NAME: str = __name__.split(".")[0]
MODULE_PATH: str = os.path.join(os.getcwd(), SRC_NAME, MODULE_NAME)
DATA_PATH: str = os.path.join(MODULE_PATH, DATA_NAME)

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

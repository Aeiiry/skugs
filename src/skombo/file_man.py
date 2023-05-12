import os

FD_BOT_FILE_PREFIX = "SG2E - Frame Data Bot Data - "
DATA_NAME = "data"
CSVS = "csvs"
GAME_DATA = "game_data"

ABS_PATH: str = os.path.abspath(os.path.dirname(__file__))

MODULE_NAME: str = os.path.basename(ABS_PATH)

DATA_PATH: str = os.path.join(ABS_PATH, DATA_NAME)
CSV_PATH: str = os.path.join(DATA_PATH, CSVS)
GAME_DATA_PATH: str = os.path.join(CSV_PATH, GAME_DATA)

CHARACTER_DATA_PATH: str = os.path.join(
    GAME_DATA_PATH, f"{FD_BOT_FILE_PREFIX}Characters.csv"
)
FRAME_DATA_PATH: str = os.path.join(GAME_DATA_PATH, f"{FD_BOT_FILE_PREFIX}Moves.csv")
MOVE_NAME_ALIASES_PATH: str = os.path.join(
    GAME_DATA_PATH, f"{FD_BOT_FILE_PREFIX}Macros.csv"
)

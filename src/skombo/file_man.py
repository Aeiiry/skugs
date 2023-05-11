import os

SRC_NAME = "src"
DATA_NAME = "data"
MODULE_NAME: str = __name__.split(".")[0]
# change dir to current file's directory
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

MODULE_PATH: str = os.getcwd()
# for data path we need to go up until we find the src folder
# then go into the module folder then into the data folder
search_path = MODULE_PATH
while (
        (search_path := os.path.split(search_path)[1]) != SRC_NAME
        and search_path != ""
        and ".tox" not in search_path
):
    search_path = os.path.split(search_path)[0]

search_path = os.path.join(os.path.abspath(search_path), SRC_NAME)
DATA_PATH: str = os.path.join(search_path, MODULE_NAME, DATA_NAME)
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

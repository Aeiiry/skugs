# Two sets of user input are required:
# Character name, and combo notation + damage

# Character name
from loguru import logger as log
import sys
import pandas as pd
import re
import skombo
import os


def get_character_name() -> str:
    """Get character name from user"""
    return input("Enter character name: ")


def get_combo_notation():
    """Get combo notation from user"""
    # Show prompt
    print("Enter combo notation (Ctrl+Z, Enter to finish):")

    return re.sub(r"\s+", " ", "".join(sys.stdin.readlines()))


def get_damage():
    """Get damage from user"""
    return input("Enter damage: ")


while True:
    # Check if a test combo csv exists
    test_data = skombo.TESTS_DATA_PATH
    csv_row = {
        "character": get_character_name(),
        "notation": get_combo_notation(),
        "expected_damage": get_damage(),
    }

    csv_path = os.path.join(
        test_data, f"{csv_row['character']}{skombo.TEST_COMBOS_SUFFIX}"
    )

    try:
        df = pd.read_csv(csv_path)
        # set columns to combo input cols
        df = pd.concat([df, pd.DataFrame(csv_row, index=[0])], ignore_index=True)
        df = pd.DataFrame(df, columns=list(skombo.COMBO_INPUT_COLS.__dict__.values()))
    except FileNotFoundError:
        # If not, create a new one
        df = pd.DataFrame(columns=list(skombo.COMBO_INPUT_COLS.__dict__.values()))
        df = pd.concat([df, pd.DataFrame(csv_row, index=[0])], ignore_index=True)

    df.to_csv(csv_path, index=False)

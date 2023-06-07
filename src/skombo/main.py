import atexit
import datetime
from email import utils

from loguru import logger as log

import skombo

from skombo.combo_calc import ComboCalculator
from skombo.fd_ops import get_fd_bot_character_manager

START_TIME = datetime.datetime.now()


@atexit.register
def exit_handler() -> None:
    # get the end datetime
    end_time = datetime.datetime.now()

    # get execution time
    elapsed_time = end_time - START_TIME
    log.info(f"Execution time: {elapsed_time} seconds 🤠")


combo_calc = ComboCalculator(get_fd_bot_character_manager(), skombo.TEST_COMBO_CSVS[0])

combo_calc.process_combos()

combo_calc.character_manager.frame_data.to_csv("fd_cleaned.csv")

fontpaths =skombo.utils.get_font_paths()

log.info("Done!")

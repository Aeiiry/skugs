import atexit
from loguru import logger as log
import datetime
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


calc = ComboCalculator(get_fd_bot_character_manager(), skombo.TEST_COMBO_CSVS[0])

calc.process_combos()

calc.character_manager.frame_data.to_csv("fd_cleaned.csv")


log.info("Done!")

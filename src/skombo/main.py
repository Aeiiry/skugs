import atexit
import cProfile
import datetime
import os
import pstats

from loguru import logger as log

import skombo
from skombo.combo_calc import ComboCalculator
from skombo.fd_ops import get_fd_bot_character_manager

START_TIME = datetime.datetime.now()


@atexit.register
def exit_handler() -> None:
    # get the end datetime
    END_TIME = datetime.datetime.now()

    # get execution time
    elapsed_time = END_TIME - START_TIME
    log.info(f"Execution time: {elapsed_time} seconds 🤠")


prof = cProfile.Profile(subcalls=True, builtins=False)
prof.enable()


combo_calc = ComboCalculator(get_fd_bot_character_manager(), skombo.TEST_COMBO_CSVS[0])

combo_calc.process_combos()

combo_calc.character_manager.frame_data.to_csv("fd_cleaned.csv")

prof.disable()
stats = pstats.Stats(prof).sort_stats("tottime")
# stats.print_stats(10)
stats.dump_stats(os.path.join(skombo.LOG_DIR, f"{skombo.MODULE_NAME}.prof"))

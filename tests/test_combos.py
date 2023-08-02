import pytest
from pyannotate_runtime import collect_types

from skombo.combo_calc import ComboCalculator

from skombo.fd_ops import get_fd_bot_character_manager

collect_types.init_types_collection()

from loguru import logger as log

from skombo.utils import expand_all_x_n
import skombo


def test_main():
    combo_calc = ComboCalculator(
        get_fd_bot_character_manager(), skombo.TEST_COMBO_CSVS[0]
    )
    combo_calc.process_combos()

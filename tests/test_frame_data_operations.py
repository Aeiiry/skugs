import skug_fd_parse
import pytest
from skug_fd_parse import frame_data_operations as fdo

from skug_fd_parse.frame_data_operations import (
    attempt_to_int,
    remove_spaces,
    separate_damage,
    expand_all_x_n,
    expand_x_n,
    apply_to_columns,
    clean_frame_data,
)

import pandas as pd


def test_fdo_main():
    assert fdo.main() == 0


def test_expand_all_x_n():
    assert expand_all_x_n("1x2") == "1,1"
    assert expand_all_x_n("1x2,2x3") == "1,1,2,2,2"
    assert expand_all_x_n("1x2,2x3,3x4") == "1,1,2,2,2,3,3,3,3"
    assert expand_all_x_n("100,101,60x2,20") == "100,101,60,60,20"
    assert expand_all_x_n("25x2,10x3(5x3) [10x2(5x2)]") == "25,25,10,10,10(5,5,5) [10,10(5,5)]"

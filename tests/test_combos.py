import pytest
from pyannotate_runtime import collect_types

collect_types.init_types_collection()

from loguru import logger as log

from skombo.utils import expand_all_x_n


@pytest.mark.parametrize(
    "input_str, expected_output",
    [
        ("1x2", "1,1"),
        ("1x2,2x3", "1,1,2,2,2"),
        ("1x2,2x3,3x4", "1,1,2,2,2,3,3,3,3"),
        ("100,101,60x2,20", "100,101,60,60,20"),
        ("25x2,10x3(5x3) [10x2(5x2)]", "25,25,10,10,10(5,5,5)[10,10(5,5)]"),
        ("10x3(5x3) [10x2(5x2)] 25x2", "10,10,10(5,5,5)[10,10(5,5)]25,25"),
    ],
)
def test_expand_all_x_n(input_str, expected_output) -> None:
    calculated_output = expand_all_x_n(input_str)
    log.info(f"Expanded ( {input_str} ) ==> ( {calculated_output} )")
    assert calculated_output == expected_output

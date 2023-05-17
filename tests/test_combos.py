import pathlib
import warnings

import pytest
from tabulate import tabulate

from skombo import *
from skombo.combo_calc import parse_combos_from_csv

# disable mypy for this file
# mypy: ignore-errors
from skombo.frame_data_operations import expand_all_x_n


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


@pytest.mark.parametrize(
    "test_csv_path, character",
    [
        (
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                TEST_DATA_FOLDER,
                character.lower() + TEST_COMBOS_SUFFIX,
            ),
            character,
        )
        for character in CHARS.__dict__
    ],
)
def test_combos(test_csv_path: str, character: str, profile) -> None:
    log.info(f"Testing combos for [[{character}]]")
    if not pathlib.Path(test_csv_path).is_file():
        log.warning(f"!!!!! No test file found for [[{character}]] !!!!!")
        warnings.warn(f"!!!!! No test file found for [[{character}]] !!!!!")
        pytest.skip(f"Test file not found: {test_csv_path}")

    combos, combo_damage = parse_combos_from_csv(test_csv_path, calc_damage=True)

    for i, combo in enumerate(combos):
        log.info(f"Combo: \n{tabulate(combo, headers='keys', tablefmt='psql')}")  # type: ignore
        damage_diff: int = (
            combo_damage[i] - combo.at[combo.__len__() - 1, "summed_damage"]
        )

        damage_diff_percent: float = round(abs(damage_diff / combo_damage[i] * 100), 2)
        log.info(f"Damage difference: {damage_diff_percent}%")

        assert 1 > damage_diff_percent >= 0

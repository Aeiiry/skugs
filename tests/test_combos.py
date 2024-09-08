import pytest
from pyannotate_runtime import collect_types
from skombo.utils import split_meter
from skombo.utils import expand_all_x_n
import pandas as pd
from skombo.combo_calc import flatten_combo_df
from skombo import FD_COLS


def test_expand_all_x_n():
    """Test the expand_all_x_n utility function."""
    result = expand_all_x_n("250 x4")
    assert result == "250,250,250,250"


def test_group_and_aggregate_combo_data():
    data = {
        FD_COLS.m_name: ["move1", "move1", "move2", "move2"],
        FD_COLS.char: ["char1", "char1", "char1", "char1"],
        FD_COLS.dmg: [10, 20, 30, 40],
        "scaled_damage": [5, 10, 15, 20],
        FD_COLS.hit_scaling: [0.9, 0.8, 0.7, 0.6],
        FD_COLS.mod_scaling: [1.0, 1.0, 1.0, 1.0],
        "summed_damage": [10, 20, 30, 40],
    }
    combo_df = pd.DataFrame(data)
    result = flatten_combo_df(combo_df)

    assert len(result) == 2
    assert result.loc[0, FD_COLS.m_name] == "move1"
    assert result.loc[1, FD_COLS.m_name] == "move2"
    assert result.loc[0, "scaled_damage"] == 15
    assert result.loc[1, "scaled_damage"] == 35


def test_correct_split_with_parentheses():

    result = split_meter("example(on_whiff)")
    assert result == ("example", "on_whiff")

# Run main.py
def test_main():
    collect_types.init_types_collection()
    from skombo import main
    collect_types.dump_stats("type_info.json")
    assert main is not None



if __name__ == "__main__":
    pytest.main()

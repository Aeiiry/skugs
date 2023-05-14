import cProfile
import os
import pstats

import pandas as pd

from skombo import combo as combo_moves
from skombo import sklog as sklog
from skombo.file_man import CSV_PATH, DATA_PATH, MODULE_NAME, ABS_PATH

log = sklog.get_logger()


def main() -> None:
    csv_path: str = os.path.join(CSV_PATH, "annie_combos.csv")
    combos: list[tuple[pd.DataFrame, int]] = combo_moves.parse_combos_from_csv(csv_path)

    for i, combo in enumerate(combos):
        combo_df = combo[0]
        expected_damage: int = combo[1]

        calculated_damage: float

        calculated_damage, combo_df = combo_moves.naiive_damage_calc(combo_df)
        calculated_damage = round(calculated_damage)
        diff_percentage = round((calculated_damage - expected_damage) / expected_damage * 100, 2)
        log.info(
            f"Combo [{i + 1}] did [{calculated_damage}] damage, expected damage was [{expected_damage}][Diff: {diff_percentage}%]"
        )
        # create debug dir if it doesn't exist
        if not os.path.exists("debug"):
            os.makedirs("debug")
        combo_df.to_csv(
            f"debug/combo{i + 1}.csv",
        )


if __name__ == "__main__":
    log.info("========== Main started ==========")
    log.info(f"MODULE_NAME: {MODULE_NAME}")

    log.info(f"MODULE_PATH: {ABS_PATH}")

    log.info(f"CSV_PATH: {CSV_PATH}")

    log.info(f"DATA_PATH: {DATA_PATH}")
    pd.options.display.max_rows = None  # type: ignore
    pd.options.display.max_columns = None  # type: ignore
    pd.options.display.width = None  # type: ignore
    pd.options.display.max_colwidth = 25
    with cProfile.Profile() as pr:
        main()

    stats = pstats.Stats(pr)
    stats.dump_stats("skug.prof")
    log.info("========== Main finished ==========")

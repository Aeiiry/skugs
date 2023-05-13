import os

import pstats, cProfile
import pandas as pd
from skombo import combo as combo_moves
from skombo import sklog as sklog
from skombo.file_man import CSV_PATH, DATA_PATH, MODULE_NAME, ABS_PATH

log = sklog.get_logger()


def main() -> None:
    csv_path = os.path.join(CSV_PATH, "annie_combos.csv")
    combos = combo_moves.parse_combos_from_csv(csv_path)

    for i, combo in enumerate(combos):
        combo_df = combo[0]
        combo_damage = combo[1]
        damage, combo_df = combo_moves.naiive_damage_calc(combo_df)
        damage = round(damage)

        log.info(
            f"Combo [{i}] did [{damage}] damage, expected damage was [{combo_damage}]"
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

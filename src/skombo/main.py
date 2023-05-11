import pstats
import sys
import yappi

from skombo import combo as combo_moves
from skombo import sklog as sklog
from skombo.file_man import CSV_PATH, DATA_PATH, MODULE_NAME, MODULE_PATH
import os

log = sklog.get_logger()
yappi.set_clock_type("cpu")  # Use set_clock_type("wall") for wall time

yappi.start()


def main() -> None:
    csv_path = os.path.join(CSV_PATH, "annie_combos.csv")
    combos = combo_moves.parse_combos_from_csv(csv_path)


if __name__ == "__main__":
    print("Running main.py")
    log.info(f"MODULE_NAME: {MODULE_NAME}")

    log.info(f"MODULE_PATH: {MODULE_PATH}")

    log.info(f"CSV_PATH: {CSV_PATH}")

    log.info(f"DATA_PATH: {DATA_PATH}")

    main()
    stats = yappi.get_func_stats()

    stats.save("skug.prof", type="pstat")

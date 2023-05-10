import cProfile
import pstats
from skombo.file_man import CSV_PATH, DATA_PATH, MODULE_NAME, MODULE_PATH

from skombo import combo as combo_moves
from skombo import sklog as sklog

log = sklog.get_logger()


def main() -> None:
    combos = combo_moves.parse_combos_from_csv(
        "src\\skombo\\data\\csvs\\annie_combos.csv"
    )


if __name__ == "__main__":
    log.info(f"MODULE_NAME: {MODULE_NAME}")

    log.info(f"MODULE_PATH: {MODULE_PATH}")

    log.info(f"CSV_PATH: {CSV_PATH}")

    log.info(f"DATA_PATH: {DATA_PATH}")

    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats = stats.strip_dirs()
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats("skombo")
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_callers("concat", 10)
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_callees("concat", 10)

    stats.dump_stats("skug.prof")

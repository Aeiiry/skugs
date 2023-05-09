import cProfile
import pstats
from skug_fd_parse.file_man import CSV_PATH, DATA_PATH, MODULE_NAME, MODULE_PATH

from skug_fd_parse import combo as combo_moves
from skug_fd_parse import sklog as sklog

log = sklog.get_logger()


def main() -> None:
    combos = combo_moves.parse_combos_from_csv(
        "src\\skug_fd_parse\\data\\csvs\\annie_combos.csv"
    )


if __name__ == "__main__":
    log.info(f"MODULE_NAME: {MODULE_NAME}")

    log.info(f"MODULE_PATH: {MODULE_PATH}")

    log.info(f"CSV_PATH: {CSV_PATH}")

    log.info(f"DATA_PATH: {DATA_PATH}")

    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats("skug_fd_parse")
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_callers("concat", 10)
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_callees("concat", 10)

    stats.dump_stats("skug.prof")

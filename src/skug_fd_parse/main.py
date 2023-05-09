import cProfile
import pstats

from skug_fd_parse import combo_parse_calc as combo_move_names


def main() -> None:
    with cProfile.Profile() as pr:
        combos = combo_move_names.parse_combos_from_csv(
            "src\\skug_fd_parse\\data\\csvs\\annie_combos.csv"
        )
    stats = pstats.Stats(pr)

    stats.dump_stats("skug.prof")


if __name__ == "__main__":
    main()

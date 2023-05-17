import os

import skombo
from skombo.combo_calc import parse_combos_from_csv
import cProfile
import pstats

log = skombo.log


def main() -> None:
    combos = parse_combos_from_csv(
        os.path.join(
            skombo.ABS_PATH,
            (skombo.CHARS.AN.lower() + skombo.TEST_COMBOS_SUFFIX),
        ),
        calc_damage=True,
    )

    return None


if __name__ == "__main__":
    main()

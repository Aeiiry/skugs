import os

import skombo
from skombo.combo_calc import parse_combos_from_csv

_log = skombo.LOG


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

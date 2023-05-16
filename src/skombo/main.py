import os

import skombo
from skombo.combo_calc import parse_combos_from_csv


def main() -> None:
    parse_combos_from_csv(
        os.path.join(
            skombo.ABS_PATH,
            (skombo.CHARACTERS["AN"].lower() + skombo.TEST_COMBOS_SUFFIX),
        ),
        calc_damage=True,
    )
    return None


if __name__ == "__main__":
    main()

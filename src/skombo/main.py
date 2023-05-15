from ast import Slice

import pandas as pd
from tabulate import tabulate

from skombo import *
from skombo.frame_data_operations import get_fd_bot_data


def main() -> None:
    return None


def hacky_sort_thing():
    fd: pd.DataFrame = get_fd_bot_data()
    fd_copy = fd.copy()
    fd = fd[
        ~fd["properties"].apply(
            lambda x: re.search(r"LAUNCH|KNOCKDOWN|KD|SPLAT|BOUNCE", x, re.IGNORECASE)
            is not None
            if isinstance(x, str)
            else False
        )
    ]
    fd = fd[
        ~fd["on_hit_effect"].apply(
            lambda x: re.search(r"LAUNCH|KNOCKDOWN|KD|SPLAT|BOUNCE", x, re.IGNORECASE)
            is not None
            if isinstance(x, str)
            else False
        )
    ]
    fd = fd[fd["move_category"].str.contains(r"NORMAL|SPECIAL", regex=True)]

    # Drop all rows containing non list[int] values
    fd = fd[fd["damage"].apply(lambda x: isinstance(x, list))]
    fd = fd[fd["startup"].apply(lambda x: isinstance(x, list))]

    # Drop all rows containing lists with any non int values
    fd = fd[fd["damage"].apply(lambda x: all(isinstance(i, int) for i in x))]
    fd = fd[fd["startup"].apply(lambda x: all(isinstance(i, int) for i in x))]

    # Drop all rows containing empty lists
    fd = fd[fd["damage"].apply(lambda x: len(x) > 0)]
    fd = fd[fd["startup"].apply(lambda x: len(x) > 0)]

    fd["made_up_value"] = fd.apply(
        lambda x: round(
            x["damage"][0] / (x["startup"][0] + x["hitstun"][0] + x["hitstop"][0]), 2
        )
        if all(
            isinstance(i, list)
            for i in [x["damage"], x["startup"], x["hitstun"], x["hitstop"]]
        )
        and all(
            isinstance(i, int)
            for i in [x["damage"][0], x["startup"][0], x["hitstun"][0], x["hitstop"][0]]
        )
        and x["startup"][0] > 0
        else 0,
        axis=1,
    )

    # Filter out index 1 names that start with J
    fd = fd[fd.index.get_level_values(1).str.contains(r"^[^J]", regex=True)]
    fd = fd[~fd.index.get_level_values(1).str.contains(r"[xX]\s?\d", regex=True)]
    pd.options.display.max_rows = None  # type: ignore
    # Sort by index 0 (string value) nd made_up_value, select top 10

    fd["damage"] = fd["damage"].apply(lambda x: x[0])
    fd["startup"] = fd["startup"].apply(lambda x: x[0])
    fd["hitstun"] = fd["hitstun"].apply(lambda x: x[0] if isinstance(x, list) else x)
    fd["hitstop"] = fd["hitstop"].apply(lambda x: x[0] if isinstance(x, list) else x)
    fd = fd.loc[
        :,
        [
            "made_up_value",
            "damage",
            "startup",
            "hitstun",
            "hitstop",
            "properties",
            "on_hit_effect",
        ],
    ]
    fd = fd.groupby(fd.index.levels[0].values).head(10)  # type: ignore
    fd = fd.sort_values(["made_up_value"], ascending=False)

    log.info(f"\n{fd}")


if __name__ == "__main__":
    main()

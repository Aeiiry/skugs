import cProfile
import os
import pstats
import re
from typing import Any, Callable, List

import pandas as pd
from pandas import Index
from pandas.core.frame import DataFrame
from pandas.core.series import Series

import skug_fd_parse.constants as const
import skug_fd_parse.file_management as fm
from skug_fd_parse.skug_logger import log


def attempt_to_int(value: str | int) -> str | int:
    return int(value) if isinstance(value, str) and value.isnumeric() else value


def remove_spaces(string: str) -> str:
    return string.replace(" ", "") if isinstance(string, str) else string


def separate_damage(string: str) -> List[str]:
    string = remove_spaces(string)
    return string.split(",")


def expand_all_x_n(damage: str) -> str:
    if isinstance(damage, str):
        while True:
            if x_n_match := const.RE_X_N.search(damage):
                damage = expand_x_n(x_n_match)
            elif x_n_brackets_matches := const.RE_BRACKETS_X_N.search(damage):
                damage = expand_x_n(x_n_brackets_matches)
            else:
                break
    return damage


def expand_x_n(match: re.Match[str]) -> str:
    num = int(match.group(3))
    damage: str = match.group(1).strip()
    original_damage: str = match.group(0)
    if "[" in original_damage:
        damage = re.sub(r"[\[\]]", "", original_damage).replace(" ", "")
        expanded_list: list[str] = damage.split(",") * num
        expanded_damage = ",".join(expanded_list)
    else:
        expanded_damage = ",".join([damage] * num)
    return (
        match.string[: match.start()] + expanded_damage + match.string[match.end():]
        if match.end()
        else match.string[: match.start()] + expanded_damage
    )


def apply_to_columns(
        df: DataFrame, func: Callable, columns: list[str] | None = None
) -> DataFrame:
    sliced_df: DataFrame = df if columns is None else df.loc[:, columns]
    func_columns: Index | list[str] = df.columns if columns is None else columns

    sliced_df = sliced_df.applymap(
        lambda x: func(x) if pd.notnull(x) and callable(func) else x
    )
    df.loc[:, func_columns] = sliced_df
    return df


def clean_frame_data(frame_data: DataFrame) -> DataFrame:
    frame_data = initial_string_cleaning(frame_data)

    frame_data = separate_annie_stars(frame_data)

    frame_data = separate_damage_chip_damage(frame_data)

    frame_data = frame_data.applymap(
        lambda x: int(x) if isinstance(x, str) and "-" in x and x.isnumeric() else x
    )

    return frame_data


def initial_string_cleaning(frame_data: DataFrame) -> DataFrame:
    columns_to_remove_chars: list[str] = frame_data.columns.tolist()
    columns_to_remove_chars.remove("alt_names")

    function_column_dict: dict[Callable, list[str]] = {
        lambda x: const.RE_CHARACTERS_TO_REMOVE.sub("", x): columns_to_remove_chars,
        lambda x: x.split("\n"): ["alt_names"],
        separate_damage: ["properties"],
        expand_all_x_n: ["damage", "meter"],
    }
    for func, columns in function_column_dict.items():
        frame_data = apply_to_columns(frame_data, func, columns)

    return frame_data


def separate_damage_chip_damage(frame_data: DataFrame) -> DataFrame:
    frame_data["chip_damage"] = frame_data["damage"].apply(
        lambda d: d[d.find("(") + 1: d.find(")")] if isinstance(d, str) else d
    )
    function_column_dict: dict[Callable, List[str]] = {
        lambda d: d[: d.find("(")] if isinstance(d, str) else d: ["damage"],
        lambda x: [
            int(d.strip()) if d.strip().isnumeric() else d
            for d in (x.split(",") if isinstance(x, str) and x != "" else [])
        ]: ["damage", "chip_damage"],
    }
    for func, columns in function_column_dict.items():
        frame_data = apply_to_columns(frame_data, func, columns)

    return frame_data


def add_new_columns(
        frame_data: DataFrame, new_columns: dict[str, str], offset=1
) -> DataFrame:
    for reference_column, new_column in new_columns.items():
        frame_data_columns: list[str] = frame_data.columns.tolist()

        if reference_column is None:
            old_index: int = len(frame_data_columns)
        else:
            old_index = frame_data_columns.index(reference_column)
        frame_data_columns.insert(old_index + offset, new_column)

        frame_data = frame_data.reindex(columns=frame_data_columns, fill_value=None)

    return frame_data


def separate_annie_stars(frame_data: DataFrame) -> DataFrame:
    star_power_annie_rows: DataFrame = frame_data[
        (
                frame_data["damage"].apply(lambda x: isinstance(x, str) and "[" in x)
                | frame_data["on_block"].apply(lambda x: isinstance(x, str) and "[" in x)
        )
        & (frame_data["character"] == "Annie")
        ]  # type: ignore

    original_annie_rows: DataFrame = star_power_annie_rows.copy()
    row: Series[Any]
    re_stars = const.RE_ANNIE_STARS
    re_any = const.RE_ANY
    star_damage = original_annie_rows["damage"].apply(
        lambda x: re_stars.search(x) or re_any.search(x)  # type: ignore
    )
    star_on_block = original_annie_rows["on_block"].apply(
        lambda x: re_stars.search(x) or re_any.search(x)  # type: ignore
    )

    original_annie_rows.loc[:, "damage"] = original_annie_rows.loc[:, "damage"].where(
        Series(not bool(match) for match in star_damage),
        Series(
            match.group(1) + match.group(4)
            if match and match.groups().__len__() > 3
            else match.group(1)
            if match and match.groups().__len__() > 0
            else match.string
            for match in star_damage
        ),
    )
    original_annie_rows.loc[:, "on_block"] = original_annie_rows.loc[
                                             :, "on_block"
                                             ].where(
        Series((not bool(match)) for match in star_on_block),
        Series(
            match.group(1) if match.groups().__len__() > 0 else match.string
            for match in star_on_block
        ),
    )
    star_power_annie_rows.loc[:, "on_block"] = star_power_annie_rows.loc[
                                               :, "on_block"
                                               ].where(
        Series((not bool(match)) for match in star_on_block),
        Series(
            match.group(3) if match.groups().__len__() > 2 else match.string
            for match in star_on_block
        ),
    )

    star_power_annie_rows.loc[:, "damage"] = star_power_annie_rows.loc[
                                             :, "damage"
                                             ].where(
        Series(not bool(match) for match in star_damage),
        Series(
            "".join(match.groups()) if match.groups() else match.string
            for match in star_damage
        ),
    )
    star_power_annie_rows.loc[:, "move_name"] = star_power_annie_rows.loc[
                                                :, "move_name"
                                                ].apply(lambda name: name + "_STAR_POWER")

    original_annie_rows = original_annie_rows.reset_index(drop=True)
    star_power_annie_rows = star_power_annie_rows.reset_index(drop=True)

    combined_annie: DataFrame = pd.concat(
        [original_annie_rows, star_power_annie_rows]
    ).sort_index()

    log.debug(
        f"star_power_annie_rows:\n {combined_annie.loc[:, ['move_name', 'damage', 'on_block']]}"
    )

    frame_data = frame_data.drop(original_annie_rows.index)

    frame_data = pd.concat([combined_annie, frame_data]).sort_index()

    return frame_data


def format_column_headings(df: DataFrame) -> DataFrame:
    df_lower_cols: list[str] = [col.replace(" ", "_").lower() for col in df.columns]
    df.columns = df_lower_cols

    return df


def capitalise_names(name: str) -> str:
    return (
        " ".join([word.capitalize() for word in name.split(" ")])
        if pd.notnull(name)
        else name
    )


def main():
    log.info("========== Starting skug_stats ==========")
    log.info("Loading csvs into dataframes")
    log.info(f"Currect working directory: {os.getcwd()}")

    with open(fm.CHARACTER_DATA_PATH, "r", encoding="utf8") as characters_file:
        characters_df = format_column_headings(pd.read_csv(characters_file))

    with open(fm.FRAME_DATA_PATH, "r", encoding="utf8") as frame_file:
        frame_data = format_column_headings(pd.read_csv(frame_file).convert_dtypes())

    log.info("Loaded csvs into dataframes")

    characters_df["character"] = characters_df["character"].apply(capitalise_names)
    frame_data["character"] = frame_data["character"].apply(capitalise_names)

    new_columns_dict = {"damage": "chip_damage"}

    frame_data = add_new_columns(frame_data, new_columns_dict)
    frame_data = clean_frame_data(frame_data)

    log.info("Created character and move objects")

    try:
        frame_data.to_csv("output.csv", index=False)
    except PermissionError:
        log.error("Could not export to csv, ensure output.csv is not open")
    else:
        log.info("Exported to csv")

    log.info("========== Finished skug_stats ==========")
    return 0


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)

    pd.set_option("display.max_colwidth", 40)

    with cProfile.Profile() as profiler:
        main()

    profiler.dump_stats("stats.prof")
    stats = pstats.Stats(profiler)

    stats.sort_stats("tottime").print_stats(10)

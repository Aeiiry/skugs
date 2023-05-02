import cProfile
import pstats
import re
from typing import Any, Dict, List

# Import constants in global scope
import constants as const
import file_management as fm
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from skug_logger import log


def attempt_to_int(value: str | int) -> str | int:
    return int(value) if isinstance(value, str) and value.isnumeric() else value


def remove_spaces(string: str) -> str:
    return string.replace(" ", "")


def separate_damage(string: str) -> List[str]:
    string = remove_spaces(string)
    return string.split(",")


def expand_all_x_n(damage: str) -> str:
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
    expanded_damage = ""
    if "[" in original_damage:
        damage = re.sub(r"[\[\]]", "", original_damage).replace(" ", "")
        expanded_list: list[str] = damage.split(",") * num
        expanded_damage: str = ",".join(expanded_list)
    else:
        expanded_damage = ",".join([damage] * num)
    return (
        match.string[: match.start()] + expanded_damage + match.string[match.end() :]
        if match.end()
        else match.string[: match.start()] + expanded_damage
    )


def get_move_properties(move_series: pd.Series) -> Dict[str, Any]:
    return {
        attribute_name: move_series[column_name]
        for column_name, attribute_name in const.FD_COLUMNS_TO_MOVE_ATTR_DICT.items()
    }


def clean_frame_data(frame_data: DataFrame) -> DataFrame:
    # Remove spaces from move names
    frame_data.loc[:, "move_name"] = frame_data.loc[:, "move_name"].apply(
        lambda x: x.replace(" ", "_")
    )

    # Remove unwanted characters, as defined in constants.py
    frame_data = initial_string_cleaning(frame_data)
    # Separate Annie star power moves
    frame_data = separate_annie_stars(frame_data)
    # Reorder columns so chip damage is next to damage
    frame_data = separate_damage_chipdamage(frame_data)

    # Find values with - and turn into ints
    frame_data = frame_data.applymap(
        lambda x: int(x) if isinstance(x, str) and "-" in x and x.isnumeric() else x
    )

    return frame_data


def initial_string_cleaning(frame_data: DataFrame) -> DataFrame:
    columns_to_clean: list[str] = frame_data.columns.tolist()
    columns_to_clean.remove("alt_names")

    frame_data.loc[:, columns_to_clean] = frame_data.loc[:, columns_to_clean].applymap(
        lambda x: const.RE_CHARACTERS_TO_REMOVE.sub("", x) if pd.notnull(x) else x
    )
    # Split alt_names into a list by \n
    frame_data.loc[:, "alt_names"] = frame_data.loc[:, "alt_names"].apply(
        lambda x: x.split("\n") if pd.notnull(x) else x
    )
    # Split properties into a list by ,
    frame_data.loc[:, "properties"] = frame_data.loc[:, "properties"].apply(
        # Also turn - into None
        lambda x: x.split(",")
        if pd.notnull(x)
        else x
    )
    # expand xN (e.g. 2x3 -> 2,2,2)
    frame_data["damage"] = frame_data["damage"].apply(
        lambda d: expand_all_x_n(d) if isinstance(d, str) else d
    )
    return frame_data


def separate_damage_chipdamage(frame_data: DataFrame) -> DataFrame:
    frame_data_columns: list[str] = frame_data.columns.tolist()
    # find index of damage column
    damage_index: int = frame_data_columns.index("damage")
    # insert chip damage column next to damage
    frame_data_columns.insert(damage_index + 1, "chip_damage")
    # Re-create dataframe with new column order, chip damage will be empty so it needs to be explicitly set
    frame_data = frame_data.reindex(columns=frame_data_columns).assign(chip_damage=None)
    # Separate damage into damage and chip damage, chip is in parentheses in the string, e.g. 100(50) or 100,50 (50,25)
    frame_data["chip_damage"] = frame_data["damage"].apply(
        lambda d: d[d.find("(") + 1 : d.find(")")] if isinstance(d, str) else d
    )

    # Remove chip damage substring from damage
    frame_data["damage"] = frame_data["damage"].apply(
        lambda d: d[: d.find("(")] if isinstance(d, str) else d
    )
    # Strip spaces from damage and chip damage and separate into lists by comma
    frame_data.loc[:, ["damage", "chip_damage"]] = frame_data.loc[
        :, ["damage", "chip_damage"]
    ].applymap(
        lambda x: [
            int(d.strip()) if d.strip().isnumeric() else d
            for d in (x.split(",") if isinstance(x, str) and x != "" else [])
        ]
        if x
        else x
    )

    return frame_data


def separate_annie_stars(frame_data: DataFrame) -> DataFrame:
    star_power_annie_rows: DataFrame = frame_data[
        # Find rows with star power
        # damage or on_block contains a value in brackets
        # Annie only
        (frame_data["character"] == "Annie")
        & (
            frame_data["damage"].str.contains(r"\[.*\]")
            | frame_data["on_block"].str.contains(r"\[.*\]")
        )
    ]

    original_annie_rows: DataFrame = star_power_annie_rows.copy()
    row: Series[Any]
    star_damage: list[re.Match[str]] = []
    star_on_block: list[re.Match[str]] = []

    for _, row in original_annie_rows.iterrows():
        star_damage_search: re.Match[str] | None = const.RE_ANNIE_STARS.search(
            row["damage"]
        ) or const.RE_ANY.search(row["damage"])

        star_damage.append(star_damage_search)  # type: ignore

        star_on_block_search: re.Match[str] | None = const.RE_ANNIE_STARS.search(
            row["on_block"]
        ) or const.RE_ANY.search(row["on_block"])
        star_on_block.append(star_on_block_search)  # type: ignore

    original_annie_rows.loc[:, "damage"] = original_annie_rows.loc[:, "damage"].where(
        # List of bools from list of re.match | none
        Series(not bool(match) for match in star_damage),
        # Group 1 and 4 from the regex search
        Series(
            match.group(1) + match.group(4) if match and match.groups().__len__() > 3
            # check null again
            else match.group(1)
            if match and match.groups().__len__() > 0
            else match.string
            for match in star_damage
        ),
    )
    original_annie_rows.loc[:, "on_block"] = original_annie_rows.loc[
        :, "on_block"
    ].where(
        # Just group 1 this time
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
        # List of bools from list of re.match | none
        Series(not bool(match) for match in star_damage),
        # Group 1 and 4 from the regex search
        Series(
            "".join(match.groups()) if match.groups() else match.string
            for match in star_damage
        ),
    )
    star_power_annie_rows.loc[:, "move_name"] = star_power_annie_rows.loc[
        :, "move_name"
    ].apply(lambda name: name + "_STAR_POWER")

    # Re-index the original_annie_rows and star_power_annie_rows
    original_annie_rows = original_annie_rows.reset_index(drop=True)
    star_power_annie_rows = star_power_annie_rows.reset_index(drop=True)

    # Interleave the two dataframes
    combined_annie: DataFrame = pd.concat(
        [original_annie_rows, star_power_annie_rows]
    ).sort_index()
    # Remove the original rows from frame_data
    log.debug(
        f"star_power_annie_rows:\n {combined_annie.loc[:,['move_name', 'damage', 'on_block']]}"
    )
    frame_data = frame_data.drop(original_annie_rows.index)
    # Add the combined_annie to the frame_data
    frame_data = pd.concat([combined_annie, frame_data]).sort_index()

    return frame_data


def format_column_headings(df: DataFrame) -> DataFrame:
    df_lower_cols: list[str] = [col.replace(" ", "_").lower() for col in df.columns]
    df.columns = df_lower_cols

    return df


# ==================== #
def main() -> None:
    log.info("========== Starting skug_stats ==========")
    # Open csvs and load into dataframes
    characters_df: DataFrame = format_column_headings(
        pd.read_csv(fm.CHARACTER_DATA_PATH)
    )
    frame_data: DataFrame = format_column_headings(pd.read_csv(fm.FRAME_DATA_PATH))
    # move_aliases = format_column_headings(pd.read_csv(fm.MOVE_NAME_ALIASES_PATH))
    # Change character names to be Upper case first letter lower case rest

    characters_df["character"] = characters_df["character"].apply(capitalise_names)
    frame_data["character"] = frame_data["character"].apply(capitalise_names)

    frame_data = clean_frame_data(frame_data)

    # moves = extract_moves(frame_data, characters_df["character"].to_list())
    log.info("Created character and move objects")

    # export to csv
    frame_data.to_csv("output.csv", index=False)


def capitalise_names(name: str) -> str:
    return (
        " ".join([word.capitalize() for word in name.split(" ")])
        if pd.notnull(name)
        else name
    )


if __name__ == "__main__":
    # Don't limit columns
    pd.set_option("display.max_columns", None)
    # Column max width
    pd.set_option("display.max_colwidth", 40)

    with cProfile.Profile() as profiler:
        main()

    profiler.dump_stats("stats.prof")
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumtime")

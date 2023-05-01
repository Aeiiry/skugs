"""Main, for now, file for all the skug stuff. Contains the Move and Character classes, as well as functions for extracting data from the move dataframes.
"""

import re
import cProfile
import pstats
from dataclasses import dataclass, field
from typing import List, Dict, Self, Tuple, Any, Optional, Literal, Union
import pandas as pd
from pandas.core.base import PandasObject
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from skug_logger import log

# Import constants in global scope
import constants as const
import file_management as fm


@dataclass
class Hit:
    damage: Optional[int | str] = None
    chip: Optional[int | str] = None


@dataclass
class Move:
    character: str
    name: str
    alt_names: Optional[List[str]] = None

    guard: Optional[str] = None
    properties: Optional[List[str]] = None

    on_hit: Optional[Union[int, str]] = None
    on_block: Optional[Union[int, str]] = None

    startup: Optional[Union[int, str]] = None
    active: Optional[Union[int, str]] = None
    recovery: Optional[Union[int, str]] = None

    hitstun: Optional[Union[int, str]] = None
    blockstun: Optional[Union[int, str]] = None
    hitstop: Optional[Union[int, str]] = None
    meter_gain_loss: Optional[Union[int, str]] = None

    notes: Optional[str] = None

    hits_str: Optional[str] = None

    hits: List[Hit] = field(default_factory=list)
    hits_alt: Dict[str, List[Hit]] = field(default_factory=dict)

    category: Optional[str] = None

    def damage_per_hit(self) -> list[float]:
        return get_damage_per_hit(self.hits) if self.hits.__len__() > 1 else []

    def simple_summed_dmg(self) -> int:
        return simple_damage_calc(self.hits) if self.hits else 0

    def hits_as_list(
        self,
        type: Literal["damage", "chip"] = "damage",
        alt_hits_key: str | None = None,
    ) -> list[str]:
        if alt_hits_key:
            return [getattr(hit, type) for hit in self.hits_alt[alt_hits_key]]
        return [getattr(hit, type) for hit in self.hits]

    def get_non_standard_fields(self) -> Self:
        """Mostly for debugging, so that it is easier to figure out what data still needs to be interpreted in specific ways"""
        fields = self.__dict__
        ideal_types = const.MOVE_PROPERTY_IDEAL_TYPES
        """ Dict of string attrs and their ideal types"""
        intersection = fields.keys() & ideal_types.keys()
        # difference = fields.keys() ^ ideal_types.keys()
        # Assign the values of the intersection of the two sets to a new dict
        intersection = {key: fields[key] for key in intersection}

        self.non_standard_fields = [
            {key: value}
            for key, value in intersection.items()
            if value and not isinstance(value, ideal_types[key])
        ]
        return self

    def __post_init__(self) -> None:
        """
        Initialise the Move object after it has been created, performing any necessary sanitisation and calculations
        """
        # Assign move properties to the object

        self.alt_names = (
            self.alt_names.split("\n")
            if isinstance(self.alt_names, str)
            else self.alt_names
        )
        self.hits, self.hits_alt = get_hits(self)
        self.category = get_category(self)
        self.super_level = get_super_level(self)
        # Format the move object as a printable table


@dataclass
class Character:
    name: str
    shortened_names: List[str] = field(default_factory=list)
    color: str | None = None
    moves: Dict[str, Move] = field(default_factory=dict)
    movement_options: List[str] = field(default_factory=list)
    cancel_chain: List[str] = field(default_factory=list)


def get_super_level(move: Move) -> int:
    if (
        move.category == "super"
        and isinstance(move.meter_gain_loss, str)
        and (level := (move.meter_gain_loss.replace("-", "").replace("%", "")))
        and level.isnumeric()
    ):
        return int(level) // 100
    return 0


def get_damage_per_hit(hits: List[Hit]) -> list[float]:
    """Get the damage per hit for a list of hits, returning a list of floats that represent the damage per hit up to that point."""

    damage_per_hit: list[float] = []
    total_damage: int = 0
    for hit in hits:
        if isinstance(hit.damage, int):
            total_damage += hit.damage
            damage_per_hit.append(round(total_damage / (len(damage_per_hit) + 1), 2))
    return damage_per_hit


def get_category(move: Move) -> str:
    """Get the category of a move, returning a string that represents the category. Most categories are defined in constants.py"""

    normal_move_regex: re.Pattern[str] = re.compile(
        r"([\djc]\.?)[lmh][pk]", flags=re.IGNORECASE
    )
    if normal_move_regex.search(move.name):
        return "normal"

    return next(
        (
            key
            for key, value in const.MOVE_CATEGORIES.items()
            if re.search(
                rf"(^{value}([\s_]|$)|([\s_]|^){value}([\s_]|$))|([\s_]{value}[\s_])",
                move.name,
                flags=re.IGNORECASE,
            )
        ),
        "super"
        if move.meter_gain_loss
        and (isinstance(move.meter_gain_loss, str) and move.meter_gain_loss[0] == "-")
        else "other",
    )


def get_hits(move: Move) -> tuple[list[Hit], dict[str, List[Hit]]]:
    """Get the hits for a move, returning a list of hits and a dictionary of alternative hits labelled by name."""
    # Remove any newlines from the damage or move name
    log.debug(f"===== Getting hits for {move.character} {move.name} =====")

    move = sanitise_move(move)

    if isinstance(move.hits_str, str) and move.hits_str not in ["-", "nan"]:
        (
            _,
            hits,
            alt_hits,
        ) = extract_damage(move.hits_str)
    else:
        hits = []
        alt_hits = {}

    return hits, alt_hits


def extract_damage(hits_str: str) -> tuple[str, list[Hit], dict[str, List[Hit]]]:
    """Extract the damage from a move, returning the damage string, a list of hits and a dictionary of alternative hits labelled by name."""
    damage_str = hits_str.lower()
    alt_damage: str = ""
    expanded = expand_all_x_n(damage_str)
    damage_str = expanded or damage_str
    alt_hits_list = []
    hits_list = []
    alt_hits_dict: dict[str, List[Hit]] = {}

    split_damage = damage_str.split("or")
    if len(split_damage) > 1:
        alt_damage = split_damage[1].replace(const.HATRED_INSTALL, "").strip()
        damage_str = split_damage[0].strip()

    damage_str, hits_list = extract_damage_chip(damage_str)

    if alt_damage:
        damage_str, alt_hits_list = extract_damage_chip(alt_damage)

    if alt_hits_list:
        alt_hits_dict["alt"] = alt_hits_list
    log.debug(f"Original damage string: '{hits_str}'")

    return damage_str, hits_list, alt_hits_dict


def extract_damage_chip(
    damage_str: str,
) -> tuple[str, list[Hit]]:
    """Extract the damage and chip damage from a damage string, returning the damage string and a list of hits."""
    chip_list: List[str | int] = []
    damage_list: List[str | int] = []
    hit_list: List[Hit] = []
    if find_chip := const.RE_IN_PAREN.finditer(damage_str):
        for chip in find_chip:
            if separated_damage := separate_damage(chip.group(1)):
                chip_list.extend(map(attempt_to_int, separated_damage))
            else:
                chip_list.append(chip.group(1))
            # Remove the chip damage from the damage string, using positional information from the regex
            damage_str = (
                damage_str[: chip.start()] + damage_str[chip.end() :]
                if chip.end()
                else damage_str[: chip.start()]
            )
    damage_list = list(map(attempt_to_int, separate_damage(damage_str)))
    # list comprehension to make list of Hit objects from damage and chip lists
    # account for cases where there is no chip damage
    hit_list = (
        [Hit(damage=damage, chip=chip) for damage, chip in zip(damage_list, chip_list)]
        if chip_list
        else [Hit(damage=damage, chip=0) for damage in damage_list]
    )

    return damage_str, hit_list


def expand_all_x_n(damage: str) -> str | None:
    """Expand all instances of xN in a damage string, returning the expanded string."""
    while True:
        if x_n_match := const.RE_X_N.search(damage):
            damage = expand_x_n(x_n_match)
        elif x_n_brackets_matches := const.RE_BRACKETS_X_N.search(damage):
            damage = expand_x_n(x_n_brackets_matches)
        else:
            break
    return damage


def attempt_to_int(value: str | int) -> str | int:
    """Attempt to convert a string to an int, returning the int if successful or the original value if not."""
    return int(value) if isinstance(value, str) and value.isnumeric() else value


def remove_spaces(string: str) -> str:
    """Remove all spaces from a string."""
    return string.replace(" ", "")


def separate_damage(string: str) -> List[str]:
    """Separate a string of damage into a list of damage values.
    e.g. "1, 2, 3" -> ["1", "2", "3"]"""
    string = remove_spaces(string)
    return string.split(",")


def sanitise_move(move: Move) -> Move:
    """Sanitise a move's name and raw hits, returning the sanitised move."""
    move.name = move.name.replace(" ", "_")
    move.name = move.name.replace("\n", "")
    move.hits_str = str(move.hits_str).replace("\n", "")
    # Replace '*' with 'x' in damage
    move.hits_str = move.hits_str.replace("*", "x")
    # Replace → with -> in damage
    move.hits_str = move.hits_str.replace("→", "->")

    return move


def expand_x_n(match: re.Match[str]) -> str:
    """Expand an xN string, returning the expanded string."""
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


def extract_moves(
    frame_data: DataFrame,
    characters: Union[str, List[str], None] = None,
):
    moves: list[Move] = []
    if characters:
        if isinstance(characters, str):
            characters = [characters]

        for character in characters:
            character_moves: DataFrame = frame_data[
                frame_data["character"] == character
            ]

            for _, move_series in character_moves.iterrows():
                # Replace  '-' with None
                move_series: Series[Any] = move_series.replace("-", None)

                move_properties: Dict[str, Any] = get_move_properties(move_series)

                move_obj = Move(**move_properties)
                # Remove spaces from move names
                move_obj.name = move_obj.name.replace(" ", "_")
                altered_name: str = (
                    move_series["character"] + "_" + move_series["move_name"]
                )
                # Remove spaces from move names
                altered_name = altered_name.replace(" ", "_")
                moves.append(move_obj)
    return moves


def get_move_properties(move_series: pd.Series) -> Dict[str, Any]:
    return {
        attribute_name: move_series[column_name]
        for column_name, attribute_name in const.FD_COLUMNS_TO_MOVE_ATTR_DICT.items()
    }


def sort_moves(moves: List[Move], key: str, reverse: bool = False) -> List[Move]:
    """Sort the moves by the given key, returning the sorted list of moves. Keys are the attributes of the Move class."""
    return sorted(moves, key=lambda move: getattr(move, key), reverse=reverse)


def simple_damage_calc(hits: List[Hit], starting_hit: int = 0) -> int:
    """Naiive damage calc for a list of hits, returning the damage."""
    # hits  scales down at a compounding 87.5% per hit after the 3rd hit in a combo
    # min damage for moves with >=1000 base damage is 27.5% of the base damage
    # min damage for any other move is 20% of the base damage

    summed_damage: int | float = 0
    scaling = 1

    # Check if list contains only ints
    for hit_number, hit in enumerate(hits):
        if isinstance(hit.damage, int):
            hit_damage: int = hit.damage
            hit_num: int = hit_number + starting_hit
            if hit_num < 2:
                summed_damage += hit_damage
            else:
                summed_damage += hit_damage * scaling
                summed_damage = round(summed_damage - 0.5)
                scaling *= 0.875
                if scaling < 0.275 and hit_damage >= 1000:
                    scaling = 0.275
                elif scaling < 0.2:
                    scaling = 0.2
    return int(summed_damage)


def clean_frame_data(frame_data: DataFrame) -> DataFrame:
    plus_minus_column_labels: list[str] = ["on_block", "on_hit"]

    frame_data = separate_annie_stars(frame_data)

    for move_data in plus_minus_column_labels:
        # Remove '+' and '±' from on_block and on_hit columns
        frame_data[move_data] = frame_data[move_data].apply(
            lambda x: x.replace("+", "") if pd.notnull(x) else x
        )
        frame_data[move_data] = frame_data[move_data].apply(
            lambda x: x.replace("±", "") if pd.notnull(x) else x
        )

        # Find values with - and turn into ints
        frame_data[move_data] = frame_data[move_data].apply(
            lambda x: int(x) if pd.notnull(x) and "-" in x and x.isnumeric() else x
        )

    frame_data["meter"] = frame_data["meter"].apply(
        lambda x: x.replace("%", "") if pd.notnull(x) else x
    )
    return frame_data


def separate_annie_stars(frame_data) -> DataFrame:
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
        star_damage_search = const.RE_ANNIE_STARS.search(
            row["damage"]
        ) or const.RE_ANY.search(row["damage"])

        star_damage.append(star_damage_search)  # type: ignore

        star_on_block_search = const.RE_ANNIE_STARS.search(
            row["on_block"]
        ) or const.RE_ANY.search(row["on_block"])
        star_on_block.append(star_on_block_search)  # type: ignore

    original_annie_rows.loc[:,"damage"] = original_annie_rows.loc[:,"damage"].where(
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
    original_annie_rows.loc[:,"on_block"] = original_annie_rows.loc[:,"on_block"].where(
        # Just group 1 this time
        Series((not bool(match)) for match in star_on_block),
        Series(
            match.group(1) if match.groups().__len__() > 0 else match.string
            for match in star_on_block
        ),
    )
    star_power_annie_rows.loc[:,"on_block"] = star_power_annie_rows.loc[:,"on_block"].where(
        Series((not bool(match)) for match in star_on_block),
        Series(
            match.group(3) if match.groups().__len__() > 2 else match.string
            for match in star_on_block
        ),
    )

    star_power_annie_rows.loc[:,"damage"] = star_power_annie_rows.loc[:,"damage"].where(
        # List of bools from list of re.match | none
        Series(not bool(match) for match in star_damage),
        # Group 1 and 4 from the regex search
        Series(
            "".join(match.groups()) if match.groups() else match.string
            for match in star_damage
        ),
    )
    star_power_annie_rows.loc[:,"move_name"] = star_power_annie_rows.loc[:,"move_name"].apply(
        lambda name: name + "_STAR_POWER"
    )

    # Add the star power rows to the frame data, and update the original rows to remove the star power values
    frame_data = frame_data.drop(original_annie_rows.index)
    frame_data = pd.concat([frame_data, original_annie_rows])
    frame_data = pd.concat([frame_data, star_power_annie_rows])
    return frame_data


def format_column_headings(df: DataFrame) -> DataFrame:
    """Format the column headings of the dataframe, converting to lower case and replacing spaces with "_", returning the dataframe."""
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(" ", "_")
    return df


# ==================== #
def main() -> None:
    """Main function."""
    log.info("========== Starting skug_stats ==========")
    # Open csvs and load into dataframes
    characters_df = format_column_headings(pd.read_csv(fm.CHARACTER_DATA_PATH))
    frame_data = format_column_headings(pd.read_csv(fm.FRAME_DATA_PATH))
    frame_data.to_numpy()
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
    with cProfile.Profile() as profiler:
        main()

    profiler.dump_stats("skug_stats.prof")
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumtime")

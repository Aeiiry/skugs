# %%
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional, Literal, Union
import pandas as pd
from skug_logger import sklogger

# Import constants in global scope
import constants as const
import file_management as fm


# %%
@dataclass
class Hit:
    damage: int | str | None
    chip: int | str | None


@dataclass
class Move:
    move_properties_dict: dict[str, Any]
    character: str = ""
    name: str = ""
    alt_names: Union[str, List[str]] = ""

    guard: str = ""
    properties: Union[str, List[str]] = ""

    meter_gain_loss: Union[int, str] = ""

    on_hit: Union[int, str] = ""
    on_block: Union[int, str] = ""

    startup: Union[int, str] = ""
    active: Union[int, str] = ""
    recovery: Union[int, str] = ""

    hitstun: Union[int, str] = ""
    blockstun: Union[int, str] = ""
    hitstop: Union[int, str] = ""
    meter_gain_loss: Union[int, str] = ""

    notes: str = ""

    hits_str: str = ""

    hits: List[Hit] = field(default_factory=list)
    hits_alt: dict[str, List[Hit]] = field(default_factory=dict)

    category: str = ""

    def __post_init__(self) -> None:
        # Assign all the properties from the dictionary
        for key, value in self.move_properties_dict.items():
            # Check if attribute exists on the class
            if hasattr(self, key):
                # check type of attribute and assign value if it matches
                if value is None:
                    value = ""

                if isinstance(getattr(self, key), type(value)):
                    setattr(self, key, value)
                else:
                    # If the type doesn't match, log it
                    sklogger.warning(
                        f"Type mismatch for {key}! With value {value} and type {type(value)}"
                    )
        self.alt_names = (
            self.alt_names.split("\n")
            if isinstance(self.alt_names, str)
            else self.alt_names
        )
        self.hits, self.hits_alt = get_hits(self)
        self.category = get_category(self)
        self.simple_summed_dmg: int = simple_damage_calc(self.hits) if self.hits else 0
        self.damage_per_hit: list[float] = get_damage_per_hit(self.hits)

    def __str__(self) -> str:
        return self.name


@dataclass
class Character:
    name: str
    shortened_names: List[str] = field(default_factory=list)
    color: str | None = None
    moves: Dict[str, Move] = field(default_factory=dict)
    movement_options: List[str] = field(default_factory=list)
    cancel_chain: List[str] = field(default_factory=list)


def get_damage_per_hit(hits: List[Hit]) -> list[float]:
    # For each hit, get the damage and divide by the number of hits up to that point
    damage_per_hit: list[float] = []
    total_damage: int = 0
    for hit in hits:
        if isinstance(hit.damage, int):
            total_damage += hit.damage
            damage_per_hit.append(round(total_damage / (len(damage_per_hit) + 1), 2))
    return damage_per_hit


# %%
def get_category(move: Move) -> str:
    category = "other"

    normal_move_regex: re.Pattern[str] = re.compile(
        r"([\djc]\.?)[lmh][pk]", flags=re.IGNORECASE
    )
    if normal_move_regex.search(move.name):
        category = "normal"

    for key, value in const.MOVE_CATEGORIES.items():
        if re.search(f"\\^\\|\\\\s{value}\\\\s\\|\\$", move.name, flags=re.IGNORECASE):
            category: str = key
            break

    if move.meter_gain_loss and (
        isinstance(move.meter_gain_loss, str) and move.meter_gain_loss[0] == "-"
    ):
        category = "super"
    # sklogger.debug(f"Category: {category}")

    return category


def get_hits(move: Move):
    # Remove any newlines from the damage or move name
    sklogger.debug(f"===== Getting hits for {move.character} {move.name} =====")

    move = sanitise_move(move)

    if move.hits_str not in ["-", "nan"]:
        (
            _,
            hits,
            alt_hits,
        ) = extract_damage(move)
    else:
        hits = []
        alt_hits = {}

    return hits, alt_hits


def extract_damage(
    move: Move,
):
    damage_str = move.hits_str.lower()
    alt_damage: str = ""
    expanded = expand_all_x_n(damage_str)
    damage_str = expanded or damage_str
    alt_hits_list = []
    hits_list = []
    alt_hits_dict: dict[str, List[Hit]] = {}
    if move.character == "ANNIE":
        damage_str, stars = parse_annie_stars(damage_str)
        if stars:
            alt_hits_dict["stars"] = stars

    split_damage = damage_str.split("or")
    if len(split_damage) > 1:
        alt_damage = split_damage[1].replace(const.HATRED_INSTALL, "").strip()
        damage_str = split_damage[0].strip()

    damage_str, hits_list = extract_damage_chip(damage_str)

    if alt_damage:
        damage_str, alt_hits_list = extract_damage_chip(alt_damage)

    if alt_hits_list:
        alt_hits_dict["alt"] = alt_hits_list
    sklogger.debug(f"Original damage string: '{move.hits_str}'")

    return damage_str, hits_list, alt_hits_dict


def extract_damage_chip(
    damage_str: str,
) -> tuple[str, list[Hit]]:
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
    hit_list = [
        Hit(damage, chip_damage) for damage, chip_damage in zip(damage_list, chip_list)
    ]

    return damage_str, hit_list


def expand_all_x_n(damage: str) -> str | None:
    while True:
        if x_n_match := const.RE_X_N.search(damage):
            damage = expand_x_n(x_n_match)
        elif x_n_brackets_matches := const.RE_BRACKETS_X_N.search(damage):
            damage = expand_x_n(x_n_brackets_matches)
        else:
            break
    return damage


def attempt_to_int(value: str | int) -> str | int:
    return int(value) if isinstance(value, str) and value.isnumeric() else value


def remove_spaces(string: str) -> str:
    return string.replace(" ", "")


def split_on_comma(string: str) -> List[str]:
    return string.split(",")


def separate_damage(string: str) -> List[str]:
    string = remove_spaces(string)
    return split_on_comma(string)


def parse_annie_stars(damage: str) -> Tuple[str, List[Hit]]:
    initial_search = const.RE_ANNIE_STARS_INITIAL.search(damage)
    refined_match = (
        const.RE_ANNIE_STARS_REFINED.findall(damage) if initial_search else None
    )
    surrounding_commas = None
    if initial_search:
        surrounding_commas = re.compile(
            f"\\s?,?\\s?{re.escape(initial_search.group(0))}"
        )
    new_damage_str = damage
    stars_hits_damage_chip: List[Hit] = []
    stars_hits = []
    stars_chip = []
    if (
        initial_search
        and refined_match
        and (stars_refined_dmg := refined_match[0][0])
        and isinstance(stars_refined_dmg, str)
    ):
        stars_hits = list(map(attempt_to_int, separate_damage(stars_refined_dmg)))

    if (
        initial_search
        and refined_match
        and (stars_refined_chip := refined_match[0][1])
        and isinstance(stars_refined_chip, str)
    ):
        stars_chip = list(map(attempt_to_int, separate_damage(stars_refined_chip)))

    if (
        initial_search
        and surrounding_commas
        and (surrounding_commas_match := surrounding_commas.search(damage))
    ):
        new_damage_str = damage.replace(surrounding_commas_match.group(), "")
        new_damage_str = re.sub(r",\s?%", "", new_damage_str)

        stars_hits_damage_chip.extend(
            Hit(damage=hit, chip=chip) for hit, chip in zip(stars_hits, stars_chip)
        )

    return new_damage_str, stars_hits_damage_chip


def sanitise_move(move: Move) -> Move:
    move.name = move.name.replace("\n", "")
    move.hits_str = str(move.hits_str).replace("\n", "")
    # Replace '*' with 'x' in damage
    move.hits_str = move.hits_str.replace("*", "x")
    # Replace → with -> in damage
    move.hits_str = move.hits_str.replace("→", "->")

    return move


def expand_x_n(match: re.Match[str]) -> str:
    n = int(match.group(3))
    damage: str = match.group(1).strip()
    original_damage: str = match.group(0)
    expanded_damage = ""
    if "[" in original_damage:
        damage = re.sub(r"[\[\]]", "", original_damage).replace(" ", "")
        expanded_list: list[str] = damage.split(",") * n
        expanded_damage: str = ",".join(expanded_list)
    else:
        expanded_damage = ",".join([damage] * n)
    return (
        match.string[: match.start()] + expanded_damage + match.string[match.end() :]
        if match.end()
        else match.string[: match.start()] + expanded_damage
    )


def extract_moves(
    frame_data: pd.DataFrame, characters: dict[str, Character], moves: List[Move]
) -> List[Move]:
    move_str: pd.Series[Any]

    for _, move_str in frame_data.iterrows():
        # Replace  '-' with None
        move_str = move_str.replace("-", None)
        move_pd_to_obj_dict: dict[str, str] = const.FD_COLUMNS_TO_MOVE_ATTR_DICT
        move_obj: Move
        move_properties: dict[str, Any] = {
            move_obj_attribute: move_str[frame_data_column]
            for frame_data_column, move_obj_attribute in move_pd_to_obj_dict.items()
            if move_str[frame_data_column] is not None
        }
        move_obj = Move(move_properties)
        # Remove spaces from move names
        move_obj.name = move_obj.name.replace(" ", "_")
        altered_name: str = move_str["Character"] + "_" + move_str["MoveName"]
        # Remove spaces from move names
        altered_name = altered_name.replace(" ", "_")
        characters[move_str["Character"]].moves[move_obj.name] = move_obj
        moves.append(move_obj)
    return moves


# %%
def sort_moves(moves: List[Move], key: str, reverse: bool = False) -> List[Move]:
    return sorted(moves, key=lambda move: getattr(move, key), reverse=reverse)


def simple_damage_calc(hits: List[Hit], starting_hit: int = 0) -> int:
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


# %%
# Rename old log files

# ==================== #
sklogger.info("========== Starting skug_stats ==========")
# Open csvs and load into dataframes
characters_df = pd.read_csv(fm.CHARACTER_DATA_PATH)
frame_data = pd.read_csv(fm.FRAME_DATA_PATH)
move_aliases = pd.read_csv(fm.MOVE_NAME_ALIASES_PATH)

# Change NaN to None
frame_data = frame_data.where(pd.notnull(frame_data), None)

# Remove spaces from column names
characters_df.columns = characters_df.columns.str.replace(" ", "")
frame_data.columns = frame_data.columns.str.replace(" ", "")
move_aliases.columns = move_aliases.columns.str.replace(" ", "")
sklogger.info("Loaded csvs")
sklogger.info(f"characters_df: {characters_df.columns}")
sklogger.info(f"frame_data: {frame_data.columns}")
sklogger.info(f"move_aliases: {move_aliases.columns}")
characters: dict[str, Character] = {}
moves: list[Move] = []
# Create character objects
row: pd.Series
for _, row in characters_df.iterrows():
    name = row["Character"]
    shortened_names = row["CharacterStart"].split(" ")
    color = row["Color"]
    # Create character object
    new_character = Character(name=name, shortened_names=shortened_names, color=color)
    characters[name] = new_character

# Get moves for each character
moves = extract_moves(frame_data, characters, moves)
sklogger.info("Created character and move objects")

moves_temp: list[Move] = moves.copy()

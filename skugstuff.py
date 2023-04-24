import logging
import logging.handlers
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
import pandas as pd

# Import constants in global scope
from constants import *  # NOSONAR


@dataclass
class Hit:
    damage: int | str | None
    chip: int | str | None


@dataclass
class Move:
    move_properties_dict: dict[str, Any]
    character: str = ""
    name: str = ""
    alt_names: str | List[str] = ""

    guard: str = ""
    properties: str | List[str] = ""

    meter_gain_loss: int | str = ""

    on_hit: int | str = ""
    on_block: int | str = ""

    startup: int | str = ""
    active: int | str = ""
    recovery: int | str = ""

    hitstun: int | str = ""
    blockstun: int | str = ""
    hitstop: int | str = ""

    notes: str = ""

    hits_str: str = ""

    hits: List[Hit] = field(default_factory=list)
    hits_special: dict[str, List[Hit]] = field(default_factory=dict)

    category: str = field(default="other")

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
                    logger.warning(
                        f"Type mismatch for {key}! With value {value} and type {type(value)}"
                    )
        self.repeats = 0
        self.hits, self.hits_special = get_hits(self)
        self.category = get_category(self)
        self.chip_damage: int = self.get_chip_damage()
        self.simple_summed_dmg: int = simple_damage_calc(self.hits) if self.hits else 0

    def __str__(self) -> str:
        return self.name

    def get_chip_damage(self) -> int:
        chip_damage: int = sum(
            hit.damage for hit in self.hits if isinstance(hit.damage, int)
        )
        return chip_damage


@dataclass
class Character:
    name: str
    shortened_names: List[str] = field(default_factory=list)
    color: str | None = None
    moves: Dict[str, Move] = field(default_factory=dict)
    movement_options: List[str] = field(default_factory=list)
    cancel_chain: List[str] = field(default_factory=list)


def get_category(move: Move) -> str:
    category = "other"

    normal_move_regex: re.Pattern[str] = re.compile(
        r"([\djc]\.?)[lmh][pk]", flags=re.IGNORECASE
    )
    if normal_move_regex.search(move.name):
        category = "normal"

    for key, value in MOVE_CATEGORIES.items():
        if move.name.find(value) != -1:
            category: str = key
            break

    if move.meter_gain_loss and (
        isinstance(move.meter_gain_loss, str) and move.meter_gain_loss[0] == "-"
    ):
        category = "super"
    logger.debug(f"Category: {category}")

    return category


def get_hits(move: Move):
    # Remove any newlines from the damage or move name
    logger.info(f"Getting hits for {move.name}")
    move = sanitise_move(move)
    damage: str = move.hits_str

    # Create a list of Hit objects that contains the damage and possible chip damage for each hit
    hits_damage_chip: List[Hit] = []
    special_hits_damage_chip: dict[str, List[Hit]] = {}
    stars_hits_damage_chip: List[Hit] = []

    if damage not in ["-", "nan"]:
        damage, special_hits_damage_chip = extract_damage(
            move, damage, stars_hits_damage_chip, special_hits_damage_chip
        )
    logger.debug(f"Hits: {hits_damage_chip}")
    if special_hits_damage_chip:
        logger.debug(f"Special hits: {special_hits_damage_chip}")
    return hits_damage_chip, special_hits_damage_chip


def extract_damage(
    move: Move,
    damage: str,
    stars_hits_damage_chip: List[Hit],
    special_hits_damage_chip: dict[str, List[Hit]],
):
    logger.debug(f"Getting hits for {move.name}")
    logger.debug(f"Starting damage string: '{damage}'")

    damage = (expanded := expand_all_x_n(damage)) or damage
    if expanded:
        logger.debug(f"Expanded damage string: '{damage}'")

    if move.character == "ANNIE":
        damage, stars_hits_damage_chip = parse_annie_stars(damage)
        if stars_hits_damage_chip:
            logger.debug(f"Damage string after parsing Annie stars: '{damage}'")
            logger.debug(f"Stars hits: {stars_hits_damage_chip}")
    number_of_ors: int = len(re.findall(r"OR", damage))
    if number_of_ors > 0:
        logger.debug(f"Number of ORs: {number_of_ors}")
    while (find_chip := RE_IN_PAREN.search(damage)) is not None:
        if find_chip.group(1):
            chip_damage = find_chip.group(1)
            chip_damage = list(map(attempt_to_int, separate_damage(chip_damage)))
            logger.debug(f"Chip damage: {chip_damage}")
            damage = damage.replace(find_chip.group(0), "")
            logger.debug(f"Damage string after removing chip damage: '{damage}'")

    # Split the damage string into a list of hits
    hits = list(map(attempt_to_int, separate_damage(damage)))
    logger.debug(f"Damage list: {hits}")

    # Add stars_hits_damage_chip to hits_damage_chip if they exist
    if stars_hits_damage_chip:
        special_hits_damage_chip["stars"] = stars_hits_damage_chip

    return damage, special_hits_damage_chip


def expand_all_x_n(damage: str) -> str | None:
    if x_n_matches := RE_X_N.findall(damage):
        for x_n_match in x_n_matches:
            if x_n_match:
                damage_before_xn = x_n_match[0]
                damage = expand_x_n(damage, damage_before_xn, int(x_n_match[2]))
    elif x_n_brackets_matches := RE_BRACKETS_X_N.findall(damage):
        for x_n_brackets_match in x_n_brackets_matches:
            damage = expand_x_n(
                damage, x_n_brackets_match[0], int(x_n_brackets_match[2])
            )
    else:
        return None
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
    initial_search = RE_ANNIE_STARS_INITIAL.search(damage)
    refined_match = RE_ANNIE_STARS_REFINED.findall(damage) if initial_search else None
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
        new_damage_str = new_damage_str.rstrip(",")

        stars_hits_damage_chip.extend(
            Hit(damage=hit, chip=chip) for hit, chip in zip(stars_hits, stars_chip)
        )

    return new_damage_str, stars_hits_damage_chip


def sanitise_move(move: Move) -> Move:
    move.name = move.name.replace("\n", "")
    for hit in move.hits or []:
        hit.damage = str(hit.damage).replace("\n", "")
        # Replace '*' with 'x' in damage
        hit.damage = hit.damage.replace("*", "x")
        # Replace → with -> in damage
        hit.damage = hit.damage.replace("→", "->")

    return move


def expand_x_n(parent_string: str, damage: str, n: int) -> str:
    original_damage: str = damage
    expanded_damage = ""
    if "[" in original_damage:
        damage = re.sub(r"[\[\]]", "", original_damage).replace(" ", "")
        expanded_list: list[str] = damage.split(",") * n
        expanded_damage: str = ",".join(expanded_list)
    else:
        expanded_damage = ",".join([damage] * n)
    expanded_parent_string: str = parent_string
    if original_damage in expanded_parent_string:
        original_damage += f"x{n}"
        expanded_parent_string = expanded_parent_string.replace(
            original_damage, expanded_damage
        )
    return expanded_parent_string


def main() -> None:
    # Rename old log files

    # ==================== #
    logger.info("\n\n========== Starting skug_stats ==========\n\n")
    # Open csvs from the current directory
    characters_df: pd.DataFrame = pd.read_csv("characters.csv")
    frame_data: pd.DataFrame = pd.read_csv("frame_data.csv")
    # Change NaN to None
    frame_data = frame_data.where(pd.notnull(frame_data), None)

    move_aliases: pd.DataFrame = pd.read_csv("move_aliases.csv")
    # Remove spaces from column names
    characters_df.columns = characters_df.columns.str.replace(" ", "")
    frame_data.columns = frame_data.columns.str.replace(" ", "")
    move_aliases.columns = move_aliases.columns.str.replace(" ", "")
    logger.info("Loaded csvs")
    logger.info(f"characters_df: {characters_df.columns}")
    logger.info(f"frame_data: {frame_data.columns}")
    logger.info(f"move_aliases: {move_aliases.columns}")
    characters: dict[str, Character] = {}
    moves: list[Move] = []
    # Create character objects
    row: pd.Series[Any]
    for _, row in characters_df.iterrows():
        name = row["Character"]
        shortened_names = row["CharacterStart"].split(" ")
        color = row["Color"]
        # Create character object
        new_character = Character(
            name=name, shortened_names=shortened_names, color=color
        )
        characters[name] = new_character

    # Get moves for each character
    moves = extract_moves(frame_data, characters, moves)
    logger.info("Created character and move objects")

    moves_temp: list[Move] = moves.copy()

    moves_temp = sort_moves(moves_temp, "simple_summed_dmg", reverse=True)

    # Log top 10 moves by damage for each character
    # Characters is dict of str and Character objects
    character: Character
    for character in characters.values():
        move_num = 0
        logger.info(f"Top 10 moves for {character.name}")
        for move in moves_temp:
            if move.character == character.name:
                logger.info(f"{move.name}: {move.simple_summed_dmg}")
                move_num += 1
            if move_num == 10:
                break
        logger.info("\n")


def extract_moves(
    frame_data: pd.DataFrame, characters: dict[str, Character], moves: List[Move]
) -> List[Move]:
    move_str: pd.Series[Any]

    move_pd_to_obj_dict: dict[str, str] = {
        "MoveName": "name",
        "Character": "character",
        "AltNames": "alt_names",
        "Guard": "guard",
        "Properties": "properties",
        "Startup": "startup",
        "Active": "active",
        "Recovery": "recovery",
        "OnBlock": "on_block",
        "OnHit": "on_hit",
        "Footer": "notes",
        "Damage": "hits_str",
    }

    for _, move_str in frame_data.iterrows():
        # Replace  '-' with None
        move_str = move_str.replace("-", None)

        move_obj: Move
        move_properties: dict[str, Any] = {
            move_obj_attribute: move_str[frame_data_column]
            for frame_data_column, move_obj_attribute in move_pd_to_obj_dict.items()
            if move_str[frame_data_column] is not None
        }
        move_obj = Move(move_properties)
        # Remove spaces from move names
        move_obj.name = move_obj.name.replace(" ", "")
        altered_name: str = move_str["Character"] + "_" + move_str["MoveName"]
        # Remove spaces from move names
        altered_name = altered_name.replace(" ", "")
        characters[move_str["Character"]].moves[move_obj.name] = move_obj
        moves.append(move_obj)
    return moves


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


def init_logger() -> logging.Logger:
    logfilename = "skug_stats"
    if os.path.exists(f"{logfilename}.log"):
        # Rename the old log file, replacing the old one if it exists
        os.remove(f"{logfilename}_old.log") if os.path.exists(
            f"{logfilename}_old.log"
        ) else None
        try:
            os.rename(f"{logfilename}.log", f"{logfilename}_old.log")
        except PermissionError:
            print("Could not rename old log file")
            print("Please close any open log files and try again")
    logfilehandler = logging.FileHandler(f"{logfilename}.log", mode="w")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    # Create root logger
    result: logging.Logger = logging.getLogger()
    result.setLevel(logging.DEBUG)
    # Add the console handler to the root logger
    result.addHandler(console)
    # Add the file handler to the root logger
    result.addHandler(logfilehandler)
    logfileformat = "%(relativeCreated)d - %(levelname)s - %(message)s"
    result.handlers[1].setFormatter(logging.Formatter(logfileformat))
    return result


logger: logging.Logger = (
    logging.getLogger() if logging.getLogger().handlers else init_logger()
)

if __name__ == "__main__":
    main()

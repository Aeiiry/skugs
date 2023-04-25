import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
import pandas as pd
from skug_logger import sklogger

# Import constants in global scope
from constants import *  # NOSONAR
import file_management as fm


@dataclass
class Hit:
    damage: int | str | None
    chip: int | str | None
    alt_damage: int | str | None = None
    alt_chip: int | str | None = None


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
                    sklogger.warning(
                        f"Type mismatch for {key}! With value {value} and type {type(value)}"
                    )
        self.repeats = 0
        self.hits, self.hits_special, self.hits_alt = get_hits(self)
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
        if re.search(f"\\^\\|\\\\s{value}\\\\s\\|\\$", move.name, flags=re.IGNORECASE):
            category: str = key
            break

    if move.meter_gain_loss and (
        isinstance(move.meter_gain_loss, str) and move.meter_gain_loss[0] == "-"
    ):
        category = "super"
    sklogger.debug(f"Category: {category}")

    return category


def get_hits(move: Move):
    # Remove any newlines from the damage or move name
    sklogger.debug(f"\n\n===== Getting hits for {move.character} {move.name} =====\n")

    move = sanitise_move(move)
    damage: str = move.hits_str

    # Create a list of Hit objects that contains the damage and possible chip damage for each hit
    hits_damage_chip: List[Hit] = []
    special_hits_damage_chip: dict[str, List[Hit]] = {}
    stars_hits_damage_chip: List[Hit] = []

    if damage not in ["-", "nan"]:
        (
            damage,
            special_hits_damage_chip,
            hits_damage_chip,
            alt_hits_damage_chip,
        ) = extract_damage(
            move, damage, stars_hits_damage_chip, special_hits_damage_chip
        )
    sklogger.debug(f"Hits: {hits_damage_chip}")
    return hits_damage_chip, special_hits_damage_chip, alt_hits_damage_chip


def extract_damage(
    move: Move,
    damage: str,
    stars_hits_damage_chip: List[Hit],
    special_hits_damage_chip: dict[str, List[Hit]],
):
    damage = damage.lower()
    sklogger.debug(f"Starting damage string: '{damage}'")
    alt_damage: str = ""
    expanded = expand_all_x_n(damage)
    damage = expanded or damage
    if expanded:
        sklogger.debug(f"Expanded damage string: '{damage}'")

    if move.character == "ANNIE":
        damage, stars_hits_damage_chip = parse_annie_stars(damage)
        if stars_hits_damage_chip:
            sklogger.debug(f"Damage string after parsing Annie stars: '{damage}'")
            sklogger.debug(f"Stars hits: {stars_hits_damage_chip}")

    split_damage = damage.split("or")
    if len(split_damage) > 1:
        alt_damage = split_damage[1].replace("(during hi)", "").strip()
        damage = split_damage[0].strip()

    damage_list, chip_list, damage = extract_damage_chip(damage)
    if alt_damage:
        alt_damage_list, alt_chip_list, damage = extract_damage_chip(alt_damage)
        sklogger.debug(f"Alt damage list: {alt_damage_list}")
        sklogger.debug(f"Alt chip list: {alt_chip_list}")
    else:
        alt_damage_list = []
        alt_chip_list = []

    sklogger.debug(f"Damage list: {damage_list}")
    sklogger.debug(f"Chip damage list: {chip_list}")

    if stars_hits_damage_chip:
        special_hits_damage_chip["stars"] = stars_hits_damage_chip

    hits_damage_chip = [
        Hit(damage, chip_damage) for damage, chip_damage in zip(damage_list, chip_list)
    ]
    if alt_damage_list:
        alt_hits_damage_chip = [
            Hit(damage, chip_damage)
            for damage, chip_damage in zip(alt_damage_list, alt_chip_list)
        ]
    else:
        alt_hits_damage_chip = []
    return damage, special_hits_damage_chip, hits_damage_chip, alt_hits_damage_chip


def extract_damage_chip(
    damage_str: str,
) -> tuple[list[str | int], list[str | int], str]:
    chip_list: List[str | int] = []
    damage_list: List[str | int] = []
    if find_chip := RE_IN_PAREN.finditer(damage_str):
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
    return damage_list, chip_list, damage_str


def expand_all_x_n(damage: str) -> str | None:
    while True:
        if x_n_match := RE_X_N.search(damage):
            damage = expand_x_n(x_n_match)
        elif x_n_brackets_matches := RE_BRACKETS_X_N.search(damage):
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
        move_pd_to_obj_dict: dict[str, str] = FD_COLUMNS_TO_MOVE_ATTR_DICT
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


def main() -> None:
    # Rename old log files

    # ==================== #
    sklogger.info("\n\n========== Starting skug_stats ==========\n\n")
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
    sklogger.info("Created character and move objects")

    moves_temp: list[Move] = moves.copy()

    moves_temp = sort_moves(moves_temp, "simple_summed_dmg", reverse=True)

    # Log top 10 moves by damage for each character
    # Characters is dict of str and Character objects
    character: Character
    for character in characters.values():
        move_num = 0
        sklogger.info(f"Top 10 moves for {character.name}")
        for move in moves_temp:
            if move.character == character.name:
                sklogger.info(f"{move.name}: {move.simple_summed_dmg}")
                move_num += 1
            if move_num == 10:
                break
        sklogger.info("\n")


if __name__ == "__main__":
    main()

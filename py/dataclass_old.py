import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Self, Union

from pandas import DataFrame, Series

import constants as const
from skug_logger import log


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

    def __init__(self):
        self.non_standard_fields = None

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
        fields: dict[str, Any] = self.__dict__
        ideal_types: dict[str, Any] = const.MOVE_PROPERTY_IDEAL_TYPES
        intersection = fields.keys() & ideal_types.keys()
        # difference = fields.keys() ^ ideal_types.keys()
        # Assign the values of the intersection of the two sets to a new dict
        intersection = {key: fields[key] for key in intersection}

        self.non_standard_fields: list[dict[str, Any]] = [
            {key: value}
            for key, value in intersection.items()
            if value and not isinstance(value, ideal_types[key])
        ]
        return self

    def __post_init__(self) -> None:
        # Assign move properties to the object

        self.alt_names = (
            self.alt_names.split("\n")
            if isinstance(self.alt_names, str)
            else self.alt_names
        )
        self.hits, self.hits_alt = get_hits(self)
        self.category = get_category(self)
        self.super_level: int = get_super_level(self)
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
    level = 0
    if (
            move.category == "super"
            and isinstance(move.meter_gain_loss, str)
            and (level := (move.meter_gain_loss.replace("-", "").replace("%", "")))
            and level.isnumeric()
    ):
        return int(level) // 100
    return 0


def get_damage_per_hit(hits: List[Hit]) -> list[float]:
    damage_per_hit: list[float] = []
    total_damage: int = 0
    for hit in hits:
        if isinstance(hit.damage, int):
            total_damage += hit.damage
            damage_per_hit.append(round(total_damage / (len(damage_per_hit) + 1), 2))
    return damage_per_hit


def get_category(move: Move) -> str:
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


def extract_damage(hits_str: str, expand_all_x_n=None) -> tuple[str, list[Hit], dict[str, List[Hit]]]:
    damage_str = hits_str.lower()
    alt_damage: str = ""
    expanded = expand_all_x_n(damage_str)
    damage_str = expanded or damage_str
    alt_hits_list = []

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
        separate_damage=None, attempt_to_int=None) -> tuple[str, list[Hit]]:
    chip_list: List[str | int] = []
    if find_chip := const.RE_IN_PAREN.finditer(damage_str):
        for chip in find_chip:
            if separated_damage := separate_damage(chip.group(1)):
                chip_list.extend(map(attempt_to_int, separated_damage))
            else:
                chip_list.append(chip.group(1))
            # Remove the chip damage from the damage string, using positional information from the regex
            damage_str = (
                damage_str[: chip.start()] + damage_str[chip.end():]
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


def sanitise_move(move: Move) -> Move:
    move.name = move.name.replace(" ", "_")
    move.name = move.name.replace("\n", "")
    move.hits_str = str(move.hits_str).replace("\n", "")
    # Replace '*' with 'x' in damage
    move.hits_str = move.hits_str.replace("*", "x")
    # Replace → with -> in damage
    move.hits_str = move.hits_str.replace("→", "->")

    return move


def extract_moves(
        frame_data: DataFrame,
        characters: Union[str, List[str], None] = None,
        get_move_properties=None) -> list[Move]:
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

                move_obj = Move()
                # Remove spaces from move names
                move_obj.name = move_obj.name.replace(" ", "_")
                altered_name: str = (
                        move_series["character"] + "_" + move_series["move_name"]
                )
                # Remove spaces from move names
                altered_name.replace(" ", "_")
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

import functools
import re
from collections.abc import Callable
from typing import Any

import pandas as pd
from loguru import logger as log

import skombo


def re_split(ratio, sep: str | None = None):
    sep = r"," if sep is None else sep
    if isinstance(ratio, str):
        result = re.split(sep, ratio)
    else:
        result = ratio
        log.warning(f"split called with non-string: {result}")
    return result


def extract_blockstop(hitstop: str):
    return re.search(r"\(([^()]*)\s?on block[^()]*\)", hitstop)


def split_meter(meter: str) -> tuple[str | None, str | None]:
    if isinstance(meter, str) and (search := skombo.RE_IN_PAREN.search(meter)):
        on_whiff: str | Any = search.group(1)
    else:
        on_whiff = None
    on_hit: str | None = (
        meter.replace(f"({on_whiff})", "") if isinstance(meter, str) else None
    )
    return on_hit, on_whiff


def filter_dict(
    dict_to_filter: dict[str, Any],
    colfilter: str | list[str],
    filter_values: bool = False,
) -> dict[str, Any]:
    """
    Return a dict excluding the keys in the filter param\n
    Retains order so isn't exceptionally fast\n
    filter_values = True will filter values instead, only supports string values\n
    """
    log.debug(f"Filtering {dict_to_filter} with {colfilter}")
    colfilter = list(colfilter) if isinstance(colfilter, str) else colfilter
    if filter_values:
        return {
            k: v
            for k, v in dict_to_filter.items()
            if not isinstance(v, str) or v not in colfilter
        }
    else:
        return {k: v for k, v in dict_to_filter.items() if k not in colfilter}


def format_column_headings(df: pd.DataFrame) -> pd.DataFrame:
    """Format column headings to lowercase with underscores"""
    df.columns = [col.replace(" ", "_").lower() for col in df.columns]
    return df


@functools.cache
def expand_all_x_n(string: str) -> str:
    if isinstance(string, str):
        while (x_n_match := skombo.RE_X_N.search(string)) or (
            x_n_match := skombo.RE_BRACKETS_X_N.search(string)
        ):
            string = expand_x_n(x_n_match)
        # Additional cleanup for splitting on commas
        string = re.sub(r"\s?,\s?", ",", string)
    return string


@functools.cache
def expand_x_n(match: re.Match[str]) -> str:
    x_n = int(match.group(3))
    number: str = match.group(1).strip()
    number_x_n_original: str = match.group(0)
    if "[" in number_x_n_original:
        number = re.sub(r"[\[\]]", "", number_x_n_original).replace(" ", "")
        expanded_list: list[str] = number.split(",") * x_n
        expanded_numbers: str = ",".join(expanded_list)
    else:
        expanded_numbers = ",".join([number] * x_n)
    return (
        match.string[: match.start()] + expanded_numbers + match.string[match.end() :]
        if match.end()
        else match.string[: match.start()] + expanded_numbers
    ).replace(" ", "")


from timeit import default_timer as timer


def timer_func(func: Callable):  # type: ignore
    def wrapper(*args, **kwargs):
        t1: float = timer()
        result = func(*args, **kwargs)
        t2: float = timer()
        # Display total time in milliseconds
        log.trace(f"{func.__name__}() executed in [{(t2 - t1) * 1000:0.4f}] ms")
        return result

    return wrapper


def for_all_methods(decorator: Callable):  # type: ignore
    def decorate(cls):
        for attr in cls.__dict__:  # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate

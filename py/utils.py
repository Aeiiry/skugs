import os
import pandas
import typing


def get_csv_list(path: str) -> list[str]:
    """Returns a list of all csv files in a given path with their relative path"""
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]


def create_df_copy_columns(df: pandas.DataFrame) -> pandas.DataFrame:
    """Creates a copy of a dataframe with the same columns"""

    copy_df: pandas.DataFrame = pandas.DataFrame(columns=df.columns)

    return copy_df


def get_first_value_from_df(df: pandas.DataFrame, key: str) -> typing.Any:
    """Gets the first value from a dataframe for a given key"""
    value: typing.Any = df[key][0]
    return value


def set_column_value(df: pandas.DataFrame, column: str, value: str) -> None:
    """Set an entire column to a given value

    Args:
        df (pandas.DataFrame): dataframe to set the column value in
        column (str): column to set the value in
        value (str): value to set the column to
    """

    df[column] = value


def split_columns(
    df: pandas.DataFrame, column_name: str, seperator: str
) -> pandas.DataFrame:
    """Split a column into multiple rows based on a given seperator

    Args:
        df (pandas.DataFrame): dataframe to split the column in
        column_name (str): column to split
        seperator (str): seperator to split the column on

    Returns:
        pandas.DataFrame: dataframe with the column split
    """
    splitdf = df.copy()
    # split the values in a column on a given seperator

    splitdf[column_name] = splitdf[column_name].str.split(seperator)
    # explode the column so that each value is on a row
    splitdf = splitdf.explode(column_name)
    return splitdf

# parsers.py
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from enum import Enum


class YchartsDataVar(Enum):
    OPERATING_MARGIN = "operating_margin_ttm"
    REVENUE = "revenues_annual"
    TOTAL_ASSETS = "assets_annual"


def _parse_html_to_df(page_text: str) -> pd.DataFrame | None:
    soup = BeautifulSoup(page_text, "lxml")
    element = soup.select_one("#ycn-historical-data-table-0")

    if element is not None:
        tables = element.find_all("table", {"class": "table"})
        parsed_tables = pd.read_html(str(tables), flavor="lxml")
        df = pd.concat(parsed_tables)
        df.columns = ("date", "value")
        df["date"] = pd.to_datetime(df["date"], format="%B %d, %Y")
        df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
        return df
    else:
        return None


# Function to convert values
def _convert_value(value: str) -> np.float64:
    if value.endswith("B"):
        return np.float64(value[:-1]) * 1_000_000_000
    elif value.endswith("M"):
        return np.float64(value[:-1]) * 1_000_000
    elif value.endswith("K"):
        return np.float64(value[:-1]) * 1_000
    else:
        return np.float64(value)


def parse_html_to_pd(page_text: str, ych_var_type: YchartsDataVar) -> pd.DataFrame | None:
    df = _parse_html_to_df(page_text)
    if df is not None:
        match ych_var_type:
            case YchartsDataVar.OPERATING_MARGIN:
                df["value"] = df["value"].str.strip("%").astype(np.float64)
                # sort the DataFrame by the "date" column
                df = df.sort_values(by="date")
                # group the DataFrame by "date" column, using a dynamic
                # time-based grouping where each group spans 3 months
                df = df.groupby(pd.Grouper(key="date", freq="3M")).mean()
            case YchartsDataVar.REVENUE | YchartsDataVar.TOTAL_ASSETS:
                df["value"] = df["value"].apply(_convert_value)  # type: ignore
                df = df.sort_values(by="date")
            case _ as not_implemented_enum:
                raise NotImplementedError(
                    f"implement parsing for {not_implemented_enum.name} first!"
                )
        return df
    else:
        return None


def parse_html_to_pl(page_text: str):
    # df = parse_html_to_df(page_text)
    # df["date"] = pd.to_datetime(df["date"], format="%B %d, %Y")
    # df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    # pldf = pl.from_pandas(df)
    # pldf = pldf.with_columns(
    #     pldf["value"].str.replace("%", "").cast(pl.Float64).alias("value")
    # )
    # return pldf
    pass

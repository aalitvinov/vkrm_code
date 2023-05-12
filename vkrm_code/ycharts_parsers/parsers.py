# parsers.py
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import polars as pl
from enum import Enum


class YchartsDataVar(Enum):
    OPERATING_MARGIN = "operating_margin_ttm"
    REVENUE = "revenues_annual"
    TOTAL_ASSETS = "assets_annual"


def parse_html_to_df(page_text: str) -> pd.DataFrame:
    soup = BeautifulSoup(page_text, "lxml")
    data_tables = soup.find("div", {"ng-bind-html": "$ctrl.dataTableHtml"})

    element = soup.select_one("#ycn-historical-data-table-0")

    tables = element.find_all("table", {"class": "table"})  # type: ignore
    parsed_tables = pd.read_html(str(tables), flavor="lxml")
    df = pd.concat(parsed_tables)
    df.columns = ("date", "value")
    return df


def parse_html_to_pd(page_text: str) -> pd.DataFrame:
    df = parse_html_to_df(page_text)
    df["date"] = pd.to_datetime(df["date"], format="%B %d, %Y")
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["value"] = df["value"].str.strip("%").astype(np.float64)
    return df


def parse_html_to_pl(page_text: str) -> pl.DataFrame:
    df = parse_html_to_df(page_text)
    df["date"] = pd.to_datetime(df["date"], format="%B %d, %Y")
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    pldf = pl.from_pandas(df)
    pldf = pldf.with_columns(
        pldf["value"].str.replace("%", "").cast(pl.Float64).alias("value")
    )
    return pldf

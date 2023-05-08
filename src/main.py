from json import load
from time import sleep
from bs4 import BeautifulSoup
from tqdm import tqdm
import numpy as np
import pandas as pd
import polars as pl
import os
from dotenv import load_dotenv
from playwright.sync_api import Playwright, sync_playwright

from parsers import parse_html_to_pl

TEST_TICKER = "NOK"
HEADLESS = True

load_dotenv()
YCHART_EMAIL = str(os.environ.get("YCHART_EMAIL"))
YCHART_PASSWORD = str(os.environ.get("YCHART_PASSWORD"))

with open("./ch_data/ticker_list.json") as fj:
    ticker_list = load(fj)


def run(playwright: Playwright, tickers: list[str]) -> None:
    browser_context = playwright.chromium.launch_persistent_context(
        "./ch_data/", headless=HEADLESS
    )
    page = browser_context.pages[0]
    # Auth sequence -------
    page.goto("https://ycharts.com/login")
    if page.get_by_text(
        "Welcome back!"
    ).is_visible():  # if login page is visible -> run
        sleep(1)
        page.get_by_placeholder("name@company.com").click()
        page.get_by_placeholder("name@company.com").fill(YCHART_EMAIL)
        page.get_by_placeholder("Password").click()
        page.get_by_placeholder("Password").fill("YCHART_PASSWORD")
        page.locator("label").filter(has_text="Remember me on this device").click()
        page.get_by_role("button", name="Sign In").click()
    # ---------------------

    filenames = list()
    for file in os.listdir("opm_parqs"):
        filename, _ = os.path.splitext(file)
        filenames.append(filename)

    dfs_pl: list[pl.DataFrame] = []
    for ticker in (pbar := tqdm(tickers)):
        pbar.set_description(f"Processing {ticker}")

        url = f"https://ycharts.com/companies/{ticker}/operating_margin_ttm"
        page.goto(url)
        try:
            pldf = (
                parse_html_to_pl(page.content())
                .sort(by="date")
                .groupby_dynamic(index_column="date", every="3mo")
                .agg([pl.mean("value")])
            )
            pldf.columns = ["date", ticker]
            pldf.write_parquet(f"./opm_parqs/{ticker}.parquet")
            dfs_pl.append(pldf)
        except:
            continue

    browser_context.close()


with sync_playwright() as playwright:
    filenames = list()
    for file in os.listdir("opm_parqs"):
        filename, _ = os.path.splitext(file)
        filenames.append(filename)
    last_ticker_idx = ticker_list.index(filenames[-1])
    run(playwright, tickers=ticker_list[last_ticker_idx + 1 :])

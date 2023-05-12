from json import load
from time import sleep
import pandas as pd
from tqdm import tqdm
import polars as pl
import os
from dotenv import load_dotenv
from playwright.sync_api import Playwright, sync_playwright, Page

from ycharts_parsers import parse_html_to_pl, parse_html_to_pd, YchartsDataVar


TEST_TICKER = "NOK"
HEADLESS = False
YCH_DATA_TYPE = YchartsDataVar.OPERATING_MARGIN

PLAYWRIGHT_CACHE = "./data/.playwright_cache"
if not os.path.exists(PLAYWRIGHT_CACHE):
    os.makedirs(PLAYWRIGHT_CACHE)

PARQUETS_PATH = f"./data/ycharts/{YCH_DATA_TYPE.value}_parqs"
if not os.path.exists(PARQUETS_PATH):
    os.makedirs(PARQUETS_PATH)


load_dotenv()
YCHART_EMAIL = str(os.environ.get("YCHART_EMAIL"))
YCHART_PASSWORD = str(os.environ.get("YCHART_PASSWORD"))


# with open("./data/yf/new_tickers_2146.json") as fj:
#     ticker_list = load(fj)
ticker_list = ["NVR", "NOK"]


def extract_ycharts_data(
    page: Page, data_type: YchartsDataVar, tickers: list[str], path
):
    for ticker in (pbar := tqdm(tickers)):
        pbar.set_description(f"Processing {ticker}")

        url = f"https://ycharts.com/companies/{ticker}/{data_type.value}"
        page.goto(url)
        try:
            # pldf = (
            #     parse_html_to_pl(page.content())
            #     .sort(by="date")
            #     .groupby_dynamic(index_column="date", every="3mo")
            #     .agg([pl.mean("value")])
            # )
            # pldf.columns = ["date", ticker]
            # pldf.write_parquet(f"{path}/{ticker}.parquet")

            pdf = parse_html_to_pd(page.content())
            # sort the DataFrame by the "date" column
            pdf = pdf.sort_values(by="date")
            # group the DataFrame by "date" column, using a dynamic time-based grouping where each group spans 3 months
            pdf = pdf.groupby(pd.Grouper(key="date", freq="3M")).mean()
            # reset the index and rename the columns
            pdf = pdf.reset_index().rename(columns={"value": ticker})
            # write the DataFrame to a Parquet file
            pdf.to_parquet(f"{path}/{ticker}.parquet", index=False)

        except Exception as exc:
            raise exc


def run(playwright: Playwright, tickers: list[str]) -> None:
    browser_context = playwright.chromium.launch_persistent_context(
        PLAYWRIGHT_CACHE,
        headless=HEADLESS,  # , args=["--disable-features=ImprovedCookieControls"]
    )
    page = browser_context.new_page()
    page.goto("chrome://settings/", wait_until="domcontentloaded")
    page.locator('a[href="/privacy"]').click()
    page.get_by_label("Cookies and other site data").click()
    page.get_by_label("Allow all cookies").click()
    page.close()
    sleep(1)
    page = browser_context.pages[0]
    # Auth sequence -------
    page.goto("https://ycharts.com/login")
    if page.get_by_text(
        "Welcome back!"
    ).is_visible():  # if login page is visible -> run
        page.get_by_placeholder("name@company.com").click()
        page.get_by_placeholder("name@company.com").fill(YCHART_EMAIL)
        page.get_by_placeholder("Password").click()
        page.get_by_placeholder("Password").fill(YCHART_PASSWORD)
        page.locator("label").filter(has_text="Remember me on this device").click()
        page.get_by_role("button", name="Sign In").click()
    # ---------------------

    extract_ycharts_data(page, YCH_DATA_TYPE, tickers, path=PARQUETS_PATH)

    browser_context.close()


with sync_playwright() as playwright:
    filenames: list[str] = list()
    for file in os.listdir(PARQUETS_PATH):
        filename, _ = os.path.splitext(file)
        filenames.append(filename)
    if filenames:
        last_ticker_idx = ticker_list.index(filenames[-1])
        ticker_list = ticker_list[last_ticker_idx + 1 :]
    run(playwright, tickers=ticker_list)

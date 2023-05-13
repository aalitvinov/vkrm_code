from json import load
from time import sleep
from tqdm import tqdm
import os
from dotenv import load_dotenv
from playwright.sync_api import Playwright, sync_playwright, Page
from tenacity import retry, stop_after_attempt, wait_random

from ycharts_parsers import parse_html_to_pd, YchartsDataVar


TEST_TICKER = "NOK"
HEADLESS = True
YCH_DATA_TYPE = YchartsDataVar.TOTAL_ASSETS

PLAYWRIGHT_CACHE = "./data/.playwright_cache"
if not os.path.exists(PLAYWRIGHT_CACHE):
    os.makedirs(PLAYWRIGHT_CACHE)

PARQUETS_PATH = f"./data/ycharts/{YCH_DATA_TYPE.value}_parqs"
if not os.path.exists(PARQUETS_PATH):
    os.makedirs(PARQUETS_PATH)


load_dotenv()
YCHART_EMAIL = str(os.environ.get("YCHART_EMAIL"))
YCHART_PASSWORD = str(os.environ.get("YCHART_PASSWORD"))


with open("./data/yf/new_tickers_2146.json") as fj:
    ticker_list = load(fj)
# ticker_list = ["NVR", "NOK"]


def extract_ycharts_data(page: Page, ych_var_type: YchartsDataVar, tickers: list[str], path: str):
    @retry(stop=stop_after_attempt(3), wait=wait_random(1.5, 3), reraise=True)
    def run_extraction():
        page.goto(url)
        try:
            pdf = parse_html_to_pd(page.content(), ych_var_type)
            if pdf is not None:
                # reset the index and rename the columns
                pdf = pdf.reset_index().rename(columns={"value": ticker})
                # write the DataFrame to a Parquet file
                pdf.to_parquet(f"{path}/{ticker}.parquet", index=False)
            else:
                pass
        except Exception as exc:
            raise exc

    for ticker in (pbar := tqdm(tickers)):
        pbar.set_description(f"Processing {ticker}")

        url = f"https://ycharts.com/companies/{ticker}/{ych_var_type.value}"
        run_extraction()


def run(playwright: Playwright, tickers: list[str]) -> None:
    browser_context = playwright.chromium.launch_persistent_context(
        user_data_dir=PLAYWRIGHT_CACHE,
        headless=HEADLESS,
    )
    page = browser_context.pages[0]
    # Auth sequence -------
    page.goto("https://ycharts.com/login")
    if page.get_by_text("Welcome back!").is_visible():  # if login page is visible -> run
        page = browser_context.new_page()
        page.goto("chrome://settings/", wait_until="domcontentloaded")
        page.locator('a[href="/privacy"]').click()
        page.get_by_label("Cookies and other site data").click()
        page.get_by_label("Allow all cookies").click()
        page.close()
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

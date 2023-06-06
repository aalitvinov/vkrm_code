import pytest
from pathlib import Path
import pandas as pd

from vkrm_code.strategies import PortfolioType, Universe, Portfolio, Strategy


FIXTURE_DIR = Path(__file__).parent.resolve() / "test_files"
TESTY_TICKERS = ["ADBE", "PDCE", "KBAL", "F", "LPL", "SCX", "GOOG"]


def parse_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / path, parse_dates=True, index_col=0)


@pytest.fixture
def returns() -> pd.DataFrame:
    return parse_csv("test_returns.csv")


@pytest.fixture
def volume() -> pd.DataFrame:
    return parse_csv("test_volume.csv")


@pytest.fixture
def ychopmdf() -> pd.DataFrame:
    return parse_csv("test_ychopmdf.csv")


@pytest.fixture
def asset_turnover() -> pd.DataFrame:
    return parse_csv("test_asset_turnover.csv")


@pytest.fixture
def diff_vector() -> pd.DataFrame:
    return parse_csv("test_diff_vector.csv")


@pytest.fixture
def company_info() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / "test_company_info.csv", index_col=0)


@pytest.fixture
def universe(
    diff_vector: pd.DataFrame,
    ychopmdf: pd.DataFrame,
    asset_turnover: pd.DataFrame,
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    company_info: pd.DataFrame,
) -> Universe:
    return Universe(diff_vector, ychopmdf, asset_turnover, returns, volume, company_info)


class TestUniverse:
    def test_verify_candidates(self, universe: Universe):
        res = set()
        for i in range(2011, 2024):
            res.update(
                universe.verify_candidates(TESTY_TICKERS, pd.Timestamp(year=i, month=4, day=1))
            )
        assert res == {"KBAL", "LPL", "PDCE", "SCX"}


class TestPortfolio:
    def test_fit(self):
        pass


class TestStrategy:
    def test_multi_delete(self):
        pass

    def construct_portfolios(self):
        pass

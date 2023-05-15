from dataclasses import dataclass, fields
from enum import Enum, auto
from typing import Any
from tqdm import tqdm
import pandas as pd


class PortfolioType(Enum):
    """
    Enumeration representing the two types of portfolios:
    LOHT -- Low Operating margin, High asset Turnover
    HOLT -- High Operating margin, Low asset Turnover
    """

    LOHT = auto()
    HOLT = auto()


@dataclass
class Universe:
    """Class representing the universe of stocks and their respective data."""

    factors_df: pd.DataFrame
    operating_margin: pd.DataFrame
    asset_turnover: pd.DataFrame
    yoy_returns: pd.DataFrame
    volumes: pd.DataFrame
    sectors_info: pd.DataFrame

    def verify_candidates(self, candidates: list[str], year: pd.Timestamp) -> list[str]:
        """
        Verify the validity of the given candidates by checking for missing data in the universe.
        Return the list of invalid candidates.
        """
        candidates_with_nas = set()
        for field in fields(self):
            if field.name != "sectors_info":
                data_df_for_candidates = getattr(self, field.name).loc[:, candidates]
                count_nas = data_df_for_candidates.isna().sum()
                # print(field.name, count_nas.loc[count_nas > 0].index.to_list())
                candidates_with_nas.update(count_nas.loc[count_nas > 0].index.to_list())
            else:
                continue
        return sorted(candidates_with_nas)


@dataclass
class Portfolio:
    """Class representing a portfolio of stocks and their respective data."""

    create_date: pd.Timestamp
    hold: pd.DateOffset
    ptype: PortfolioType

    def __post_init__(self):
        """Initialize the symbols and info attributes with None."""
        self.symbols: list[str] | None = None
        self.info: pd.DataFrame | None = None
        self.mean_return: float | None = None

    def fit(self, candidates: list[str], universe: Universe):
        """
        Fit the portfolio using the given candidates and the universe data.
        Update the symbols and info attributes with the appropriate data.
        """
        self.symbols = candidates
        self.info = pd.concat(
            [
                universe.factors_df.loc[self.create_date, candidates],  # type: ignore
                universe.yoy_returns.loc[self.create_date, candidates],  # type: ignore
                universe.yoy_returns.loc[self.create_date + self.hold, candidates],  # type: ignore
                universe.operating_margin.loc[self.create_date, candidates],  # type: ignore
                universe.operating_margin.loc[self.create_date + self.hold, candidates],  # type: ignore
                universe.asset_turnover.loc[self.create_date, candidates],  # type: ignore
                universe.asset_turnover.loc[self.create_date + self.hold, candidates],  # type: ignore
                universe.volumes.loc[self.create_date, candidates],  # type: ignore
                universe.volumes.loc[self.create_date + self.hold, candidates],  # type: ignore
            ],
            axis=1,
        )
        self.info.columns = [
            "diff_vector",
            "r_i",
            "r_ii",
            "opm_i",
            "opm_ii",
            "a_turnover_i",
            "a_turnover_ii",
            "volume_i",
            "volume_ii",
        ]
        self.mean_return = self.info.loc[:, "r_ii"].mean()

class Strategy:
    """Class representing a strategy for constructing portfolios using a given universe."""

    def __init__(self):
        self.portfolios: dict[str, tuple[Portfolio, Portfolio]] = dict()

    @classmethod
    def multi_delete(cls, list_: list[Any], args) -> list[Any]:
        """
        Remove the specified elements from the given list and return the updated list.
        """
        list_ = [e for e in list_ if e not in args]
        return list_

    def construct_portfolios(
        self,
        universe: Universe,
        hold: pd.DateOffset,
        plength: int = 6,
    ):
        """
        Construct portfolios for each year in the given `universe` using
        the specified holding period `hold` and portfolio length `plength`.
        """
        for year in tqdm(universe.factors_df.index[:-1]):
            lohts, holts = [], []
            ranking: list[str] = (
                universe.factors_df.loc[year].sort_values(ascending=False).dropna()
            ).index.to_list()
            while (found := False) is not True and (counter := 100) > 0:
                lohts, holts = ranking[:plength], ranking[-plength:]
                lohts_misses = universe.verify_candidates(lohts, year + hold)
                holts_misses = universe.verify_candidates(holts, year + hold)
                # print(f"{lohts_misses=}")
                # print(f"{holts_misses=}")
                ranking = type(self).multi_delete(ranking, lohts_misses)
                ranking = type(self).multi_delete(ranking, holts_misses)
                # print(len(ranking), len(lohts_misses), len(holts_misses))
                if (len(lohts_misses) + len(holts_misses)) == 0:
                    found = True  # noqa: F841
                    break
                counter -= 1
            loht_portfolio = Portfolio(create_date=year, hold=hold, ptype=PortfolioType.LOHT)
            loht_portfolio.fit(candidates=lohts, universe=universe)
            holt_portfolio = Portfolio(create_date=year, hold=hold, ptype=PortfolioType.HOLT)
            holt_portfolio.fit(candidates=holts, universe=universe)
            year_str = year.strftime("%Y-%m-%d")
            self.portfolios[year_str] = (loht_portfolio, holt_portfolio)  # type: ignore
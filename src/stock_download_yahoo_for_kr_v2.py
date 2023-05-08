# %% [markdown]
# # Importing dependencies

# %%
# Import libraries
import requests
import sys
import wget
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import time
import io
import tqdm.notebook as tq
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as plticker
import json

pd.options.mode.chained_assignment = None

# %% [markdown]
# # Выгрузка === obsolete

# %%
opmed_ticker_list = requests.get(
    "https://raw.githubusercontent.com/qununc/financials_repo/main/data/opmed_ticker_list.json"
).json()

# %%
opmed_ticker_list[-5:]

# %%
# Input Start and End Date
start = datetime.datetime(2008, 1, 1)
end = datetime.datetime(2022, 5, 1)

# %%
# Symbols = ['^GSPC']

# %%
aapl_yf = yf.Ticker("AAPL")
aapl_yf.info["industry"]

# %%
aapl_yf.info["sector"]

# %%
snp500_values = (
    yf.download(
        "ACC", interval="3mo", start=start, end=end, progress=False, show_errors=False
    )
    .loc[:, ["Adj Close", "Volume"]]
    .dropna(axis=0)
    .query("index.dt.month in [4]")
)

# %%
snp500_values


# %%
def yf_stock_3m_download(ticker_name):
    return (
        yf.download(
            ticker_name,
            interval="3mo",
            start=start,
            end=end,
            progress=False,
            show_errors=False,
        )
        .loc[:, ["Adj Close", "Volume"]]
        .dropna(axis=0)
        .query("index.dt.month in [4]")
    )


Symbols = opmed_ticker_list

t0 = time.time()

# create empty dataframe
stock_final = pd.DataFrame()

# iterate over each symbol
for i in tq.tqdm(Symbols):
    # print the symbol which is being downloaded
    # print( str(Symbols.index(i)) + str(' : ') + i, sep=',', end=',', flush=True)

    try:
        # download the stock price
        stock = []
        stock = yf_stock_3m_download(i)

        # append the individual stock prices
        if len(stock) == 0:
            None
        else:
            stock["Name"] = i
            stock_final = stock_final.append(stock, sort=False)
    except Exception:
        None

t1 = time.time()

total = t1 - t0

# %%
stock_final.head()

# %%
stock_final.info()

# %%
# stock_final['Volume'] = stock_final['Volume'].astype(np.uint32)
stock_final["Name"] = stock_final["Name"].astype("category")
stock_final = stock_final.dropna().reset_index()
stock_final.to_csv("stock_final_2012-2022.csv.xz", compression="xz", index=False)

# %%
stock_final.info()

# %% [markdown]
# # Предобработка

# %%
# import pandas as pds
# import numpy as np
stock_final = pd.read_csv(
    "https://github.com/qununc/financials_repo/raw/main/KR/stock_final_2012-2022.csv.xz"
)
stock_final["Volume"] = stock_final["Volume"].astype(np.uint64)
stock_final["Name"] = stock_final["Name"].astype("category")
stock_final = stock_final.set_index("Date")
stock_final.info()

# %%
stock_final.Name.unique()

# %%
stock_final.head(4)

# %%
stock_final[stock_final["Volume"] > 3 * 10**6].sort_values(
    "Adj Close", ascending=False
).loc[:, "Name"].unique()
# %%
stock_result = (
    stock_final[stock_final["Volume"] > 3 * 10**6]
    .drop(["Volume"], axis=1)
    .reset_index()
)
stock_result.rename({"Adj Close": "adj_close"}, axis=1, inplace=True)
# stock_result['adj_close'] = stock_result['adj_close'].round(2)
stock_result["Date"] = pd.to_datetime(stock_result["Date"], infer_datetime_format=True)

# %%
stock_result.info()

# %%
stock_result.tail(3)

# %%
stock_result = stock_result.drop_duplicates(ignore_index=True)
stocks_pivoted = stock_result.pivot_table(
    index="Name", columns="Date", values="adj_close", aggfunc="mean"
)
stocks_pivoted = stocks_pivoted.dropna(
    axis=0, thresh=3
)  # drop rows with  less then 3 points of data
stocks_pivoted.head()

# %%
# stocks_pivoted.isna().sum(axis=1).sort_values().loc['AMD']

# %%
stocks_pivoted.isna().sum(axis=1).sort_values()[:-8:-1]

# %%
stock_returned = stocks_pivoted.copy().pct_change(axis=1).drop(["2008-04-01"], axis=1).T
stock_returned = stock_returned[stock_returned != 0].drop(
    ["VHC", "BPT", "PBT", "SJT"], axis=1
)
stock_returned.head(3)

# %% [markdown]
# # Формирование портфелей по OPM

# %%
filtered_opm_df = pd.read_csv(
    "https://github.com/qununc/financials_repo/raw/main/KR/filtered_opm_df.csv.gz"
).drop(["2021"], axis=1)
filtered_opm_df = (
    filtered_opm_df.set_index(["ticker"])
    .iloc[:, ::-1]
    .loc[stocks_pivoted.index]
    .dropna(how="all")
    .drop_duplicates()
)

filtered_opm_df.columns = [str(int(year_k) + 1) for year_k in filtered_opm_df.columns]
filtered_opm_df = filtered_opm_df.drop(
    filtered_opm_df[filtered_opm_df == 100].dropna(how="all").index
)
filtered_opm_df.info()

# %%
filtered_opm_df.index

# %%
filtered_opm_df.head(3)

# %%
stock_returned.T.head(3)

# %%
portf_dict = {key: None for key in filtered_opm_df.columns}
for year, data_series in filtered_opm_df.iteritems():
    joined_df = pd.merge(
        filtered_opm_df[year],
        stock_returned.loc[str(int(year) + 1), :].iloc[0],
        left_index=True,
        right_index=True,
    ).dropna(how="any")

    joined_df.columns = [
        f"Op. m. in {joined_df.columns[0]}",
        f"returns in {str(int(year) + 1)}",
    ]

    joined_df_sorted = joined_df.sort_values(by=joined_df.columns[0])

    result_of_join = joined_df_sorted.drop(joined_df_sorted.index[6:-6])
    result_of_join.index.name = "ticker"
    portf_dict[year] = result_of_join

# %%
portf_dict["2010"]

# %%
portf_dict["2020"].iloc[6:, 1].median(), portf_dict["2020"].iloc[6:, 1].mean()

# %%
pd.concat(
    (portf.reset_index() for year, portf in portf_dict.items()),
    axis=1,
).to_excel("portfolios_2009-2021.xlsx")

# %%
# Input Start and End Date
start = datetime.datetime(2008, 1, 1)
end = datetime.datetime(2022, 5, 1)

snp500_values = (
    yf.download(
        "^GSPC", interval="3mo", start=start, end=end, progress=False, show_errors=False
    )
    .loc[:, ["Adj Close"]]
    .dropna(axis=0)
    .query("index.dt.month in [4]")
)

snp500_values["Adj Close"].pct_change().loc["2009"]

# %%
results_df = pd.DataFrame(
    index=pd.Index(
        [
            "P1 ~ Min Op. Margin Mean Returns",
            "P2 ~ Max Op. Margin Mean Returns",
            "S&P 500 YoY Returns",
        ],
        name=None,
    )
)
for year, portf in portf_dict.items():
    results_df[year] = [
        portf.iloc[:6, 1].mean(),
        portf.iloc[6:, 1].mean(),
        snp500_values["Adj Close"].pct_change().loc[str(int(year) + 1)][0],
    ]
results_df

# %%
(results_df.iloc[0] - results_df.iloc[1]) < 0

# %%
plt.rcParams["font.size"] = "15"

results_df.T.plot(
    kind="bar",
    figsize=(12, 8),
    xlabel=r"$i$ year",
    ylabel="YoY Return",
    width=0.68,
)

(results_df.iloc[0] - results_df.iloc[1]).plot(
    kind="bar",
    secondary_y=False,
    color=np.where((results_df.iloc[0] - results_df.iloc[1]) < 0, "red", "deepskyblue"),
    grid=True,
    label="P1 - P2",
    alpha=0.25,
    width=0.68,
)

plt.legend(loc="lower left")
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.15)
plt.savefig("portf_plot.png", dpi=500)
# plt.savefig('portf_plot.svg')

# %%
portf_dict["2021"]

# %%

# merging.py
# %%
import polars as pl
from glob import glob
from tqdm import tqdm


# %%
df_pl_res = [pl.read_parquet(fp) for fp in tqdm(glob("opm_parqs/*.parquet"))]

# %%
df_merged = df_pl_res[0]
for df in tqdm(df_pl_res[1:]):
    df_merged = df_merged.join(df, on="date", how="outer")
print(df_merged.shape)

# %%
df_merged.write_parquet("./ch_data/merged.parquet")
df_merged.write_csv("./ch_data/merged.csv")

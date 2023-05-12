import polars as pl

df = pl.read_parquet("./data/ycharts/operating_margin_ttm_parqs/NVR.parquet", use_pyarrow=True)
print(df)
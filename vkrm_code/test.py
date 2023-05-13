import polars as pl

df = pl.read_parquet(
    "./data/ycharts/revenues_annual_parqs/ABEV.parquet", use_pyarrow=True
)
print(df)

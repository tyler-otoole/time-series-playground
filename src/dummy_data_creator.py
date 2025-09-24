import pandas as pd
import numpy as np
import holidays

# configuration
series_name = "series_A"
start_date = "2021-01-01"
end_date   = "2023-12-31"

# make the date range
dates = pd.date_range(start=start_date, end=end_date, freq="D")

# holiday calendar
us_holidays = holidays.US()

# build the dataframe
df = pd.DataFrame({
    "series_name": series_name,
    "date": dates,
})
df["month_name"]   = df["date"].dt.month_name()
df["weekday_name"] = df["date"].dt.day_name()
df["holiday"]      = df["date"].isin(us_holidays).astype(int)

# value creation
rng = np.random.default_rng(42)
base = 100

# add month, weekday, and holiday effects
month_effects   = {"December": +20, "July": -12}
weekday_effects = {"Monday": +8, "Friday": -5}
holiday_effect  = 25

df["value"] = (
    base
    + df["month_name"].map(month_effects).fillna(0)
    + df["weekday_name"].map(weekday_effects).fillna(0)
    + df["holiday"] * holiday_effect
    # add random noise
    + rng.normal(0, 6, len(df))
).round().astype(int)

# save to csv
outfile = "dummy_timeseries.csv"
df.to_csv(outfile, index=False)

# print confirmation
print(f"Wrote {len(df):,} rows to {outfile}")
print(df.head(10))
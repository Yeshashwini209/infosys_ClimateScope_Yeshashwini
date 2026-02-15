import pandas as pd
import numpy as np

print("\nStarting Aggregation...\n")

# Load processed dataset
df = pd.read_csv("processed_weather_data.csv")

# Convert date column
df["last_updated"] = pd.to_datetime(df["last_updated"])

# Set index for resampling
df.set_index("last_updated", inplace=True)

# Select numeric columns only
numeric_cols = df.select_dtypes(include=np.number).columns

# Weekly aggregation
weekly_data = df[numeric_cols].resample("W").mean()
weekly_data.to_csv("weekly_weather_data.csv")

# Monthly aggregation
monthly_data = df[numeric_cols].resample("M").mean()
monthly_data.to_csv("monthly_weather_data.csv")

print("\n✅ Weekly & Monthly Data Saved")
print(" - weekly_weather_data.csv")
print(" - monthly_weather_data.csv")

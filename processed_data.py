import pandas as pd

print("\nStarting Data Processing...\n")

# Load cleaned dataset
df = pd.read_csv("cleaned_weather_data.csv")

# Convert temperature if needed
if "temperature_fahrenheit" in df.columns:
    df["temp_c_from_f"] = (df["temperature_fahrenheit"] - 32) * 5/9

# Convert wind speed
if "wind_mph" in df.columns:
    df["wind_kph_from_mph"] = df["wind_mph"] * 1.60934

# Extract date features
df["last_updated"] = pd.to_datetime(df["last_updated"])
df["year"] = df["last_updated"].dt.year
df["month"] = df["last_updated"].dt.month
df["week"] = df["last_updated"].dt.isocalendar().week

# Save processed dataset
df.to_csv("processed_weather_data.csv", index=False)

print("\n✅ Processed Data Saved as processed_weather_data.csv")

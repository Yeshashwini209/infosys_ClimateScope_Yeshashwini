# 
#  Data Cleaning Script - Milestone 1

import pandas as pd
import numpy as np

print("\nStarting Data Cleaning...\n")

# Load dataset
df = pd.read_csv("GlobalWeatherRepository.csv")

print("Dataset Loaded Successfully ")

# Inspect structure
print("\nDataset Info:")
print(df.info())

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert date column
df["last_updated"] = pd.to_datetime(
    df["last_updated"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

# Drop rows where date is invalid
df.dropna(subset=["last_updated"], inplace=True)

# Fill numeric missing values with median
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("cleaned_weather_data.csv", index=False)

print("\n✅ Cleaned Data Saved as cleaned_weather_data.csv")

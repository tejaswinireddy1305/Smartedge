import pandas as pd
import numpy as np
import os

# ============================================================
# 1. Paths
# ============================================================

INPUT_FILE = r"D:\smart edge\dataset\processed\clean_requests.csv"

OUTPUT_DIR = r"D:\smart edge\dataset\processed"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "final_ml_dataset.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Load data
# ============================================================

print("Loading cleaned data...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")


# ============================================================
# 3. Convert timestamp
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df = df.dropna(
    subset=["timestamp", "url"]
)

df["url"] = df["url"].astype(str)

df["response_bytes"] = pd.to_numeric(
    df["response_bytes"],
    errors="coerce"
).fillna(0)


# ============================================================
# 4. Keep successful requests
# ============================================================

df["status_code"] = pd.to_numeric(
    df["status_code"],
    errors="coerce"
)

df = df[
    (df["status_code"] >= 200) &
    (df["status_code"] < 400)
]


# ============================================================
# 5. Sort chronologically
# ============================================================

df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Valid requests: {len(df):,}")


# ============================================================
# 6. Create time features
# ============================================================

df["hour"] = df["timestamp"].dt.hour

df["day_of_week"] = df["timestamp"].dt.dayofweek


# ============================================================
# 7. Create time windows
# ============================================================

# Divide requests into 10-minute windows.
df["time_window"] = df["timestamp"].dt.floor("10min")


# ============================================================
# 8. Calculate request count per URL per window
# ============================================================

window_counts = (
    df.groupby(["time_window", "url"])
      .size()
      .reset_index(name="window_requests")
)


# ============================================================
# 9. Create PAST request frequency
# ============================================================

# Shift by one time window.
# This prevents the current/future window from being used
# to calculate the prediction features.

window_counts = window_counts.sort_values(
    ["url", "time_window"]
)

window_counts["past_frequency"] = (
    window_counts
    .groupby("url")["window_requests"]
    .shift(1)
)

window_counts["past_frequency"] = (
    window_counts["past_frequency"]
    .fillna(0)
)


# ============================================================
# 10. Create previous request time
# ============================================================

df["previous_request"] = (
    df.groupby("url")["timestamp"].shift(1)
)

df["recency_seconds"] = (
    df["timestamp"] -
    df["previous_request"]
).dt.total_seconds()

df["recency_seconds"] = (
    df["recency_seconds"]
    .fillna(999999)
)


# ============================================================
# 11. Merge past frequency into request data
# ============================================================

df = df.merge(
    window_counts[
        [
            "time_window",
            "url",
            "past_frequency"
        ]
    ],
    on=["time_window", "url"],
    how="left"
)


# ============================================================
# 12. Create FUTURE popularity target
# ============================================================

# For each URL, calculate how many times it appears
# in the NEXT 10-minute window.

future_counts = window_counts[
    [
        "time_window",
        "url",
        "window_requests"
    ]
].copy()

future_counts["prediction_window"] = (
    future_counts["time_window"] -
    pd.Timedelta(minutes=10)
)

future_counts = future_counts.rename(
    columns={
        "window_requests": "future_requests"
    }
)


# Merge future requests
df = df.merge(
    future_counts[
        [
            "prediction_window",
            "url",
            "future_requests"
        ]
    ],
    left_on=["time_window", "url"],
    right_on=["prediction_window", "url"],
    how="left"
)

df["future_requests"] = (
    df["future_requests"]
    .fillna(0)
)


# ============================================================
# 13. Create popularity target
# ============================================================

# A content item is considered popular if it receives
# at least one request in the next time window.

df["popularity_label"] = (
    df["future_requests"] > 0
).astype(int)


# ============================================================
# 14. Content size
# ============================================================

df["content_size"] = df["response_bytes"]


# ============================================================
# 15. Select final ML columns
# ============================================================

final_df = df[
    [
        "timestamp",
        "url",
        "hour",
        "day_of_week",
        "past_frequency",
        "recency_seconds",
        "content_size",
        "future_requests",
        "popularity_label"
    ]
]


# ============================================================
# 16. Clean invalid values
# ============================================================

final_df = final_df.replace(
    [np.inf, -np.inf],
    np.nan
)

final_df = final_df.dropna()


# ============================================================
# 17. Save final dataset
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 18. Print summary
# ============================================================

print("\n==========================================")
print("FINAL ML DATASET CREATED")
print("==========================================")

print(f"Rows    : {len(final_df):,}")
print(f"Columns : {len(final_df.columns)}")

print("\nColumns:")

for col in final_df.columns:
    print(f" - {col}")

print("\nPopularity distribution:")

print(
    final_df["popularity_label"]
    .value_counts()
    .rename({
        0: "Not Popular",
        1: "Popular"
    })
)

print("\nSample:")
print(final_df.head(10))

print("\nSaved to:")
print(OUTPUT_FILE)
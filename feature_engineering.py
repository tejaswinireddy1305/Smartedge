import pandas as pd
import numpy as np
import os

# ============================================================
# SMARTEDGE - LEAKAGE-FREE ML DATASET CREATION
# ============================================================

# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

INPUT_FILE = r"D:\smart edge\dataset\processed\clean_requests.csv"

OUTPUT_DIR = r"D:\smart edge\dataset\processed"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "ml_dataset.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. Configuration
# ------------------------------------------------------------

# Future window used to create the prediction target.
# 5 minutes = 300 seconds.
FUTURE_WINDOW_SECONDS = 300

print("=" * 70)
print("SMARTEDGE ML DATASET CREATION")
print("=" * 70)

print(f"Future prediction window: {FUTURE_WINDOW_SECONDS} seconds")


# ------------------------------------------------------------
# 3. Load dataset
# ------------------------------------------------------------

print("\nLoading cleaned dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")


# ------------------------------------------------------------
# 4. Basic cleaning
# ------------------------------------------------------------

print("\nCleaning dataset...")

# Convert timestamp
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

# Remove invalid timestamps
df = df.dropna(
    subset=["timestamp"]
)

# Clean URL
df["url"] = (
    df["url"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Remove empty URLs
df = df[df["url"] != ""]

# Convert response size
df["response_bytes"] = pd.to_numeric(
    df["response_bytes"],
    errors="coerce"
)

df["response_bytes"] = (
    df["response_bytes"]
    .fillna(0)
)

# Remove invalid sizes
df = df[
    df["response_bytes"] >= 0
]

# Convert status code
df["status_code"] = pd.to_numeric(
    df["status_code"],
    errors="coerce"
)

# Keep successful / redirect responses
df = df[
    (df["status_code"] >= 200) &
    (df["status_code"] < 400)
]


# ------------------------------------------------------------
# 5. Sort chronologically
# ------------------------------------------------------------

print("Sorting requests chronologically...")

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ------------------------------------------------------------
# 6. Create time features
# ------------------------------------------------------------

print("Creating time features...")

df["hour"] = (
    df["timestamp"].dt.hour
)

df["day_of_week"] = (
    df["timestamp"].dt.dayofweek
)


# ------------------------------------------------------------
# 7. Create content size
# ------------------------------------------------------------

df["content_size"] = (
    df["response_bytes"]
)


# ------------------------------------------------------------
# 8. Create PAST frequency
# ------------------------------------------------------------

print("Creating past request frequency...")

# IMPORTANT:
# This uses ONLY requests that occurred BEFORE
# the current request.

df["past_frequency"] = (
    df.groupby("url")
      .cumcount()
)


# ------------------------------------------------------------
# 9. Create PAST recency
# ------------------------------------------------------------

print("Creating past recency...")

df["previous_request"] = (
    df.groupby("url")["timestamp"]
      .shift(1)
)

df["recency_seconds"] = (
    df["timestamp"] -
    df["previous_request"]
).dt.total_seconds()


# First occurrence of a URL
# has no previous request.
df["recency_seconds"] = (
    df["recency_seconds"]
    .fillna(999999)
)


# ------------------------------------------------------------
# 10. Create FUTURE TARGET
# ------------------------------------------------------------

print(
    "Creating future request target..."
)

# We need to know whether the SAME URL appears
# again within the next 5 minutes.
#
# IMPORTANT:
# This information is used ONLY as the target.
# It is NOT used as an input feature.

df["next_request"] = (
    df.groupby("url")["timestamp"]
      .shift(-1)
)

df["seconds_until_next_request"] = (
    df["next_request"] -
    df["timestamp"]
).dt.total_seconds()


# Target:
#
# 1 = URL requested again within 5 minutes
# 0 = URL NOT requested again within 5 minutes

df["future_request"] = (
    (
        df["seconds_until_next_request"] > 0
    ) &
    (
        df["seconds_until_next_request"]
        <= FUTURE_WINDOW_SECONDS
    )
).astype(int)


# ------------------------------------------------------------
# 11. Select ML dataset
# ------------------------------------------------------------

ml_df = df[
    [
        "timestamp",
        "url",
        "hour",
        "day_of_week",
        "past_frequency",
        "recency_seconds",
        "content_size",
        "future_request"
    ]
].copy()


# ------------------------------------------------------------
# 12. Remove invalid values
# ------------------------------------------------------------

ml_df = ml_df.replace(
    [np.inf, -np.inf],
    np.nan
)

ml_df = ml_df.dropna()


# ------------------------------------------------------------
# 13. Remove extremely large recency values
# ------------------------------------------------------------

# Keep the special first-request value.
ml_df["recency_seconds"] = (
    ml_df["recency_seconds"]
    .clip(lower=0)
)


# ------------------------------------------------------------
# 14. Display dataset statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET STATISTICS")
print("=" * 70)

print(
    f"Final rows: {len(ml_df):,}"
)

print(
    f"Final columns: {len(ml_df.columns)}"
)

print("\nFeatures:")

features = [
    "hour",
    "day_of_week",
    "past_frequency",
    "recency_seconds",
    "content_size"
]

for feature in features:
    print(f" - {feature}")


print("\nTarget:")

print(
    "future_request"
)

print("\nTarget distribution:")

distribution = (
    ml_df["future_request"]
    .value_counts()
    .sort_index()
)

print(
    distribution.rename({
        0: "No Future Request",
        1: "Future Request"
    })
)


# ------------------------------------------------------------
# 15. Target percentage
# ------------------------------------------------------------

positive_count = (
    ml_df["future_request"]
    .sum()
)

total_count = len(
    ml_df
)

positive_percentage = (
    positive_count /
    total_count *
    100
)

print(
    f"\nFuture request rate: "
    f"{positive_percentage:.2f}%"
)


# ------------------------------------------------------------
# 16. Save dataset
# ------------------------------------------------------------

print("\nSaving ML dataset...")

ml_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 17. Sample
# ------------------------------------------------------------

print("\nSample data:")

print(
    ml_df.head(10).to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 70)

print(
    f"Output file:\n{OUTPUT_FILE}"
)

print("=" * 70)
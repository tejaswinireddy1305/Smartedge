import os
from collections import OrderedDict, Counter

import joblib
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\smart edge"

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "final_ml_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "smartedge_random_forest.pkl"
)

RESULT_PATH = os.path.join(
    BASE_DIR,
    "cache",
    "results",
    "cache_size_sensitivity.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

CACHE_SIZES = [25, 50, 100, 200]

SIMULATION_REQUESTS = 100000

PREDICTION_THRESHOLD = 0.5

# Simulated latency assumptions
CACHE_LATENCY_MS = 10.0
ORIGIN_LATENCY_MS = 100.0


# ============================================================
# LOAD DATA
# ============================================================

def prepare_data():

    print("Loading dataset...")

    df = pd.read_csv(DATASET_PATH)

    print(
        f"Dataset rows: {len(df):,}"
    )

    # Timestamp
    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["timestamp"]
        )

    required_columns = [
        "url",
        "hour",
        "day_of_week",
        "past_frequency",
        "recency_seconds",
        "content_size"
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    numeric_columns = [
        "hour",
        "day_of_week",
        "past_frequency",
        "recency_seconds",
        "content_size"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=required_columns
    )

    if "timestamp" in df.columns:

        df = df.sort_values(
            "timestamp"
        )

    df = df.head(
        SIMULATION_REQUESTS
    ).copy()

    print(
        f"Simulation requests: "
        f"{len(df):,}"
    )

    print(
        f"Unique objects: "
        f"{df['url'].nunique():,}"
    )

    return df


# ============================================================
# LRU CACHE
# ============================================================

def run_lru(requests, capacity):

    cache = OrderedDict()

    hits = 0
    misses = 0

    for row in requests.itertuples(index=False):

        url = row.url

        if url in cache:

            hits += 1

            cache.move_to_end(url)

        else:

            misses += 1

            if len(cache) >= capacity:

                cache.popitem(
                    last=False
                )

            cache[url] = row.content_size

    hit_rate = (
        hits / len(requests) * 100
    )

    return hit_rate


# ============================================================
# LFU CACHE
# ============================================================

def run_lfu(requests, capacity):

    cache = {}

    frequencies = Counter()

    hits = 0
    misses = 0

    for row in requests.itertuples(index=False):

        url = row.url

        if url in cache:

            hits += 1

            frequencies[url] += 1

        else:

            misses += 1

            if len(cache) >= capacity:

                victim = min(
                    cache,
                    key=lambda x: frequencies[x]
                )

                del cache[victim]
                del frequencies[victim]

            cache[url] = row.content_size

            frequencies[url] = 1

    hit_rate = (
        hits / len(requests) * 100
    )

    return hit_rate


# ============================================================
# SMARTEDGE
# ML ADMISSION + LRU EVICTION
# ============================================================

def run_smartedge(
    requests,
    model,
    capacity,
    probabilities
):

    cache = OrderedDict()

    hits = 0
    misses = 0

    for row, probability in zip(
        requests.itertuples(index=False),
        probabilities
    ):

        url = row.url

        # ----------------------------------------------------
        # HIT
        # ----------------------------------------------------

        if url in cache:

            hits += 1

            cache.move_to_end(url)

        # ----------------------------------------------------
        # MISS
        # ----------------------------------------------------

        else:

            misses += 1

            # ------------------------------------------------
            # ML ADMISSION
            # ------------------------------------------------

            if probability >= PREDICTION_THRESHOLD:

                # LRU eviction
                if len(cache) >= capacity:

                    cache.popitem(
                        last=False
                    )

                cache[url] = row.content_size

    hit_rate = (
        hits / len(requests) * 100
    )

    return hit_rate


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SMARTEDGE CACHE-SIZE SENSITIVITY EXPERIMENT")
    print("=" * 70)

    print()

    requests = prepare_data()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading Random Forest model...")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # Batch ML prediction ONCE
    # --------------------------------------------------------
    # Same predictions are reused for every cache size.
    # This makes the experiment faster and fairer.
    # --------------------------------------------------------

    print()
    print(
        "Generating ML predictions..."
    )

    features = requests[
        [
            "hour",
            "day_of_week",
            "past_frequency",
            "recency_seconds",
            "content_size"
        ]
    ]

    probabilities = model.predict_proba(
        features
    )[:, 1]

    print(
        "ML predictions completed."
    )

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    results = []

    print()
    print("=" * 70)
    print("CACHE SIZE RESULTS")
    print("=" * 70)

    for capacity in CACHE_SIZES:

        print()
        print(
            f"Testing cache size: "
            f"{capacity}"
        )

        # LRU
        lru_hr = run_lru(
            requests,
            capacity
        )

        # LFU
        lfu_hr = run_lfu(
            requests,
            capacity
        )

        # SmartEdge
        smartedge_hr = run_smartedge(
            requests,
            model,
            capacity,
            probabilities
        )

        # SmartEdge simulated latency
        smartedge_miss_rate = (
            100 - smartedge_hr
        )

        smartedge_latency = (
            (
                smartedge_hr / 100
            ) * CACHE_LATENCY_MS
            +
            (
                smartedge_miss_rate / 100
            ) * ORIGIN_LATENCY_MS
        )

        results.append({
            "cache_size": capacity,
            "smartedge_hit_rate": round(
                smartedge_hr,
                3
            ),
            "lru_hit_rate": round(
                lru_hr,
                3
            ),
            "lfu_hit_rate": round(
                lfu_hr,
                3
            ),
            "smartedge_latency_ms": round(
                smartedge_latency,
                3
            )
        })

        print(
            f"SmartEdge HR : "
            f"{smartedge_hr:.2f}%"
        )

        print(
            f"LRU HR       : "
            f"{lru_hr:.2f}%"
        )

        print(
            f"LFU HR       : "
            f"{lfu_hr:.2f}%"
        )

        print(
            f"SmartEdge Lat: "
            f"{smartedge_latency:.2f} ms"
        )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            RESULT_PATH
        ),
        exist_ok=True
    )

    results_df.to_csv(
        RESULT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # IEEE table format
    # --------------------------------------------------------

    print()
    print()
    print("=" * 70)
    print("TABLE VI - CACHE-SIZE SENSITIVITY")
    print("=" * 70)

    ieee_table = results_df.rename(
        columns={
            "cache_size": "Cache Size",
            "smartedge_hit_rate": "SmartEdge HR (%)",
            "lru_hit_rate": "LRU HR (%)",
            "lfu_hit_rate": "LFU HR (%)",
            "smartedge_latency_ms":
                "SmartEdge Latency (ms)"
        }
    )

    print()

    print(
        ieee_table.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Improvement calculation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SMARTEDGE IMPROVEMENT OVER LRU")
    print("=" * 70)

    for row in results:

        improvement = (
            row["smartedge_hit_rate"]
            -
            row["lru_hit_rate"]
        )

        print(
            f"Cache {row['cache_size']:>3}: "
            f"{improvement:+.2f} percentage points"
        )

    # --------------------------------------------------------
    # File
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "Results saved to:"
    )

    print(
        RESULT_PATH
    )

    print("=" * 70)


if __name__ == "__main__":

    main()
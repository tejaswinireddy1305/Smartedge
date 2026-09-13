import os
import time
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
    "scalability_results.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

LOAD_SIZES = [
    25000,
    50000,
    100000,
    200000
]

CACHE_CAPACITY = 100

PREDICTION_THRESHOLD = 0.5


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data():

    print()
    print("Loading dataset...")

    df = pd.read_csv(
        DATASET_PATH
    )

    print(
        f"Dataset rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["timestamp"]
        )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "url",
        "hour",
        "day_of_week",
        "past_frequency",
        "recency_seconds",
        "content_size"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Chronological order
    # --------------------------------------------------------

    if "timestamp" in df.columns:

        df = df.sort_values(
            "timestamp"
        )

    # Need at least 200,000 requests
    df = df.head(
        max(LOAD_SIZES)
    ).copy()

    print(
        f"Rows used for scalability test: "
        f"{len(df):,}"
    )

    print(
        f"Unique objects: "
        f"{df['url'].nunique():,}"
    )

    return df


# ============================================================
# LRU
# ============================================================

def run_lru(requests):

    cache = OrderedDict()

    start_time = time.perf_counter()

    hits = 0
    misses = 0

    for row in requests.itertuples(index=False):

        url = row.url

        if url in cache:

            hits += 1

            cache.move_to_end(url)

        else:

            misses += 1

            if len(cache) >= CACHE_CAPACITY:

                cache.popitem(
                    last=False
                )

            cache[url] = row.content_size

    execution_time = (
        time.perf_counter()
        -
        start_time
    )

    throughput = (
        len(requests)
        /
        execution_time
    )

    return {
        "hits": hits,
        "misses": misses,
        "execution_time": execution_time,
        "throughput": throughput
    }


# ============================================================
# LFU
# ============================================================

def run_lfu(requests):

    cache = {}

    frequencies = Counter()

    start_time = time.perf_counter()

    hits = 0
    misses = 0

    for row in requests.itertuples(index=False):

        url = row.url

        if url in cache:

            hits += 1

            frequencies[url] += 1

        else:

            misses += 1

            if len(cache) >= CACHE_CAPACITY:

                victim = min(
                    cache,
                    key=lambda x: frequencies[x]
                )

                del cache[victim]

                del frequencies[victim]

            cache[url] = row.content_size

            frequencies[url] = 1

    execution_time = (
        time.perf_counter()
        -
        start_time
    )

    throughput = (
        len(requests)
        /
        execution_time
    )

    return {
        "hits": hits,
        "misses": misses,
        "execution_time": execution_time,
        "throughput": throughput
    }


# ============================================================
# SMARTEDGE
# ML ADMISSION + LRU EVICTION
# ============================================================

def run_smartedge(
    requests,
    probabilities
):

    cache = OrderedDict()

    start_time = time.perf_counter()

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

            # ML admission
            if (
                probability
                >=
                PREDICTION_THRESHOLD
            ):

                if len(cache) >= CACHE_CAPACITY:

                    cache.popitem(
                        last=False
                    )

                cache[url] = row.content_size

    execution_time = (
        time.perf_counter()
        -
        start_time
    )

    throughput = (
        len(requests)
        /
        execution_time
    )

    return {
        "hits": hits,
        "misses": misses,
        "execution_time": execution_time,
        "throughput": throughput
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print("SMARTEDGE SCALABILITY EXPERIMENT")
    print("=" * 75)

    print()
    print(
        f"Cache capacity: "
        f"{CACHE_CAPACITY} objects"
    )

    print(
        f"ML threshold: "
        f"{PREDICTION_THRESHOLD}"
    )

    print(
        f"Load sizes: "
        f"{LOAD_SIZES}"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    requests = prepare_data()

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print("Loading Random Forest model...")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    # ========================================================
    # RUN EACH LOAD
    # ========================================================

    results = []

    for load_size in LOAD_SIZES:

        print()
        print("=" * 75)
        print(
            f"TESTING LOAD: "
            f"{load_size:,} REQUESTS"
        )
        print("=" * 75)

        workload = requests.head(
            load_size
        ).copy()

        # ----------------------------------------------------
        # LRU
        # ----------------------------------------------------

        print()
        print("Running LRU...")

        lru_result = run_lru(
            workload
        )

        lru_throughput = (
            lru_result["throughput"]
        )

        print(
            f"LRU throughput: "
            f"{lru_throughput:,.2f} req/s"
        )

        # ----------------------------------------------------
        # LFU
        # ----------------------------------------------------

        print()
        print("Running LFU...")

        lfu_result = run_lfu(
            workload
        )

        lfu_throughput = (
            lfu_result["throughput"]
        )

        print(
            f"LFU throughput: "
            f"{lfu_throughput:,.2f} req/s"
        )

        # ----------------------------------------------------
        # SmartEdge
        # ----------------------------------------------------

        print()
        print("Running SmartEdge...")

        features = workload[
            [
                "hour",
                "day_of_week",
                "past_frequency",
                "recency_seconds",
                "content_size"
            ]
        ]

        # Measure prediction + cache processing
        smartedge_start = (
            time.perf_counter()
        )

        probabilities = model.predict_proba(
            features
        )[:, 1]

        smartedge_result = run_smartedge(
            workload,
            probabilities
        )

        # Add ML prediction time
        prediction_time = (
            time.perf_counter()
            -
            smartedge_start
        )

        smartedge_total_time = (
            prediction_time
            +
            smartedge_result[
                "execution_time"
            ]
        )

        smartedge_throughput = (
            load_size
            /
            smartedge_total_time
        )

        print(
            f"SmartEdge throughput: "
            f"{smartedge_throughput:,.2f} req/s"
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append({

            "load_requests":
                load_size,

            "smartedge_throughput_req_per_sec":
                round(
                    smartedge_throughput,
                    3
                ),

            "lru_throughput_req_per_sec":
                round(
                    lru_throughput,
                    3
                ),

            "lfu_throughput_req_per_sec":
                round(
                    lfu_throughput,
                    3
                )
        })

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

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

    # ========================================================
    # DISPLAY TABLE VII
    # ========================================================

    print()
    print()
    print("=" * 75)
    print("TABLE VII - SCALABILITY RESULTS")
    print("=" * 75)

    ieee_table = results_df.rename(
        columns={
            "load_requests":
                "Load / Users",

            "smartedge_throughput_req_per_sec":
                "SmartEdge Thpt. (req/s)",

            "lru_throughput_req_per_sec":
                "LRU Thpt. (req/s)",

            "lfu_throughput_req_per_sec":
                "LFU Thpt. (req/s)"
        }
    )

    print()

    print(
        ieee_table.to_string(
            index=False
        )
    )

    # ========================================================
    # SCALABILITY INTERPRETATION
    # ========================================================

    print()
    print("=" * 75)
    print("SCALABILITY INTERPRETATION")
    print("=" * 75)

    print(
        "Throughput is calculated as:"
    )

    print(
        "Throughput = Number of requests / "
        "Execution time"
    )

    print()
    print(
        "SmartEdge throughput includes "
        "Random Forest prediction overhead."
    )

    print(
        "LRU and LFU throughput represents "
        "cache-processing execution."
    )

    # ========================================================
    # FILE
    # ========================================================

    print()
    print("=" * 75)

    print(
        "Results saved to:"
    )

    print(
        RESULT_PATH
    )

    print("=" * 75)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
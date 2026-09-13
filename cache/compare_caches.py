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
    "final_cache_comparison.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

CACHE_CAPACITY = 100

SIMULATION_REQUESTS = 100000

PREDICTION_THRESHOLD = 0.5


# ============================================================
# SIMULATED NETWORK LATENCY
# ============================================================
# IMPORTANT:
# These are benchmark assumptions.
# They are NOT real measured network latency.
# ============================================================

CACHE_LATENCY_MS = 10.0
ORIGIN_LATENCY_MS = 100.0


# ============================================================
# LRU CACHE
# ============================================================

def run_lru(requests):

    cache = OrderedDict()

    hits = 0
    misses = 0
    evictions = 0

    origin_requests = 0
    bandwidth_bytes = 0

    total_cache_occupancy = 0

    hit_latencies = []
    miss_latencies = []

    start_time = time.perf_counter()

    for row in requests.itertuples(index=False):

        url = row.url
        content_size = row.content_size

        # ----------------------------------------------------
        # HIT
        # ----------------------------------------------------

        if url in cache:

            hits += 1

            cache.move_to_end(url)

            hit_latencies.append(
                CACHE_LATENCY_MS
            )

        # ----------------------------------------------------
        # MISS
        # ----------------------------------------------------

        else:

            misses += 1
            origin_requests += 1

            miss_latencies.append(
                ORIGIN_LATENCY_MS
            )

            # Origin data transferred
            bandwidth_bytes += content_size

            # Evict LRU if cache is full
            if len(cache) >= CACHE_CAPACITY:

                cache.popitem(
                    last=False
                )

                evictions += 1

            # Insert object
            cache[url] = content_size

        # ----------------------------------------------------
        # Track cache utilization
        # ----------------------------------------------------

        total_cache_occupancy += len(cache)

    execution_time = (
        time.perf_counter() - start_time
    )

    average_cache_utilization = (
        total_cache_occupancy
        /
        (len(requests) * CACHE_CAPACITY)
        * 100
    )

    throughput = (
        len(requests)
        /
        execution_time
    )

    return {
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "proactive_insertions": 0,
        "origin_requests": origin_requests,
        "bandwidth_bytes": bandwidth_bytes,
        "average_cache_utilization": average_cache_utilization,
        "hit_latencies": hit_latencies,
        "miss_latencies": miss_latencies,
        "execution_time": execution_time,
        "throughput": throughput
    }


# ============================================================
# LFU CACHE
# ============================================================

def run_lfu(requests):

    cache = {}

    frequencies = Counter()

    hits = 0
    misses = 0
    evictions = 0

    origin_requests = 0
    bandwidth_bytes = 0

    total_cache_occupancy = 0

    hit_latencies = []
    miss_latencies = []

    start_time = time.perf_counter()

    for row in requests.itertuples(index=False):

        url = row.url
        content_size = row.content_size

        # ----------------------------------------------------
        # HIT
        # ----------------------------------------------------

        if url in cache:

            hits += 1

            frequencies[url] += 1

            hit_latencies.append(
                CACHE_LATENCY_MS
            )

        # ----------------------------------------------------
        # MISS
        # ----------------------------------------------------

        else:

            misses += 1
            origin_requests += 1

            miss_latencies.append(
                ORIGIN_LATENCY_MS
            )

            # Origin data transferred
            bandwidth_bytes += content_size

            # ------------------------------------------------
            # Evict least frequently used object
            # ------------------------------------------------

            if len(cache) >= CACHE_CAPACITY:

                victim = min(
                    cache,
                    key=lambda x: frequencies[x]
                )

                del cache[victim]
                del frequencies[victim]

                evictions += 1

            # Insert object
            cache[url] = content_size
            frequencies[url] = 1

        # ----------------------------------------------------
        # Track cache utilization
        # ----------------------------------------------------

        total_cache_occupancy += len(cache)

    execution_time = (
        time.perf_counter() - start_time
    )

    average_cache_utilization = (
        total_cache_occupancy
        /
        (len(requests) * CACHE_CAPACITY)
        * 100
    )

    throughput = (
        len(requests)
        /
        execution_time
    )

    return {
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "proactive_insertions": 0,
        "origin_requests": origin_requests,
        "bandwidth_bytes": bandwidth_bytes,
        "average_cache_utilization": average_cache_utilization,
        "hit_latencies": hit_latencies,
        "miss_latencies": miss_latencies,
        "execution_time": execution_time,
        "throughput": throughput
    }


# ============================================================
# SMARTEDGE
# ML ADMISSION + LRU EVICTION
# ============================================================

def run_smartedge(requests, model):

    cache = OrderedDict()

    hits = 0
    misses = 0
    evictions = 0
    proactive_insertions = 0

    origin_requests = 0
    bandwidth_bytes = 0

    total_cache_occupancy = 0

    hit_latencies = []
    miss_latencies = []

    # --------------------------------------------------------
    # Start timing BEFORE ML prediction.
    # This includes SmartEdge prediction overhead.
    # --------------------------------------------------------

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Batch ML prediction
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Process requests
    # --------------------------------------------------------

    for row, probability in zip(
        requests.itertuples(index=False),
        probabilities
    ):

        url = row.url
        content_size = row.content_size

        # ----------------------------------------------------
        # HIT
        # ----------------------------------------------------

        if url in cache:

            hits += 1

            cache.move_to_end(url)

            hit_latencies.append(
                CACHE_LATENCY_MS
            )

        # ----------------------------------------------------
        # MISS
        # ----------------------------------------------------

        else:

            misses += 1
            origin_requests += 1

            miss_latencies.append(
                ORIGIN_LATENCY_MS
            )

            # ------------------------------------------------
            # Origin data transferred
            # ------------------------------------------------

            bandwidth_bytes += content_size

            # ------------------------------------------------
            # ML ADMISSION DECISION
            # ------------------------------------------------

            if probability >= PREDICTION_THRESHOLD:

                # ------------------------------------------------
                # LRU eviction
                # ------------------------------------------------

                if len(cache) >= CACHE_CAPACITY:

                    cache.popitem(
                        last=False
                    )

                    evictions += 1

                # ------------------------------------------------
                # Insert object
                # ------------------------------------------------

                cache[url] = content_size

                proactive_insertions += 1

        # ----------------------------------------------------
        # Track cache utilization
        # ----------------------------------------------------

        total_cache_occupancy += len(cache)

    execution_time = (
        time.perf_counter() - start_time
    )

    average_cache_utilization = (
        total_cache_occupancy
        /
        (len(requests) * CACHE_CAPACITY)
        * 100
    )

    throughput = (
        len(requests)
        /
        execution_time
    )

    return {
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "proactive_insertions": proactive_insertions,
        "origin_requests": origin_requests,
        "bandwidth_bytes": bandwidth_bytes,
        "average_cache_utilization": average_cache_utilization,
        "hit_latencies": hit_latencies,
        "miss_latencies": miss_latencies,
        "execution_time": execution_time,
        "throughput": throughput
    }


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

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Convert numeric columns
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

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=required_columns
    )

    print(
        f"Rows after cleaning: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Chronological ordering
    # --------------------------------------------------------

    if "timestamp" in df.columns:

        df = df.sort_values(
            "timestamp"
        )

    # --------------------------------------------------------
    # Select workload
    # --------------------------------------------------------

    df = df.head(
        SIMULATION_REQUESTS
    ).copy()

    print(
        f"Simulation rows: "
        f"{len(df):,}"
    )

    print(
        f"Unique objects in simulation: "
        f"{df['url'].nunique():,}"
    )

    return df


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    result,
    algorithm
):

    hits = result["hits"]
    misses = result["misses"]

    total_requests = (
        hits + misses
    )

    # --------------------------------------------------------
    # Hit rate
    # --------------------------------------------------------

    hit_rate = (
        hits / total_requests * 100
        if total_requests > 0
        else 0
    )

    # --------------------------------------------------------
    # Miss rate
    # --------------------------------------------------------

    miss_rate = (
        misses / total_requests * 100
        if total_requests > 0
        else 0
    )

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    hit_latencies = result[
        "hit_latencies"
    ]

    miss_latencies = result[
        "miss_latencies"
    ]

    average_hit_latency = (
        sum(hit_latencies)
        /
        len(hit_latencies)
        if hit_latencies
        else 0
    )

    average_miss_latency = (
        sum(miss_latencies)
        /
        len(miss_latencies)
        if miss_latencies
        else 0
    )

    total_latency = (
        sum(hit_latencies)
        +
        sum(miss_latencies)
    )

    average_latency = (
        total_latency
        /
        total_requests
        if total_requests > 0
        else 0
    )

    # --------------------------------------------------------
    # Bandwidth conversion
    # --------------------------------------------------------

    bandwidth_bytes = result[
        "bandwidth_bytes"
    ]

    bandwidth_kb = (
        bandwidth_bytes / 1024
    )

    bandwidth_mb = (
        bandwidth_bytes / (1024 * 1024)
    )

    # --------------------------------------------------------
    # Return all metrics
    # --------------------------------------------------------

    return {

        "algorithm": algorithm,

        "requests": total_requests,

        "hits": hits,

        "misses": misses,

        "hit_rate": round(
            hit_rate,
            3
        ),

        "miss_rate": round(
            miss_rate,
            3
        ),

        "average_latency_ms": round(
            average_latency,
            3
        ),

        "hit_latency_ms": round(
            average_hit_latency,
            3
        ),

        "miss_latency_ms": round(
            average_miss_latency,
            3
        ),

        "origin_requests": result[
            "origin_requests"
        ],

        "bandwidth_bytes": round(
            bandwidth_bytes,
            2
        ),

        "bandwidth_kb": round(
            bandwidth_kb,
            3
        ),

        "bandwidth_mb": round(
            bandwidth_mb,
            3
        ),

        "cache_utilization_percent": round(
            result[
                "average_cache_utilization"
            ],
            3
        ),

        "evictions": result[
            "evictions"
        ],

        "proactive_insertions": result[
            "proactive_insertions"
        ],

        "execution_time_seconds": round(
            result[
                "execution_time"
            ],
            6
        ),

        "throughput_req_per_sec": round(
            result[
                "throughput"
            ],
            3
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print("SMARTEDGE CACHE COMPARISON BENCHMARK")
    print("=" * 75)

    print()

    print(
        f"Simulation requests : "
        f"{SIMULATION_REQUESTS:,}"
    )

    print(
        f"Cache capacity      : "
        f"{CACHE_CAPACITY} objects"
    )

    print(
        f"Prediction threshold: "
        f"{PREDICTION_THRESHOLD}"
    )

    print(
        f"Cache latency       : "
        f"{CACHE_LATENCY_MS} ms "
        f"(SIMULATED)"
    )

    print(
        f"Origin latency      : "
        f"{ORIGIN_LATENCY_MS} ms "
        f"(SIMULATED)"
    )

    print()

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
    # RUN LRU
    # ========================================================

    print()
    print("-" * 75)
    print("Running LRU...")
    print("-" * 75)

    lru_result = run_lru(
        requests
    )

    lru_metrics = calculate_metrics(
        lru_result,
        "LRU"
    )

    print(
        f"LRU Hit Rate: "
        f"{lru_metrics['hit_rate']:.2f}%"
    )

    # ========================================================
    # RUN LFU
    # ========================================================

    print()
    print("-" * 75)
    print("Running LFU...")
    print("-" * 75)

    lfu_result = run_lfu(
        requests
    )

    lfu_metrics = calculate_metrics(
        lfu_result,
        "LFU"
    )

    print(
        f"LFU Hit Rate: "
        f"{lfu_metrics['hit_rate']:.2f}%"
    )

    # ========================================================
    # RUN SMARTEDGE
    # ========================================================

    print()
    print("-" * 75)
    print("Running SmartEdge...")
    print("-" * 75)

    print(
        "Batch ML prediction in progress..."
    )

    smartedge_result = run_smartedge(
        requests,
        model
    )

    smartedge_metrics = calculate_metrics(
        smartedge_result,
        "SmartEdge"
    )

    print(
        f"SmartEdge Hit Rate: "
        f"{smartedge_metrics['hit_rate']:.2f}%"
    )

    # ========================================================
    # CREATE RESULTS TABLE
    # ========================================================

    results = pd.DataFrame(
        [
            lru_metrics,
            lfu_metrics,
            smartedge_metrics
        ]
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    os.makedirs(
        os.path.dirname(
            RESULT_PATH
        ),
        exist_ok=True
    )

    results.to_csv(
        RESULT_PATH,
        index=False
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print()
    print("=" * 75)
    print("FINAL RESULTS")
    print("=" * 75)

    print()

    print(
        results.to_string(
            index=False
        )
    )

    # ========================================================
    # IEEE TABLE
    # ========================================================

    print()
    print("=" * 75)
    print("IEEE TABLE - CACHE PERFORMANCE")
    print("=" * 75)

    ieee_table = results[
        [
            "algorithm",
            "hit_rate",
            "miss_rate",
            "average_latency_ms",
            "hit_latency_ms",
            "miss_latency_ms",
            "origin_requests"
        ]
    ].copy()

    print()

    print(
        ieee_table.to_string(
            index=False
        )
    )

    # ========================================================
    # IEEE TABLE - RESOURCE / CACHE METRICS
    # ========================================================

    print()
    print("=" * 75)
    print("IEEE TABLE - RESOURCE AND CACHE METRICS")
    print("=" * 75)

    resource_table = results[
        [
            "algorithm",
            "bandwidth_mb",
            "cache_utilization_percent",
            "evictions",
            "throughput_req_per_sec"
        ]
    ].copy()

    print()

    print(
        resource_table.to_string(
            index=False
        )
    )

    # ========================================================
    # LATENCY NOTE
    # ========================================================

    print()
    print("=" * 75)
    print("LATENCY METHODOLOGY")
    print("=" * 75)

    print(
        f"Cache HIT latency  : "
        f"{CACHE_LATENCY_MS} ms"
    )

    print(
        f"Origin MISS latency: "
        f"{ORIGIN_LATENCY_MS} ms"
    )

    print()
    print(
        "IMPORTANT: Latency values are simulated "
        "benchmark assumptions."
    )

    print(
        "Throughput is measured from actual "
        "benchmark execution time."
    )

    print(
        "Bandwidth represents data transferred "
        "from the origin on cache misses."
    )

    # ========================================================
    # FILE LOCATION
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
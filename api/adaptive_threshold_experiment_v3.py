"""
SmartEdge 3.0 - Adaptive Threshold Experiment
------------------------------------------------
Runs LRU, LFU and SmartEdge 3.0 on the same request stream.

Fixes:
1. Baselines operate directly on the URL request stream, so they cannot
   accidentally receive the wrong object type and report 0% hit rate.
2. SmartEdge ML predictions are generated ONCE and reused for every
   capacity/threshold combination.
3. SmartEdge threshold is passed explicitly.
4. No pandas iterrows() inside the 72 SmartEdge experiments.
5. Results are saved to model/adaptive_threshold_results_v3.csv.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import OrderedDict
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


BASE_DIR = r"D:\smart edge"
DATASET_PATH = os.path.join(
    BASE_DIR, "dataset", "processed", "clean_requests.csv"
)
MODEL_PATH = os.path.join(
    BASE_DIR, "model", "smartedge_v3_random_forest.pkl"
)
RESULT_PATH = os.path.join(
    BASE_DIR, "model", "adaptive_threshold_results_v3.csv"
)

DEFAULT_REQUESTS = 10000
DEFAULT_CAPACITIES = [25, 50, 75, 100, 150, 200, 300, 500]
DEFAULT_THRESHOLDS = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9
]

FEATURES = [
    "hour",
    "day_of_week",
    "past_frequency",
    "recency_seconds",
    "content_size",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="SmartEdge 3.0 adaptive threshold experiment"
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=DEFAULT_REQUESTS,
        help="Number of NASA requests to simulate.",
    )

    parser.add_argument(
        "--dataset",
        default=DATASET_PATH,
        help="Path to clean_requests.csv",
    )

    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="Path to SmartEdge 3.0 Random Forest model.",
    )

    parser.add_argument(
        "--result",
        default=RESULT_PATH,
        help="CSV output path.",
    )

    return parser.parse_args()


def load_model(path: str):
    print("\nLoading SmartEdge 3.0 ML model...")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found:\n{path}"
        )

    model = joblib.load(path)

    print(f"Model: {type(model).__name__}")

    if hasattr(model, "feature_names_in_"):
        print(
            "Features:",
            list(model.feature_names_in_)
        )

        missing = [
            f for f in FEATURES
            if f not in model.feature_names_in_
        ]

        if missing:
            raise ValueError(
                f"Model is missing features: {missing}"
            )

    return model


def load_dataset(path: str, n_requests: int):
    print("\nLoading NASA request dataset...")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    # Only load the columns needed by the simulator.
    usecols = [
        "timestamp",
        "url",
        "response_bytes",
    ]

    df = pd.read_csv(
        path,
        usecols=usecols,
        nrows=n_requests,
    )

    if df.empty:
        raise ValueError("Dataset is empty.")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df["response_bytes"] = pd.to_numeric(
        df["response_bytes"],
        errors="coerce"
    ).fillna(0)

    df["url"] = df["url"].fillna("").astype(str)

    df = df.dropna(
        subset=["timestamp"]
    ).reset_index(drop=True)

    print(
        f"Requests loaded: {len(df)}"
    )

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate the five features used by SmartEdge 3.0.

    past_frequency:
        Number of earlier requests for the same URL.

    recency_seconds:
        Seconds since the previous request for the same URL.
        First occurrence gets a large value.
    """

    print("\nGenerating request features...")

    timestamps = df["timestamp"]
    urls = df["url"].to_numpy()

    hour = timestamps.dt.hour.to_numpy(dtype=np.float64)
    day_of_week = timestamps.dt.dayofweek.to_numpy(
        dtype=np.float64
    )
    content_size = df["response_bytes"].to_numpy(
        dtype=np.float64
    )

    n = len(df)

    past_frequency = np.zeros(
        n,
        dtype=np.float64
    )

    recency_seconds = np.full(
        n,
        1_000_000.0,
        dtype=np.float64
    )

    # Streaming calculation. This is much faster than pandas
    # groupby/shift for the small simulation stream and exactly
    # preserves "past" information.
    frequency_map: Dict[str, int] = {}
    last_timestamp: Dict[str, pd.Timestamp] = {}

    ts_values = timestamps.to_numpy()

    for i in range(n):
        url = urls[i]

        count = frequency_map.get(url, 0)
        past_frequency[i] = count

        previous = last_timestamp.get(url)

        if previous is not None:
            delta = (
                ts_values[i] - previous
            )
            recency_seconds[i] = (
                delta / np.timedelta64(1, "s")
            )

            # Keep recency non-negative even if the input
            # contains an out-of-order timestamp.
            if recency_seconds[i] < 0:
                recency_seconds[i] = 0.0

        frequency_map[url] = count + 1
        last_timestamp[url] = ts_values[i]

    features = pd.DataFrame(
        {
            "hour": hour,
            "day_of_week": day_of_week,
            "past_frequency": past_frequency,
            "recency_seconds": recency_seconds,
            "content_size": content_size,
        }
    )

    print(
        f"Feature matrix: {features.shape}"
    )

    return features


def run_lru(urls: np.ndarray, capacity: int) -> Dict:
    """
    Standard LRU cache.
    """

    cache = OrderedDict()
    hits = 0
    misses = 0
    evictions = 0

    for url in urls:
        if url in cache:
            hits += 1
            cache.move_to_end(url)
            continue

        misses += 1

        if capacity <= 0:
            continue

        if len(cache) >= capacity:
            cache.popitem(last=False)
            evictions += 1

        cache[url] = True

    total = len(urls)

    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": (
            hits / total * 100
            if total else 0.0
        ),
        "evictions": evictions,
        "cache_size": len(cache),
    }


def run_lfu(urls: np.ndarray, capacity: int) -> Dict:
    """
    Standard LFU cache.

    Eviction priority:
        1. Lowest frequency
        2. Oldest last-seen request

    This gives deterministic LFU behavior.
    """

    cache = set()
    frequency: Dict[str, int] = {}
    last_seen: Dict[str, int] = {}

    hits = 0
    misses = 0
    evictions = 0

    for index, url in enumerate(urls):
        if url in cache:
            hits += 1
            frequency[url] += 1
            last_seen[url] = index
            continue

        misses += 1

        if capacity <= 0:
            continue

        if len(cache) >= capacity:
            victim = min(
                cache,
                key=lambda x: (
                    frequency[x],
                    last_seen[x],
                )
            )

            cache.remove(victim)
            evictions += 1

        cache.add(url)
        frequency[url] = 1
        last_seen[url] = index

    total = len(urls)

    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": (
            hits / total * 100
            if total else 0.0
        ),
        "evictions": evictions,
        "cache_size": len(cache),
    }


def generate_predictions(
    features: pd.DataFrame,
    model,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate model outputs once.

    Returns:
        reusable_probability
        expected_cache_value
    """

    print("\nGenerating SmartEdge 3.0 predictions...")

    X = features[
        FEATURES
    ].astype(float)

    probabilities = model.predict_proba(X)

    classes = [
        int(c)
        for c in model.classes_
    ]

    # Map class -> value.
    class_values = {
        0: 0.0,
        1: 1.0,
        2: 2.0,
        3: 3.0,
        4: 4.0,
    }

    expected_value = np.zeros(
        len(X),
        dtype=np.float64
    )

    reusable_probability = np.zeros(
        len(X),
        dtype=np.float64
    )

    for column, cls in enumerate(classes):
        p = probabilities[:, column]

        expected_value += (
            p * class_values.get(
                cls,
                0.0
            )
        )

        if cls > 0:
            reusable_probability += p

    print("Predictions completed.")

    print(
        "Average reusable probability:",
        f"{reusable_probability.mean():.4f}"
    )

    print(
        "Average expected cache value:",
        f"{expected_value.mean():.4f}"
    )

    return (
        reusable_probability,
        expected_value,
    )


def run_smartedge(
    df: pd.DataFrame,
    reusable_probability: np.ndarray,
    expected_value: np.ndarray,
    capacity: int,
    threshold: float,
) -> Dict:
    """
    Fast SmartEdge 3.0 simulation.

    Admission:
        reusable_probability >= threshold

    Eviction:
        remove the cached object with the lowest SmartEdge score.

    Score:
        expected_value
        * probability
        * frequency factor
        * recency factor
        * size factor
    """

    urls = df["url"].to_numpy()
    sizes = df["response_bytes"].to_numpy(
        dtype=np.float64
    )

    n = len(urls)

    # Cache metadata stored in dictionaries.
    frequency: Dict[str, int] = {}
    last_seen: Dict[str, int] = {}
    probabilities: Dict[str, float] = {}
    expected_values: Dict[str, float] = {}
    object_sizes: Dict[str, float] = {}

    cache = set()

    hits = 0
    misses = 0
    admissions = 0
    rejected = 0
    evictions = 0

    total_probability = float(
        reusable_probability.sum()
    )

    total_expected_value = float(
        expected_value.sum()
    )

    def score(url: str, current_index: int) -> float:
        freq = frequency[url]
        last = last_seen[url]
        probability = probabilities[url]
        value = expected_values[url]
        size = object_sizes[url]

        frequency_factor = np.log1p(freq)

        age = max(
            current_index - last,
            1
        )

        recency_factor = (
            1.0 /
            (
                1.0 +
                np.log1p(age)
            )
        )

        size_factor = (
            1.0 /
            (
                1.0 +
                np.log1p(size) / 10.0
            )
        )

        return (
            value
            * probability
            * frequency_factor
            * recency_factor
            * size_factor
        )

    for i in range(n):
        url = urls[i]

        probability = float(
            reusable_probability[i]
        )

        value = float(
            expected_value[i]
        )

        if url in cache:
            hits += 1

            frequency[url] += 1
            last_seen[url] = i
            probabilities[url] = probability
            expected_values[url] = value

            continue

        misses += 1

        # Admission threshold.
        if probability < threshold:
            rejected += 1
            continue

        size = max(
            float(sizes[i]),
            1.0
        )

        # Cache has free space.
        if len(cache) < capacity:
            cache.add(url)

            frequency[url] = 1
            last_seen[url] = i
            probabilities[url] = probability
            expected_values[url] = value
            object_sizes[url] = size

            admissions += 1
            continue

        # Cache is full.
        if capacity <= 0:
            rejected += 1
            continue

        # Find lowest-value cached object.
        victim = min(
            cache,
            key=lambda x: score(
                x,
                i
            )
        )

        victim_score = score(
            victim,
            i
        )

        # Score for a new object with frequency=1,
        # recency age=0 -> factor 1.0.
        new_frequency_factor = np.log1p(1)

        new_recency_factor = 1.0

        new_size_factor = (
            1.0 /
            (
                1.0 +
                np.log1p(size) / 10.0
            )
        )

        new_score = (
            value
            * probability
            * new_frequency_factor
            * new_recency_factor
            * new_size_factor
        )

        if new_score > victim_score:
            cache.remove(victim)

            del frequency[victim]
            del last_seen[victim]
            del probabilities[victim]
            del expected_values[victim]
            del object_sizes[victim]

            evictions += 1

            cache.add(url)

            frequency[url] = 1
            last_seen[url] = i
            probabilities[url] = probability
            expected_values[url] = value
            object_sizes[url] = size

            admissions += 1
        else:
            rejected += 1

    hit_rate = (
        hits / n * 100
        if n else 0.0
    )

    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": hit_rate,
        "evictions": evictions,
        "admissions": admissions,
        "rejected": rejected,
        "cache_size": len(cache),
        "average_probability": (
            total_probability / n
            if n else 0.0
        ),
        "average_expected_value": (
            total_expected_value / n
            if n else 0.0
        ),
    }


def print_header():
    print("=" * 100)
    print(
        "SMARTEDGE 3.0 - ADAPTIVE THRESHOLD EXPERIMENT"
    )
    print("=" * 100)


def main():
    args = parse_arguments()

    start_time = time.time()

    capacities = DEFAULT_CAPACITIES
    thresholds = DEFAULT_THRESHOLDS

    print_header()

    print(
        f"\nRequests per experiment : {args.requests}"
    )

    print(
        "Capacities              :",
        capacities
    )

    print(
        "Thresholds              :",
        thresholds
    )

    # --------------------------------------------------------
    # Load everything ONCE.
    # --------------------------------------------------------

    model = load_model(
        args.model
    )

    df = load_dataset(
        args.dataset,
        args.requests
    )

    features = prepare_features(
        df
    )

    urls = df["url"].to_numpy()

    reusable_probability, expected_value = (
        generate_predictions(
            features,
            model
        )
    )

    # --------------------------------------------------------
    # Baselines.
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("CALCULATING LRU / LFU BASELINES")
    print("=" * 100)

    baseline_results = {}

    for capacity in capacities:
        print(
            f"\nCapacity {capacity}: LRU / LFU"
        )

        lru = run_lru(
            urls,
            capacity
        )

        lfu = run_lfu(
            urls,
            capacity
        )

        baseline_results[capacity] = {
            "lru": lru,
            "lfu": lfu,
        }

        print(
            f"  LRU : {lru['hit_rate']:.2f}%"
        )

        print(
            f"  LFU : {lfu['hit_rate']:.2f}%"
        )

    # --------------------------------------------------------
    # SmartEdge threshold experiments.
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("TESTING SMARTEDGE THRESHOLDS")
    print("=" * 100)

    all_results = []
    best_results = []

    for capacity in capacities:
        print("\n" + "-" * 100)
        print(
            f"CACHE CAPACITY: {capacity}"
        )
        print("-" * 100)

        best = None

        for threshold in thresholds:
            print(
                f"\nTesting capacity={capacity}, "
                f"threshold={threshold}"
            )

            result = run_smartedge(
                df=df,
                reusable_probability=reusable_probability,
                expected_value=expected_value,
                capacity=capacity,
                threshold=threshold,
            )

            lru_rate = baseline_results[
                capacity
            ]["lru"]["hit_rate"]

            lfu_rate = baseline_results[
                capacity
            ]["lfu"]["hit_rate"]

            vs_lru = (
                result["hit_rate"]
                - lru_rate
            )

            vs_lfu = (
                result["hit_rate"]
                - lfu_rate
            )

            row = {
                "capacity": capacity,
                "threshold": threshold,
                "lru_hit_rate": lru_rate,
                "lfu_hit_rate": lfu_rate,
                "smartedge_hit_rate": result[
                    "hit_rate"
                ],
                "vs_lru": vs_lru,
                "vs_lfu": vs_lfu,
                "admissions": result[
                    "admissions"
                ],
                "rejected": result[
                    "rejected"
                ],
                "evictions": result[
                    "evictions"
                ],
                "cache_size": result[
                    "cache_size"
                ],
                "average_probability": result[
                    "average_probability"
                ],
                "average_expected_value": result[
                    "average_expected_value"
                ],
            }

            all_results.append(row)

            print(
                f"  SmartEdge : "
                f"{result['hit_rate']:.2f}%"
            )

            print(
                f"  Admissions: "
                f"{result['admissions']}"
            )

            print(
                f"  Evictions : "
                f"{result['evictions']}"
            )

            if (
                best is None
                or result["hit_rate"]
                > best["smartedge_hit_rate"]
            ):
                best = row

        print("\nBEST FOR THIS CAPACITY")

        print(
            f"  Threshold : "
            f"{best['threshold']}"
        )

        print(
            f"  Hit Rate  : "
            f"{best['smartedge_hit_rate']:.2f}%"
        )

        print(
            f"  vs LRU    : "
            f"{best['vs_lru']:+.2f} pp"
        )

        print(
            f"  vs LFU    : "
            f"{best['vs_lfu']:+.2f} pp"
        )

        best_results.append(best)

    # --------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print(
        "SMARTEDGE 3.0 - ADAPTIVE THRESHOLD SUMMARY"
    )
    print("=" * 100)

    print(
        f"{'Capacity':>10}"
        f"{'LRU':>12}"
        f"{'LFU':>12}"
        f"{'Threshold':>12}"
        f"{'SmartEdge':>14}"
        f"{'vs LRU':>12}"
        f"{'vs LFU':>12}"
    )

    print("-" * 100)

    for row in best_results:
        print(
            f"{row['capacity']:>10}"
            f"{row['lru_hit_rate']:>11.2f}%"
            f"{row['lfu_hit_rate']:>11.2f}%"
            f"{row['threshold']:>12.1f}"
            f"{row['smartedge_hit_rate']:>13.2f}%"
            f"{row['vs_lru']:>+11.2f}"
            f"{row['vs_lfu']:>+11.2f}"
        )

    # Best absolute SmartEdge result.
    best_hit = max(
        best_results,
        key=lambda x: x[
            "smartedge_hit_rate"
        ]
    )

    # Best improvement over LRU.
    best_lru = max(
        best_results,
        key=lambda x: x["vs_lru"]
    )

    # Best improvement over LFU.
    best_lfu = max(
        best_results,
        key=lambda x: x["vs_lfu"]
    )

    print("\n" + "=" * 100)
    print("BEST SMARTEDGE HIT RATE")
    print("=" * 100)

    print(
        f"Capacity : "
        f"{best_hit['capacity']}"
    )

    print(
        f"Threshold: "
        f"{best_hit['threshold']}"
    )

    print(
        f"Hit Rate : "
        f"{best_hit['smartedge_hit_rate']:.2f}%"
    )

    print("\n" + "=" * 100)
    print("BEST IMPROVEMENT OVER LRU")
    print("=" * 100)

    print(
        f"Capacity    : "
        f"{best_lru['capacity']}"
    )

    print(
        f"Threshold   : "
        f"{best_lru['threshold']}"
    )

    print(
        f"Improvement : "
        f"{best_lru['vs_lru']:+.2f} percentage points"
    )

    print("\n" + "=" * 100)
    print("BEST IMPROVEMENT OVER LFU")
    print("=" * 100)

    print(
        f"Capacity    : "
        f"{best_lfu['capacity']}"
    )

    print(
        f"Threshold   : "
        f"{best_lfu['threshold']}"
    )

    print(
        f"Improvement : "
        f"{best_lfu['vs_lfu']:+.2f} percentage points"
    )

    # --------------------------------------------------------
    # Save every experiment, not just the best rows.
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        all_results
    )

    result_dir = os.path.dirname(
        os.path.abspath(args.result)
    )

    os.makedirs(
        result_dir,
        exist_ok=True
    )

    result_df.to_csv(
        args.result,
        index=False
    )

    print("\n" + "=" * 100)
    print("RESULTS SAVED")
    print("=" * 100)

    print(
        os.path.abspath(args.result)
    )

    elapsed = time.time() - start_time

    print(
        f"\nTotal experiment time: "
        f"{elapsed:.2f} seconds"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
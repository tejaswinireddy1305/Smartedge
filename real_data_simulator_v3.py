"""
SmartEdge 3.0 - Real Data Cache Simulator

Algorithms:
    1. LRU
    2. LFU
    3. SmartEdge 3.0 - ML-based predictive caching

Usage:
    python real_data_simulator_v3.py
    python real_data_simulator_v3.py --capacity 50
    python real_data_simulator_v3.py --capacity 100
    python real_data_simulator_v3.py --capacity 200
    python real_data_simulator_v3.py --capacity 500
"""

import sys
import os
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import joblib


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = r"D:\smart edge"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "smartedge_v3_random_forest.pkl"
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "clean_requests.csv"
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

DEFAULT_CACHE_CAPACITY = 100

CACHE_CAPACITY = DEFAULT_CACHE_CAPACITY

# Number of real requests used in the experiment.
REQUEST_LIMIT = 10000


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments():
    """
    Read command-line arguments.

    Example:
        python real_data_simulator_v3.py --capacity 50
    """

    global CACHE_CAPACITY

    if "--capacity" in sys.argv:

        index = sys.argv.index("--capacity")

        if index + 1 >= len(sys.argv):
            print("ERROR: --capacity requires a number.")
            sys.exit(1)

        try:
            CACHE_CAPACITY = int(sys.argv[index + 1])

        except ValueError:
            print("ERROR: Cache capacity must be an integer.")
            sys.exit(1)

        if CACHE_CAPACITY <= 0:
            print("ERROR: Cache capacity must be greater than 0.")
            sys.exit(1)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Loading SmartEdge 3.0 ML model...")

    if not os.path.exists(MODEL_PATH):

        print("\nERROR: Model file not found:")
        print(MODEL_PATH)

        sys.exit(1)

    model = joblib.load(MODEL_PATH)

    print(f"Model: {type(model).__name__}")

    features = getattr(
        model,
        "feature_names_in_",
        None
    )

    print(f"Features: {list(features) if features is not None else 'Unknown'}")

    return model


# ============================================================
# LOAD NASA DATASET
# ============================================================

def load_dataset():

    print("\nLoading NASA request dataset...")

    if not os.path.exists(DATASET_PATH):

        print("\nERROR: Dataset not found:")
        print(DATASET_PATH)

        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)

    if len(df) == 0:

        print("ERROR: Dataset is empty.")
        sys.exit(1)

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # Remove invalid timestamps
    df = df.dropna(
        subset=["timestamp"]
    ).copy()

    # Make sure URL exists
    df["url"] = df["url"].astype(str)

    # Response size
    df["response_bytes"] = pd.to_numeric(
        df["response_bytes"],
        errors="coerce"
    ).fillna(0)

    # Limit requests
    df = df.head(REQUEST_LIMIT).copy()

    print(f"Requests loaded: {len(df)}")

    return df


# ============================================================
# FEATURE GENERATION
# ============================================================

def prepare_features(df):

    """
    Generate the five features expected by SmartEdge 3.0.

    Features:
        hour
        day_of_week
        past_frequency
        recency_seconds
        content_size
    """

    print("\nGenerating request features...")

    frequencies = defaultdict(int)

    last_seen = {}

    feature_rows = []

    for _, row in df.iterrows():

        url = row["url"]

        timestamp = row["timestamp"]

        # -----------------------------------------------
        # Time features
        # -----------------------------------------------

        hour = int(timestamp.hour)

        day_of_week = int(timestamp.dayofweek)

        # -----------------------------------------------
        # Frequency before current request
        # -----------------------------------------------

        past_frequency = frequencies[url]

        # -----------------------------------------------
        # Recency before current request
        # -----------------------------------------------

        if url in last_seen:

            recency_seconds = (
                timestamp - last_seen[url]
            ).total_seconds()

            if recency_seconds < 0:
                recency_seconds = 0

        else:

            # Large value means never seen before
            recency_seconds = 999999.0

        # -----------------------------------------------
        # Content size
        # -----------------------------------------------

        content_size = float(
            row["response_bytes"]
        )

        feature_rows.append({

            "hour": hour,

            "day_of_week": day_of_week,

            "past_frequency": float(
                past_frequency
            ),

            "recency_seconds": float(
                recency_seconds
            ),

            "content_size": content_size
        })

        # Update history AFTER generating features
        frequencies[url] += 1

        last_seen[url] = timestamp

    features = pd.DataFrame(
        feature_rows
    )

    return features


# ============================================================
# LRU CACHE
# ============================================================

def run_lru(requests, capacity):

    cache = OrderedDict()

    hits = 0

    misses = 0

    evictions = 0

    for url in requests:

        # HIT
        if url in cache:

            hits += 1

            # Move recently used item to end
            cache.move_to_end(url)

        # MISS
        else:

            misses += 1

            if len(cache) >= capacity:

                cache.popitem(
                    last=False
                )

                evictions += 1

            cache[url] = True

    hit_rate = (
        hits / len(requests)
    ) * 100

    return {

        "hits": hits,

        "misses": misses,

        "hit_rate": hit_rate,

        "evictions": evictions,

        "cache_size": len(cache)
    }


# ============================================================
# LFU CACHE
# ============================================================

def run_lfu(requests, capacity):

    cache = set()

    frequency = defaultdict(int)

    insertion_order = {}

    counter = 0

    hits = 0

    misses = 0

    evictions = 0

    for url in requests:

        counter += 1

        # HIT
        if url in cache:

            hits += 1

            frequency[url] += 1

        # MISS
        else:

            misses += 1

            if len(cache) >= capacity:

                # Find least frequently used item.
                #
                # If frequencies are equal,
                # remove the oldest inserted item.

                victim = min(
                    cache,
                    key=lambda item: (
                        frequency[item],
                        insertion_order[item]
                    )
                )

                cache.remove(victim)

                frequency.pop(
                    victim,
                    None
                )

                insertion_order.pop(
                    victim,
                    None
                )

                evictions += 1

            cache.add(url)

            frequency[url] = 1

            insertion_order[url] = counter

    hit_rate = (
        hits / len(requests)
    ) * 100

    return {

        "hits": hits,

        "misses": misses,

        "hit_rate": hit_rate,

        "evictions": evictions,

        "cache_size": len(cache)
    }


# ============================================================
# SMARTEDGE 3.0
# ============================================================

def run_smartedge(
    df,
    features,
    model,
    capacity,
    threshold=0.5
):

    """
    SmartEdge 3.0 predictive caching.

    The ML model predicts five cache-value classes:

        0 = NO_REUSE
        1 = LOW_REUSE
        2 = MEDIUM_REUSE
        3 = HIGH_REUSE
        4 = VERY_HIGH_REUSE

    The model probability is converted into an
    expected cache value.

    Expected value:

        P(class) * class_value

    This is used for admission and eviction.
    """

    print("\nGenerating SmartEdge 3.0 predictions...")

    # --------------------------------------------------------
    # Make sure features match training order
    # --------------------------------------------------------

    expected_features = [
        "hour",
        "day_of_week",
        "past_frequency",
        "recency_seconds",
        "content_size"
    ]

    X = features[
        expected_features
    ].astype(float)

    # --------------------------------------------------------
    # ML prediction
    # --------------------------------------------------------

    probabilities = model.predict_proba(X)

    classes = list(
        model.classes_
    )

    # --------------------------------------------------------
    # Expected cache value
    # --------------------------------------------------------

    class_values = {
        0: 0.0,
        1: 1.0,
        2: 2.0,
        3: 3.0,
        4: 4.0
    }

    expected_values = []

    reusable_probabilities = []

    for row in probabilities:

        expected_value = 0.0

        reusable_probability = 0.0

        for probability, cls in zip(
            row,
            classes
        ):

            cls = int(cls)

            value = class_values.get(
                cls,
                0.0
            )

            expected_value += (
                probability * value
            )

            # Any reuse class above NO_REUSE
            # contributes to reusable probability.
            if cls > 0:

                reusable_probability += probability

        expected_values.append(
            expected_value
        )

        reusable_probabilities.append(
            reusable_probability
        )

    print("Predictions completed.")

    # --------------------------------------------------------
    # Cache state
    # --------------------------------------------------------

    cache = {}

    hits = 0

    misses = 0

    admissions = 0

    rejected = 0

    evictions = 0

    total_probability = 0.0

    total_expected_value = 0.0

    # --------------------------------------------------------
    # Process every request
    # --------------------------------------------------------

    for index, (_, row) in enumerate(
        df.iterrows()
    ):

        url = row["url"]

        probability = float(
            reusable_probabilities[index]
        )

        expected_value = float(
            expected_values[index]
        )

        total_probability += probability

        total_expected_value += expected_value

        # ----------------------------------------------------
        # CACHE HIT
        # ----------------------------------------------------

        if url in cache:

            hits += 1

            # Update metadata
            cache[url]["frequency"] += 1

            cache[url]["last_seen"] = index

            # Update predicted value
            cache[url]["probability"] = probability

            cache[url]["expected_value"] = expected_value

            continue

        # ----------------------------------------------------
        # CACHE MISS
        # ----------------------------------------------------

        misses += 1

        # ----------------------------------------------------
        # Admission decision
        # ----------------------------------------------------

        # A request is admitted if it has some predicted
        # reuse probability.
        #
        # This allows the model's expected cache value
        # to determine how valuable the request is.

        should_cache = (
    probability >= threshold
)

        if not should_cache:

            rejected += 1

            continue

        # ----------------------------------------------------
        # CACHE HAS SPACE
        # ----------------------------------------------------

        if len(cache) < capacity:

            cache[url] = {

                "frequency": 1,

                "last_seen": index,

                "probability": probability,

                "expected_value": expected_value,

                "size": max(
                    float(row["response_bytes"]),
                    1.0
                )
            }

            admissions += 1

            continue

        # ----------------------------------------------------
        # CACHE FULL
        # ----------------------------------------------------

        # Calculate SmartEdge score.
        #
        # We combine:
        #
        #   expected reuse
        #   frequency
        #   recency
        #   object size
        #
        # The goal is to remove the lowest-value
        # cached object.

        def cache_score(item):

            data = cache[item]

            frequency = data["frequency"]

            last_seen = data["last_seen"]

            probability = data["probability"]

            expected_value = data["expected_value"]

            size = data["size"]

            # Frequency component
            frequency_factor = np.log1p(
                frequency
            )

            # Recency component
            age = max(
                index - last_seen,
                1
            )

            recency_factor = 1.0 / (
                1.0 + np.log1p(age)
            )

            # Size penalty
            size_factor = 1.0 / (
                1.0 + np.log1p(size) / 10.0
            )

            score = (

                expected_value

                * probability

                * frequency_factor

                * recency_factor

                * size_factor
            )

            return score

        victim = min(
            cache.keys(),
            key=cache_score
        )

        victim_score = cache_score(
            victim
        )

        new_frequency_factor = np.log1p(1)

        new_recency_factor = 1.0

        new_size = max(
            float(row["response_bytes"]),
            1.0
        )

        new_size_factor = 1.0 / (
            1.0 + np.log1p(new_size) / 10.0
        )

        new_score = (

            expected_value

            * probability

            * new_frequency_factor

            * new_recency_factor

            * new_size_factor
        )

        # ----------------------------------------------------
        # Replace only if new item is better
        # ----------------------------------------------------

        if new_score > victim_score:

            del cache[victim]

            evictions += 1

            cache[url] = {

                "frequency": 1,

                "last_seen": index,

                "probability": probability,

                "expected_value": expected_value,

                "size": new_size
            }

            admissions += 1

        else:

            rejected += 1

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    total_requests = len(df)

    hit_rate = (
        hits / total_requests
    ) * 100

    average_probability = (
        total_probability / total_requests
    )

    average_expected_value = (
        total_expected_value / total_requests
    )

    return {

        "hits": hits,

        "misses": misses,

        "hit_rate": hit_rate,

        "evictions": evictions,

        "admissions": admissions,

        "rejected": rejected,

        "cache_size": len(cache),

        "average_probability":
            average_probability,

        "average_expected_value":
            average_expected_value
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():

    parse_arguments()

    print("\n")
    print("=" * 65)
    print("SMARTEDGE 3.0 REAL-DATA EXPERIMENT")
    print("=" * 65)

    print(
        f"Requests       : {REQUEST_LIMIT}"
    )

    print(
        f"Cache capacity : {CACHE_CAPACITY}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Generate features
    # --------------------------------------------------------

    features = prepare_features(
        df
    )

    # --------------------------------------------------------
    # Request stream
    # --------------------------------------------------------

    requests = df[
        "url"
    ].tolist()

    # --------------------------------------------------------
    # LRU
    # --------------------------------------------------------

    print("\nRunning LRU...")

    lru = run_lru(
        requests,
        CACHE_CAPACITY
    )

    # --------------------------------------------------------
    # LFU
    # --------------------------------------------------------

    print("Running LFU...")

    lfu = run_lfu(
        requests,
        CACHE_CAPACITY
    )

    # --------------------------------------------------------
    # SmartEdge
    # --------------------------------------------------------

    print("Running SmartEdge 3.0...")

    smartedge = run_smartedge(
        df,
        features,
        model,
        CACHE_CAPACITY
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    print(
        f"LRU       : {lru['hit_rate']:.2f}%"
    )

    print(
        f"LFU       : {lfu['hit_rate']:.2f}%"
    )

    print(
        f"SmartEdge : {smartedge['hit_rate']:.2f}%"
    )

    print("\nEvictions")

    print(
        f"LRU       : {lru['evictions']}"
    )

    print(
        f"LFU       : {lfu['evictions']}"
    )

    print(
        f"SmartEdge : {smartedge['evictions']}"
    )

    print("\nSmartEdge 3.0 ML")

    print(
        f"Admissions          : "
        f"{smartedge['admissions']}"
    )

    print(
        f"Rejected             : "
        f"{smartedge['rejected']}"
    )

    print(
        f"Cache size           : "
        f"{smartedge['cache_size']}"
    )

    print(
        f"Average reusable probability: "
        f"{smartedge['average_probability']:.4f}"
    )

    print(
        f"Average expected cache value: "
        f"{smartedge['average_expected_value']:.4f}"
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("COMPARISON")
    print("=" * 65)

    smartedge_vs_lru = (
        smartedge["hit_rate"]
        - lru["hit_rate"]
    )

    smartedge_vs_lfu = (
        smartedge["hit_rate"]
        - lfu["hit_rate"]
    )

    print(
        f"SmartEdge vs LRU : "
        f"{smartedge_vs_lru:+.2f} percentage points"
    )

    print(
        f"SmartEdge vs LFU : "
        f"{smartedge_vs_lfu:+.2f} percentage points"
    )

    print("=" * 65)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_experiment()
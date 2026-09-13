import random
import math
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta

import joblib
import pandas as pd


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = r"D:\smart edge\model\smartedge_random_forest.pkl"

model = joblib.load(MODEL_PATH)

print("SmartEdge model loaded successfully")


# ============================================================
# REQUEST GENERATOR
# ============================================================

def generate_requests(total_requests=10000, content_count=500, seed=42):

    random.seed(seed)

    contents = [
        f"content_{i}"
        for i in range(content_count)
    ]

    # Popularity distribution.
    # Lower-ranked content is requested less often.
    weights = [
        1 / ((i + 1) ** 1.2)
        for i in range(content_count)
    ]

    start_time = datetime.now()

    requests = []

    for i in range(total_requests):

        content = random.choices(
            contents,
            weights=weights,
            k=1
        )[0]

        # Simulated request time
        request_time = start_time + timedelta(
            seconds=i
        )

        # Stable content size
        content_id = int(content.split("_")[1])

        random.seed(content_id)

        content_size = random.randint(
            500,
            5000
        )

        requests.append({
            "content": content,
            "time": request_time,
            "content_size": content_size
        })

        random.seed()

    return requests


# ============================================================
# LRU
# ============================================================

def run_lru(requests, cache_capacity):

    cache = OrderedDict()

    hits = 0
    misses = 0
    evictions = 0

    for request in requests:

        content = request["content"]

        if content in cache:

            hits += 1

            cache.move_to_end(content)

        else:

            misses += 1

            if len(cache) >= cache_capacity:

                cache.popitem(last=False)

                evictions += 1

            cache[content] = True

    return {
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "hit_ratio": hits / len(requests),
        "cache_size": len(cache)
    }


# ============================================================
# LFU
# ============================================================

def run_lfu(requests, cache_capacity):

    cache = set()

    frequency = defaultdict(int)

    last_used = {}

    hits = 0
    misses = 0
    evictions = 0

    for index, request in enumerate(requests):

        content = request["content"]

        frequency[content] += 1

        if content in cache:

            hits += 1

        else:

            misses += 1

            if len(cache) >= cache_capacity:

                victim = min(
                    cache,
                    key=lambda x: (
                        frequency[x],
                        last_used.get(x, -1)
                    )
                )

                cache.remove(victim)

                evictions += 1

            cache.add(content)

        last_used[content] = index

    return {
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "hit_ratio": hits / len(requests),
        "cache_size": len(cache)
    }


# ============================================================
# SMARTEDGE
# ============================================================

def run_smartedge(
    requests,
    cache_capacity,
    threshold=0.50
):
    """
    SmartEdge 2.0

    Phase 1:
        Generate ML features for the complete request stream.

    Phase 2:
        Run Random Forest predictions in ONE batch.

    Phase 3:
        Simulate intelligent cache admission/replacement.
    """

    import numpy as np

    # ========================================================
    # PHASE 1 — BUILD FEATURES
    # ========================================================

    frequency = defaultdict(int)
    last_request = {}

    feature_rows = []

    for request in requests:

        content = request["content"]
        current_time = request["time"]

        hour = current_time.hour

        day_of_week = current_time.weekday()

        past_frequency = frequency[content]

        if content in last_request:

            recency_seconds = (
                current_time - last_request[content]
            ).total_seconds()

        else:

            recency_seconds = 999999

        content_size = request["content_size"]

        feature_rows.append([
            hour,
            day_of_week,
            past_frequency,
            recency_seconds,
            content_size
        ])

        # Update history AFTER feature extraction
        frequency[content] += 1
        last_request[content] = current_time

    # ========================================================
    # PHASE 2 — BATCH ML PREDICTION
    # ========================================================

    print("Generating ML predictions...")

    X = np.asarray(
        feature_rows,
        dtype=np.float64
    )

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)

        # Find probability of class 1
        if hasattr(model, "classes_"):

            classes = list(model.classes_)

            if 1 in classes:

                class_index = classes.index(1)

                prediction_probabilities = (
                    probabilities[:, class_index]
                )

            else:

                prediction_probabilities = (
                    probabilities[:, -1]
                )

        else:

            prediction_probabilities = (
                probabilities[:, -1]
            )

    else:

        predictions = model.predict(X)

        prediction_probabilities = (
            np.asarray(predictions, dtype=float)
        )

    print("ML predictions completed.")

    # ========================================================
    # PHASE 3 — SMARTEDGE CACHE SIMULATION
    # ========================================================

    cache = {}

    frequency = defaultdict(int)
    last_request = {}

    hits = 0
    misses = 0
    evictions = 0
    admissions = 0

    for index, request in enumerate(requests):

        content = request["content"]
        current_time = request["time"]

        probability = float(
            prediction_probabilities[index]
        )

        # ----------------------------------------------------
        # CACHE HIT
        # ----------------------------------------------------

        if content in cache:

            hits += 1

            cache[content]["last_used"] = current_time
            cache[content]["frequency"] += 1
            cache[content]["probability"] = probability

        # ----------------------------------------------------
        # CACHE MISS
        # ----------------------------------------------------

        else:

            misses += 1

            # -----------------------------------------------
            # SMARTEDGE ADMISSION
            # -----------------------------------------------

            if probability >= threshold:

                admissions += 1

                new_item = {
                    "probability": probability,
                    "frequency": 1,
                    "last_used": current_time
                }

                # -------------------------------------------
                # CACHE HAS SPACE
                # -------------------------------------------

                if len(cache) < cache_capacity:

                    cache[content] = new_item

                # -------------------------------------------
                # CACHE FULL
                # -------------------------------------------

                else:

                    def utility(item):

                        metadata = item[1]

                        prob = metadata["probability"]

                        freq_score = min(
                            math.log1p(
                                metadata["frequency"]
                            ) / 5,
                            1
                        )

                        age_seconds = max(
                            (
                                current_time
                                - metadata["last_used"]
                            ).total_seconds(),
                            1
                        )

                        recency_score = 1 / (
                            1 + age_seconds / 60
                        )

                        return (
                            0.60 * prob
                            +
                            0.25 * freq_score
                            +
                            0.15 * recency_score
                        )

                    # Find least valuable item
                    victim = min(
                        cache.items(),
                        key=utility
                    )[0]

                    victim_utility = utility(
                        (
                            victim,
                            cache[victim]
                        )
                    )

                    # ---------------------------------------
                    # Calculate incoming utility
                    # ---------------------------------------

                    incoming_frequency = 1

                    incoming_recency = 1.0

                    incoming_utility = (
                        0.60 * probability
                        +
                        0.25 * min(
                            math.log1p(
                                incoming_frequency
                            ) / 5,
                            1
                        )
                        +
                        0.15 * incoming_recency
                    )

                    # ---------------------------------------
                    # Replace only if incoming item is better
                    # ---------------------------------------

                    if incoming_utility > victim_utility:

                        del cache[victim]

                        cache[content] = new_item

                        evictions += 1

        frequency[content] += 1
        last_request[content] = current_time

    # ========================================================
    # RESULTS
    # ========================================================

    hit_ratio = (
        hits / len(requests)
        if requests
        else 0
    )

    average_prediction = (
        float(np.mean(prediction_probabilities))
        if len(prediction_probabilities) > 0
        else 0
    )

    return {
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "admissions": admissions,
        "average_prediction": average_prediction,
        "hit_ratio": hit_ratio,
        "cache_size": len(cache)
    }

    cache = {}

    frequency = defaultdict(int)

    last_request = {}

    hits = 0
    misses = 0
    evictions = 0
    admissions = 0

    prediction_sum = 0

    for request in requests:

        content = request["content"]

        current_time = request["time"]

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        hour = current_time.hour

        day_of_week = current_time.weekday()

        past_frequency = frequency[content]

        if content in last_request:

            recency_seconds = (
                current_time -
                last_request[content]
            ).total_seconds()

        else:

            recency_seconds = 999999

        content_size = request["content_size"]

        features = pd.DataFrame(
            [[
                hour,
                day_of_week,
                past_frequency,
                recency_seconds,
                content_size
            ]],
            columns=[
                "hour",
                "day_of_week",
                "past_frequency",
                "recency_seconds",
                "content_size"
            ]
        )

        # ----------------------------------------------------
        # ML PREDICTION
        # ----------------------------------------------------

        probability = 0.0

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                features
            )

            if probabilities.shape[1] > 1:

                probability = float(
                    probabilities[0][1]
                )

            else:

                probability = float(
                    probabilities[0][0]
                )

        else:

            prediction = model.predict(
                features
            )[0]

            probability = float(prediction)

        prediction_sum += probability

        # ----------------------------------------------------
        # REQUEST STATISTICS
        # ----------------------------------------------------

        frequency[content] += 1

        last_request[content] = current_time

        # ----------------------------------------------------
        # CACHE HIT
        # ----------------------------------------------------

        if content in cache:

            hits += 1

            # Refresh cached metadata
            cache[content]["last_used"] = current_time
            cache[content]["frequency"] = frequency[content]
            cache[content]["probability"] = probability

            continue

        # ----------------------------------------------------
        # CACHE MISS
        # ----------------------------------------------------

        misses += 1

        # ----------------------------------------------------
        # CACHE ADMISSION
        # ----------------------------------------------------

        if probability < threshold:

            continue

        admissions += 1

        # ----------------------------------------------------
        # CACHE HAS SPACE
        # ----------------------------------------------------

        if len(cache) < cache_capacity:

            cache[content] = {
                "probability": probability,
                "frequency": frequency[content],
                "last_used": current_time
            }

            continue

        # ----------------------------------------------------
        # CACHE IS FULL
        # ----------------------------------------------------

        # Calculate utility for every cached object.
        #
        # Higher:
        #   prediction probability
        #   frequency
        #   recency
        #
        # means more valuable.

        def utility(item):

            metadata = item[1]

            probability_score = metadata[
                "probability"
            ]

            frequency_score = math.log1p(
                metadata["frequency"]
            )

            age_seconds = max(
                (
                    current_time -
                    metadata["last_used"]
                ).total_seconds(),
                1
            )

            recency_score = 1 / (
                1 + age_seconds / 60
            )

            return (
                probability_score *
                0.60
                +
                min(frequency_score / 5, 1)
                * 0.25
                +
                recency_score
                * 0.15
            )

        # Find least valuable cached object
        victim = min(
            cache.items(),
            key=utility
        )[0]

        new_item = {
            "probability": probability,
            "frequency": frequency[content],
            "last_used": current_time
        }

        # Compare incoming item against victim
        incoming_utility = (
            probability * 0.60
            +
            min(
                math.log1p(
                    frequency[content]
                ) / 5,
                1
            ) * 0.25
            +
            0.15
        )

        victim_utility = utility(
            (victim, cache[victim])
        )

        if incoming_utility > victim_utility:

            del cache[victim]

            cache[content] = new_item

            evictions += 1

    average_prediction = (
        prediction_sum / len(requests)
    )

    return {
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "admissions": admissions,
        "average_prediction": average_prediction,
        "hit_ratio": hits / len(requests),
        "cache_size": len(cache)
    }


# ============================================================
# COMPLETE EXPERIMENT
# ============================================================

def run_experiment(
    requests=10000,
    cache_capacity=100,
    threshold=0.50
):

    print("\n" + "=" * 60)
    print("SMARTEDGE EXPERIMENT")
    print("=" * 60)

    print(f"Requests      : {requests}")
    print(f"Cache capacity: {cache_capacity}")
    print(f"Threshold     : {threshold}")

    print("\nGenerating request stream...")

    request_stream = generate_requests(
        total_requests=requests
    )

    print("Request stream generated.")

    print("\nRunning LRU...")

    lru = run_lru(
        request_stream,
        cache_capacity
    )

    print("Running LFU...")

    lfu = run_lfu(
        request_stream,
        cache_capacity
    )

    print("Running SmartEdge...")

    smartedge = run_smartedge(
        request_stream,
        cache_capacity,
        threshold
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        f"LRU       : {lru['hit_ratio'] * 100:.2f}%"
    )

    print(
        f"LFU       : {lfu['hit_ratio'] * 100:.2f}%"
    )

    print(
        f"SmartEdge : "
        f"{smartedge['hit_ratio'] * 100:.2f}%"
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

    print("\nSmartEdge ML")

    print(
        f"Predictions         : {requests}"
    )

    print(
        f"Cache admissions    : "
        f"{smartedge['admissions']}"
    )

    print(
        f"Average probability : "
        f"{smartedge['average_prediction']:.4f}"
    )

    print("=" * 60)

    return {
        "lru": lru,
        "lfu": lfu,
        "smartedge": smartedge
    }


if __name__ == "__main__":

    run_experiment(
        requests=10000,
        cache_capacity=100,
        threshold=0.50
    )
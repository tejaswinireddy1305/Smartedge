from collections import OrderedDict
from datetime import datetime
import time
import math


class SmartEdgeCache:
    """
    Live SmartEdge cache.

    Handles:
        - Cache HIT / MISS
        - ML-based admission
        - LRU-style fallback eviction
        - Performance statistics
    """

    def __init__(self, capacity=100):
        self.capacity = capacity
        self.cache = OrderedDict()

        self.total_requests = 0
        self.hits = 0
        self.misses = 0
        self.admissions = 0
        self.rejections = 0
        self.evictions = 0

        self.total_response_time = 0.0
        self.total_saved_time = 0.0

    # ---------------------------------------------------------
    # CACHE LOOKUP
    # ---------------------------------------------------------

    def get(self, url):
        """
        Return cached object if present.
        """

        self.total_requests += 1

        start = time.perf_counter()

        if url in self.cache:

            self.hits += 1

            item = self.cache.pop(url)

            item["hits"] += 1
            item["last_access"] = datetime.now().isoformat()

            self.cache[url] = item

            elapsed = (time.perf_counter() - start) * 1000

            self.total_response_time += elapsed

            return {
                "hit": True,
                "data": item,
                "response_time_ms": round(elapsed, 4)
            }

        self.misses += 1

        elapsed = (time.perf_counter() - start) * 1000

        self.total_response_time += elapsed

        return {
            "hit": False,
            "data": None,
            "response_time_ms": round(elapsed, 4)
        }

    # ---------------------------------------------------------
    # ADMISSION
    # ---------------------------------------------------------

    def admit(
        self,
        url,
        content_size,
        probability,
        expected_value,
        prediction
    ):
        """
        Decide whether a missed object should enter cache.
        """

        # SmartEdge admission threshold.
        #
        # We use 0.5 initially because this is the threshold
        # used by the trained SmartEdge model experiments.

        should_cache = probability >= 0.5

        if not should_cache:

            self.rejections += 1

            return {
                "admitted": False,
                "reason": "ML probability below threshold",
                "evicted": None
            }

        evicted = None

        # -----------------------------------------------------
        # SPACE AVAILABLE
        # -----------------------------------------------------

        if len(self.cache) < self.capacity:

            self.cache[url] = {
                "url": url,
                "content_size": int(max(content_size, 0)),
                "probability": round(float(probability), 4),
                "expected_value": round(float(expected_value), 4),
                "prediction": prediction,
                "hits": 0,
                "created_at": datetime.now().isoformat(),
                "last_access": datetime.now().isoformat()
            }

            self.admissions += 1

            return {
                "admitted": True,
                "reason": "Cache space available",
                "evicted": None
            }

        # -----------------------------------------------------
        # CACHE FULL
        # -----------------------------------------------------

        victim = self._find_victim()

        if victim is None:
            self.rejections += 1

            return {
                "admitted": False,
                "reason": "No suitable eviction candidate",
                "evicted": None
            }

        victim_item = self.cache[victim]

        victim_score = self._score(victim_item)

        new_item = {
            "url": url,
            "content_size": int(max(content_size, 0)),
            "probability": float(probability),
            "expected_value": float(expected_value),
            "prediction": prediction,
            "hits": 0,
            "created_at": datetime.now().isoformat(),
            "last_access": datetime.now().isoformat()
        }

        new_score = self._score(new_item)

        # Only replace a cached item if the new object
        # is predicted to be more valuable.

        if new_score > victim_score:

            del self.cache[victim]

            self.cache[url] = new_item

            self.admissions += 1
            self.evictions += 1

            evicted = victim

            return {
                "admitted": True,
                "reason": "New object has higher SmartEdge score",
                "evicted": evicted
            }

        self.rejections += 1

        return {
            "admitted": False,
            "reason": "Existing cached object has higher score",
            "evicted": None
        }

    # ---------------------------------------------------------
    # SMARTEDGE SCORE
    # ---------------------------------------------------------

    def _score(self, item):

        probability = float(
            item.get("probability", 0.0)
        )

        expected_value = float(
            item.get("expected_value", 0.0)
        )

        frequency = float(
            item.get("hits", 0)
        )

        size = max(
            float(item.get("content_size", 1)),
            1.0
        )

        frequency_factor = math.log1p(
            frequency + 1
        )

        size_factor = 1.0 / (
            1.0 + math.log1p(size) / 10.0
        )

        return (
            expected_value
            * probability
            * frequency_factor
            * size_factor
        )

    # ---------------------------------------------------------
    # FIND LOWEST VALUE ITEM
    # ---------------------------------------------------------

    def _find_victim(self):

        if not self.cache:
            return None

        return min(
            self.cache.keys(),
            key=lambda url: self._score(
                self.cache[url]
            )
        )

    # ---------------------------------------------------------
    # STATISTICS
    # ---------------------------------------------------------

    def stats(self):

        hit_rate = 0.0

        if self.total_requests > 0:
            hit_rate = (
                self.hits /
                self.total_requests
            ) * 100

        average_response_time = 0.0

        if self.total_requests > 0:
            average_response_time = (
                self.total_response_time /
                self.total_requests
            )

        return {
            "capacity": self.capacity,
            "current_cache_size": len(self.cache),

            "total_requests":
                self.total_requests,

            "hits":
                self.hits,

            "misses":
                self.misses,

            "hit_rate":
                round(hit_rate, 2),

            "admissions":
                self.admissions,

            "rejections":
                self.rejections,

            "evictions":
                self.evictions,

            "average_response_time_ms":
                round(
                    average_response_time,
                    4
                )
        }

    # ---------------------------------------------------------
    # CACHE CONTENTS
    # ---------------------------------------------------------

    def contents(self):

        result = []

        for url, item in self.cache.items():

            result.append({
                "url": url,
                "prediction":
                    item["prediction"],
                "probability":
                    item["probability"],
                "expected_value":
                    item["expected_value"],
                "hits":
                    item["hits"],
                "content_size":
                    item["content_size"],
                "last_access":
                    item["last_access"]
            })

        return result

    # ---------------------------------------------------------
    # CLEAR CACHE
    # ---------------------------------------------------------

    def clear(self):

        self.cache.clear()

        return {
            "status": "cleared"
        }
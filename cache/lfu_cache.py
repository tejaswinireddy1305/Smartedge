import pandas as pd
from collections import defaultdict

# ============================================================
# LFU Cache
# ============================================================

class LFUCache:

    def __init__(self, capacity):
        self.capacity = capacity

        # URL -> content size
        self.cache = {}

        # URL -> number of requests
        self.frequency = defaultdict(int)

        self.hits = 0
        self.misses = 0

    def get(self, key):

        if key in self.cache:

            self.frequency[key] += 1

            self.hits += 1

            return True

        self.misses += 1

        return False

    def put(self, key, size):

        # If already cached
        if key in self.cache:
            return

        # Cache not full
        if len(self.cache) < self.capacity:

            self.cache[key] = size
            self.frequency[key] = 1

            return

        # Find least frequently used item
        least_used = min(
            self.cache,
            key=lambda x: self.frequency[x]
        )

        # Remove it
        del self.cache[least_used]
        del self.frequency[least_used]

        # Add new content
        self.cache[key] = size
        self.frequency[key] = 1

    def hit_ratio(self):

        total = self.hits + self.misses

        if total == 0:
            return 0

        return self.hits / total


# ============================================================
# Load NASA workload
# ============================================================

INPUT_FILE = r"D:\smart edge\dataset\processed\clean_requests.csv"

print("Loading request workload...")

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = df.sort_values(
    "timestamp"
)

print(
    f"Total requests: {len(df):,}"
)


# ============================================================
# Simulation size
# ============================================================

SIMULATION_SIZE = 100_000

df = df.head(
    SIMULATION_SIZE
)

print(
    f"Simulation requests: {len(df):,}"
)


# ============================================================
# Create LFU cache
# ============================================================

CACHE_CAPACITY = 1000

cache = LFUCache(
    CACHE_CAPACITY
)


# ============================================================
# Run simulation
# ============================================================

print("\nRunning LFU cache simulation...")

for _, row in df.iterrows():

    url = row["url"]

    size = row["response_bytes"]

    # Check cache
    hit = cache.get(url)

    # Cache miss
    if not hit:

        # Retrieve from origin
        cache.put(
            url,
            size
        )


# ============================================================
# Results
# ============================================================

print("\n==========================================")
print("LFU CACHE RESULTS")
print("==========================================")

print(
    f"Cache capacity : {CACHE_CAPACITY}"
)

print(
    f"Cache hits     : {cache.hits:,}"
)

print(
    f"Cache misses   : {cache.misses:,}"
)

print(
    f"Cache hit ratio: "
    f"{cache.hit_ratio() * 100:.2f}%"
)
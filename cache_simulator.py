from collections import OrderedDict
import pandas as pd


class LRUCache:
    """
    Least Recently Used cache.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

        self.hits = 0
        self.misses = 0

    def get(self, key):

        if key in self.cache:

            # Move recently used item to the end
            self.cache.move_to_end(key)

            self.hits += 1

            return True

        self.misses += 1

        return False

    def put(self, key, size):

        # If already present
        if key in self.cache:

            self.cache.move_to_end(key)

            return

        # Remove oldest item if cache is full
        if len(self.cache) >= self.capacity:

            self.cache.popitem(last=False)

        self.cache[key] = size

    def hit_ratio(self):

        total = self.hits + self.misses

        if total == 0:
            return 0

        return self.hits / total


# ============================================================
# Load request workload
# ============================================================

INPUT_FILE = r"D:\smart edge\dataset\processed\clean_requests.csv"

print("Loading request workload...")

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp")

print(f"Total requests: {len(df):,}")


# ============================================================
# Use a manageable sample for simulation
# ============================================================

SIMULATION_SIZE = 100_000

df = df.head(SIMULATION_SIZE)

print(f"Simulation requests: {len(df):,}")


# ============================================================
# Create cache
# ============================================================

CACHE_CAPACITY = 1000

cache = LRUCache(CACHE_CAPACITY)


# ============================================================
# Run simulation
# ============================================================

print("\nRunning LRU cache simulation...")

for _, row in df.iterrows():

    url = row["url"]

    size = row["response_bytes"]

    # Check cache
    hit = cache.get(url)

    # Cache miss
    if not hit:

        # Retrieve from origin server
        cache.put(url, size)


# ============================================================
# Results
# ============================================================

print("\n==========================================")
print("LRU CACHE RESULTS")
print("==========================================")

print(f"Cache capacity : {CACHE_CAPACITY}")

print(f"Cache hits     : {cache.hits:,}")

print(f"Cache misses   : {cache.misses:,}")

print(
    f"Cache hit ratio: "
    f"{cache.hit_ratio() * 100:.2f}%"
)
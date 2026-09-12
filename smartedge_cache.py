import pandas as pd
import joblib
import os


# ============================================================
# 1. Paths
# ============================================================

DATA_FILE = r"D:\smart edge\dataset\processed\final_ml_dataset.csv"

MODEL_FILE = r"D:\smart edge\model\smartedge_random_forest.pkl"


# ============================================================
# 2. Load model
# ============================================================

print("Loading SmartEdge ML model...")

model = joblib.load(MODEL_FILE)
model.n_jobs = 1

print("Model loaded successfully!")

# ============================================================
# 3. Load request workload
# ============================================================

print("\nLoading request workload...")

df = pd.read_csv(DATA_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

print(
    f"Total requests: {len(df):,}"
)


# ============================================================
# 4. Simulation size
# ============================================================

SIMULATION_SIZE = 10000

df = df.head(
    SIMULATION_SIZE
)

print(
    f"Simulation requests: {len(df):,}"
)


# ============================================================
# 5. SmartEdge Cache
# ============================================================

class SmartEdgeCache:

    def __init__(self, capacity):

        self.capacity = capacity

        # URL -> content size
        self.cache = {}

        self.hits = 0
        self.misses = 0

        self.origin_requests = 0

    # --------------------------------------------------------
    # Cache lookup
    # --------------------------------------------------------

    def get(self, url):

        if url in self.cache:

            self.hits += 1

            return True

        self.misses += 1

        return False

    # --------------------------------------------------------
    # Store content
    # --------------------------------------------------------

    def put(self, url, size):

        if url in self.cache:
            return

        # If cache is full
        if len(self.cache) >= self.capacity:

            # Simple eviction:
            # remove first inserted item
            first_item = next(
                iter(self.cache)
            )

            del self.cache[first_item]

        self.cache[url] = size

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    def hit_ratio(self):

        total = self.hits + self.misses

        if total == 0:
            return 0

        return self.hits / total


# ============================================================
# 6. Create cache
# ============================================================

CACHE_CAPACITY = 1000

cache = SmartEdgeCache(
    CACHE_CAPACITY
)


# ============================================================
# 7. ML Features
# ============================================================

FEATURES = [
    "hour",
    "day_of_week",
    "past_frequency",
    "recency_seconds",
    "content_size"
]


# ============================================================
# 8. Run SmartEdge simulation
# ============================================================

print("\nRunning SmartEdge simulation...")

for _, row in df.iterrows():

    url = row["url"]

    size = row["content_size"]

    # --------------------------------------------------------
    # First check cache
    # --------------------------------------------------------

    if cache.get(url):

        # CACHE HIT
        continue

    # --------------------------------------------------------
    # CACHE MISS
    # --------------------------------------------------------

    cache.origin_requests += 1

    # --------------------------------------------------------
    # Prepare ML input
    # --------------------------------------------------------

    X = pd.DataFrame(
        [[
            row["hour"],
            row["day_of_week"],
            row["past_frequency"],
            row["recency_seconds"],
            row["content_size"]
        ]],
        columns=FEATURES
    )

    # --------------------------------------------------------
    # Predict probability
    # --------------------------------------------------------
   
    probability = model.predict_proba(
        X
    )[0][1]
   

    # --------------------------------------------------------
    # Smart caching decision
    # --------------------------------------------------------

    if probability >= 0.5:

        cache.put(
            url,
            size
        )


# ============================================================
# 9. Results
# ============================================================

print("\n==========================================")
print("SMARTEDGE CACHE RESULTS")
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
    f"Origin requests: {cache.origin_requests:,}"
)

print(
    f"Cache hit ratio: "
    f"{cache.hit_ratio() * 100:.2f}%"
)

print("\n==========================================")
print("SMARTEDGE SIMULATION COMPLETE")
print("==========================================")
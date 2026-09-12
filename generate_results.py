import pandas as pd
import matplotlib.pyplot as plt
import os

INPUT_FILE = r"D:\smart edge\cache\cache_comparison.csv"
OUTPUT_DIR = r"D:\smart edge\results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

print("Loading comparison results...")
print(df)

# ============================================================
# 1. Cache Hit Ratio
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    df["Strategy"],
    df["Hit Ratio (%)"]
)

plt.title("Cache Hit Ratio Comparison")
plt.xlabel("Caching Strategy")
plt.ylabel("Hit Ratio (%)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "cache_hit_ratio.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 2. Average Latency
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    df["Strategy"],
    df["Average Latency (ms)"]
)

plt.title("Average Latency Comparison")
plt.xlabel("Caching Strategy")
plt.ylabel("Average Latency (ms)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "latency_comparison.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 3. Bandwidth Saved
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    df["Strategy"],
    df["Bandwidth Saved (%)"]
)

plt.title("Bandwidth Savings Comparison")
plt.xlabel("Caching Strategy")
plt.ylabel("Bandwidth Saved (%)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "bandwidth_savings.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 4. Print best strategy
# ============================================================

best_hit = df.loc[
    df["Hit Ratio (%)"].idxmax()
]

best_latency = df.loc[
    df["Average Latency (ms)"].idxmin()
]

best_bandwidth = df.loc[
    df["Bandwidth Saved (%)"].idxmax()
]

print("\n==========================================")
print("FINAL ANALYSIS")
print("==========================================")

print(
    f"Best Hit Ratio: "
    f"{best_hit['Strategy']} "
    f"({best_hit['Hit Ratio (%)']:.2f}%)"
)

print(
    f"Lowest Latency: "
    f"{best_latency['Strategy']} "
    f"({best_latency['Average Latency (ms)']:.2f} ms)"
)

print(
    f"Best Bandwidth Saving: "
    f"{best_bandwidth['Strategy']} "
    f"({best_bandwidth['Bandwidth Saved (%)']:.2f}%)"
)

print("\nGraphs saved to:")
print(OUTPUT_DIR)
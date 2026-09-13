import runpy

# Load everything from the real-data simulator
env = runpy.run_path(
    r"D:\smart edge\api\real_data_simulator.py"
)

requests = env["requests"]
run_smartedge = env["run_smartedge"]

capacity = 100

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

print("\n")
print("=" * 75)
print("SMARTEDGE THRESHOLD EXPERIMENT")
print("=" * 75)

print(
    f"{'Threshold':<12}"
    f"{'Hit Rate':<15}"
    f"{'Admissions':<15}"
    f"{'Evictions':<15}"
    f"{'Cache Size':<15}"
)

print("-" * 75)

results = []

for threshold in thresholds:

    result = run_smartedge(
        requests,
        capacity,
        threshold
    )

    results.append(
        (
            threshold,
            result
        )
    )

    print(
        f"{threshold:<12.2f}"
        f"{result['hit_ratio'] * 100:<15.2f}"
        f"{result['admissions']:<15}"
        f"{result['evictions']:<15}"
        f"{result['cache_size']:<15}"
    )

print("=" * 75)

best = max(
    results,
    key=lambda x: x[1]["hit_ratio"]
)

print(
    f"\nBEST THRESHOLD: {best[0]:.2f}"
)

print(
    f"BEST HIT RATE : "
    f"{best[1]['hit_ratio'] * 100:.2f}%"
)

print(
    f"ADMISSIONS    : "
    f"{best[1]['admissions']}"
)

print(
    f"EVICTIONS     : "
    f"{best[1]['evictions']}"
)
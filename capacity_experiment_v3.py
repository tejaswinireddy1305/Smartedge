import subprocess
import sys
import re
import csv
from pathlib import Path


# ============================================================
# SMARTEDGE 3.0 - AUTOMATED CAPACITY EXPERIMENT
# ============================================================

PROJECT_DIR = Path(r"D:\smart edge")
SIMULATOR = PROJECT_DIR / "api" / "real_data_simulator_v3.py"

CAPACITIES = [25, 50, 75, 100, 150, 200, 300, 500, 750, 1000]

OUTPUT_DIR = PROJECT_DIR / "cache" / "results"
OUTPUT_FILE = OUTPUT_DIR / "capacity_results_v3.csv"


def run_simulation(capacity):
    print("\n" + "=" * 70)
    print(f"RUNNING CAPACITY = {capacity}")
    print("=" * 70)

    command = [
        sys.executable,
        str(SIMULATOR),
        "--capacity",
        str(capacity)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    output = result.stdout + "\n" + result.stderr

    print(output)

    if result.returncode != 0:
        print(f"ERROR: Simulation failed for capacity {capacity}")
        return None

    return parse_results(output, capacity)


def parse_results(output, capacity):

    def get_number(pattern, default=None):
        match = re.search(pattern, output)

        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return default

        return default

    def get_int(pattern, default=None):
        value = get_number(pattern, default)

        if value is None:
            return default

        return int(value)

    results = {
        "capacity": capacity,

        "lru_hit_rate": get_number(
            r"LRU\s*:\s*([\d.]+)%"
        ),

        "lfu_hit_rate": get_number(
            r"LFU\s*:\s*([\d.]+)%"
        ),

        "smartedge_hit_rate": get_number(
            r"SmartEdge\s*:\s*([\d.]+)%"
        ),

        "lru_evictions": get_int(
            r"LRU\s*:\s*(\d+)"
        ),

        "lfu_evictions": get_int(
            r"LFU\s*:\s*(\d+)"
        ),

        "smartedge_evictions": get_int(
            r"SmartEdge\s*:\s*(\d+)"
        ),

        "admissions": get_int(
            r"Admissions\s*:\s*(\d+)"
        ),

        "rejected": get_int(
            r"Rejected\s*:\s*(\d+)"
        ),

        "cache_size": get_int(
            r"Cache size\s*:\s*(\d+)"
        ),

        "avg_probability": get_number(
            r"Average reusable probability:\s*([\d.]+)"
        ),

        "avg_cache_value": get_number(
            r"Average expected cache value:\s*([\d.]+)"
        ),
    }

    # --------------------------------------------------------
    # Calculate improvements
    # --------------------------------------------------------

    smart = results["smartedge_hit_rate"]
    lru = results["lru_hit_rate"]
    lfu = results["lfu_hit_rate"]

    if smart is not None and lru is not None:
        results["vs_lru"] = round(smart - lru, 2)
    else:
        results["vs_lru"] = None

    if smart is not None and lfu is not None:
        results["vs_lfu"] = round(smart - lfu, 2)
    else:
        results["vs_lfu"] = None

    return results


def save_results(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not results:
        return

    fieldnames = list(results[0].keys())

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to:")
    print(OUTPUT_FILE)


def print_summary(results):

    print("\n")
    print("=" * 100)
    print("SMARTEDGE 3.0 - CAPACITY EXPERIMENT SUMMARY")
    print("=" * 100)

    print(
        f"{'Capacity':>10}"
        f"{'LRU':>12}"
        f"{'LFU':>12}"
        f"{'SmartEdge':>14}"
        f"{'vs LRU':>12}"
        f"{'vs LFU':>12}"
    )

    print("-" * 100)

    for r in results:

        print(
            f"{r['capacity']:>10}"
            f"{r['lru_hit_rate']:>11.2f}%"
            f"{r['lfu_hit_rate']:>11.2f}%"
            f"{r['smartedge_hit_rate']:>13.2f}%"
            f"{r['vs_lru']:>11.2f}"
            f"{r['vs_lfu']:>11.2f}"
        )

    print("=" * 100)

    # --------------------------------------------------------
    # Find best SmartEdge result
    # --------------------------------------------------------

    valid = [
        r for r in results
        if r["smartedge_hit_rate"] is not None
    ]

    if not valid:
        return

    best = max(
        valid,
        key=lambda x: x["smartedge_hit_rate"]
    )

    best_vs_lru = max(
        valid,
        key=lambda x: x["vs_lru"]
    )

    best_vs_lfu = max(
        valid,
        key=lambda x: x["vs_lfu"]
    )

    print("\nBEST SMARTEDGE HIT RATE")
    print(
        f"Capacity : {best['capacity']}"
    )
    print(
        f"Hit Rate : {best['smartedge_hit_rate']:.2f}%"
    )

    print("\nBEST IMPROVEMENT OVER LRU")
    print(
        f"Capacity : {best_vs_lru['capacity']}"
    )
    print(
        f"Improvement : "
        f"{best_vs_lru['vs_lru']:+.2f} percentage points"
    )

    print("\nBEST IMPROVEMENT OVER LFU")
    print(
        f"Capacity : {best_vs_lfu['capacity']}"
    )
    print(
        f"Improvement : "
        f"{best_vs_lfu['vs_lfu']:+.2f} percentage points"
    )

    print("\n")


def main():

    print("=" * 70)
    print("SMARTEDGE 3.0")
    print("AUTOMATED CACHE CAPACITY EXPERIMENT")
    print("=" * 70)

    print("\nCapacities to test:")
    print(CAPACITIES)

    if not SIMULATOR.exists():

        print("\nERROR:")
        print("Simulator not found:")
        print(SIMULATOR)
        return

    all_results = []

    for capacity in CAPACITIES:

        result = run_simulation(capacity)

        if result is not None:
            all_results.append(result)

    if not all_results:

        print("\nNo successful experiments.")
        return

    print_summary(all_results)

    save_results(all_results)

    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
import re
import csv
import os
from datetime import datetime

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

INPUT_FILE = r"D:\smart edge\dataset\access.log\access.log"
OUTPUT_DIR = r"D:\smart edge\dataset\processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "clean_requests.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. NASA log pattern
# --------------------------------------------------

log_pattern = re.compile(
    r'^(?P<client>\S+)'
    r'\s+-\s+-\s+'
    r'\[(?P<timestamp>[^\]]+)\]\s+'
    r'"(?P<method>\S+)\s+(?P<url>\S+)\s+(?P<protocol>[^"]+)"\s+'
    r'(?P<status>\d{3})\s+'
    r'(?P<bytes>\d+|-)$'
)


# --------------------------------------------------
# 3. Process the large file
# --------------------------------------------------

total_lines = 0
valid_lines = 0
invalid_lines = 0

with open(INPUT_FILE, "r", encoding="latin-1") as infile, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:

    writer = csv.writer(outfile)

    # Output column names
    writer.writerow([
        "timestamp",
        "client",
        "method",
        "url",
        "protocol",
        "status_code",
        "response_bytes"
    ])

    for line in infile:

        total_lines += 1

        line = line.strip()

        if not line:
            continue

        match = log_pattern.match(line)

        if not match:
            invalid_lines += 1
            continue

        data = match.groupdict()

        # Convert timestamp
        try:
            timestamp = datetime.strptime(
                data["timestamp"],
                "%d/%b/%Y:%H:%M:%S %z"
            )
        except ValueError:
            invalid_lines += 1
            continue

        # Convert response size
        if data["bytes"] == "-":
            response_bytes = 0
        else:
            response_bytes = int(data["bytes"])

        # Write cleaned row
        writer.writerow([
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            data["client"],
            data["method"],
            data["url"],
            data["protocol"],
            int(data["status"]),
            response_bytes
        ])

        valid_lines += 1

        # Progress every 100,000 lines
        if total_lines % 100000 == 0:
            print(
                f"Processed: {total_lines:,} lines | "
                f"Valid: {valid_lines:,}"
            )


# --------------------------------------------------
# 4. Summary
# --------------------------------------------------

print("\n====================================")
print("PREPROCESSING COMPLETED")
print("====================================")

print(f"Total lines     : {total_lines:,}")
print(f"Valid requests  : {valid_lines:,}")
print(f"Invalid lines   : {invalid_lines:,}")

print("\nOutput file:")
print(OUTPUT_FILE)
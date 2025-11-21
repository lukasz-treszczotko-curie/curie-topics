import csv
import json
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Read the CSV file
csv_path = script_dir / "arxiv_categories.csv"
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Skip the second row which is just documentation
# (it has "Category Name" as the code)
categories = [row for row in rows if row["Code"] != "Category Name"]

# Write to JSON file
json_path = script_dir / "arxiv_categories.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(categories, f, indent=2, ensure_ascii=False)

print(f"Converted {len(categories)} categories to JSON")

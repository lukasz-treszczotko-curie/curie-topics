import requests
import time
import json

BASE_URL = "https://api.openalex.org/topics"
PER_PAGE = 200  # maximum allowed
CURSOR = "*"
EMAIL = "your-email@example.com"  # optional, for polite use; not strictly required

all_topics = []
while True:
    params = {"per-page": PER_PAGE, "cursor": CURSOR, "mailto": EMAIL}
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    # Add new results
    all_topics.extend(data.get("results", []))
    print(f"Fetched {len(all_topics)} topics...")

    # Get next cursor for paging
    CURSOR = data["meta"].get("next_cursor")
    if not CURSOR:
        break
    time.sleep(0.2)  # be nice to the API, avoid hammering

# Save as JSON
with open("openalex_topics.json", "w", encoding="utf-8") as f:
    json.dump(all_topics, f, ensure_ascii=False, indent=2)

print(f"Saved {len(all_topics)} topics to openalex_topics.json")

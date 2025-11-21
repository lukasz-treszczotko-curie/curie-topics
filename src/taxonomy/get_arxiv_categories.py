import requests
from bs4 import BeautifulSoup
import csv

url = "https://arxiv.org/category_taxonomy"
response = requests.get(url)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

rows = []
# Find all h4 tags which contain category codes and names
for h4 in soup.find_all("h4"):
    text = h4.get_text(strip=True)

    # Format is: "code (name)" where name is in <em> tag
    # Example: "cs.AI (Artificial Intelligence)"
    if "(" in text and ")" in text:
        # Split on the first opening parenthesis
        code_part, name_part = text.split("(", 1)
        code = code_part.strip()
        name = name_part.rstrip(")").strip()

        # Get the description from the next <p> tag
        desc = ""
        next_p = h4.find_next("p")
        if next_p:
            desc = next_p.get_text(strip=True)

        rows.append([code, name, desc])

with open("arxiv_categories.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Code", "Name", "Description"])
    for row in rows:
        writer.writerow(row)

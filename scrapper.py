from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
import json5
import json
import random

options = Options()
options.add_argument("--start-maximized")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

BASE_URL = "https://www.olx.in/bicycles_c1415"
all_elements = {}  # merged dict of listing_id -> listing data, across all pages
TARGET_LISTINGS = 400
MAX_PAGES = 30  # safety cap, adjust based on ~40 listings/page

page = 1
while len(all_elements) < TARGET_LISTINGS and page <= MAX_PAGES:
    url = f"{BASE_URL}?page={page}"
    print(f"\nScraping page {page}... (have {len(all_elements)} listings so far)")

    try:
        driver.get(url)
        time.sleep(random.uniform(3, 5))  # randomized delay, gentler on rate limits

        html = driver.page_source
        match = re.search(r'window\.__APP\s*=\s*(\{.*?\});\s*</script>', html, re.DOTALL)

        if not match:
            print(f"No data found on page {page} — might be the last page, stopping.")
            break

        app_data = json5.loads(match.group(1))
        elements = app_data.get("states", {}).get("items", {}).get("elements", {})

        if not elements:
            print(f"Page {page} returned no listings — stopping.")
            break

        all_elements.update(elements)  # merge new listings into the growing dict
        print(f"Page {page}: found {len(elements)} listings")

        # Save progress after EVERY page — so a crash mid-run doesn't lose everything
        with open("all_listings_raw.json", "w", encoding="utf-8") as f:
            json.dump(all_elements, f, indent=2, ensure_ascii=False)

        page += 1
        time.sleep(random.uniform(2, 4))  # polite delay before next page

    except Exception as e:
        print(f"Error on page {page}: {e}")
        break

driver.quit()
print(f"\nDone. Total unique listings collected: {len(all_elements)}")
# Resale Price Fairness Checker

A tool that estimates a fair resale price range for used bicycles, built to solve a real problem: sellers and buyers on campus/local resale groups have no reliable way to know if a price is fair.

## Live Demo
- Frontend: https://resale-price-checker.netlify.app/
- API: https://resale-price-checker.onrender.com

## What It Does
Given a bicycle's brand, city, condition, and photo count, the tool returns a fair price estimate along with an 80% confidence range — not just a single number — based on real listing data scraped from OLX.

## How It Works
1. **Data collection**: Scraped 400 real bicycle listings from OLX (Selenium + embedded JSON extraction, since the site blocks standard HTTP requests)
2. **Cleaning & feature engineering**: Handled messy free-text descriptions (age, condition, gear count) via regex-based extraction with missing-value indicators; filtered dealer/shop listings
3. **Modeling**: Gradient Boosting with quantile regression (10th/50th/90th percentile) to produce a calibrated price range instead of a point estimate
4. **Results**: 14.9% lower MAE than a brand-average baseline, with 79.5% of true prices falling within the predicted 80% confidence range
5. **Serving**: Flask REST API (`/predict`, `/comparables`) serving the trained model
6. **Frontend**: Simple HTML/JS interface calling the API

## Tech Stack
- **Scraping**: Selenium, BeautifulSoup, json5
- **Data processing**: pandas, regex
- **Modeling**: scikit-learn (GradientBoostingRegressor, quantile loss)
- **Backend**: Flask, flask-cors
- **Frontend**: HTML/CSS/JavaScript
- **Deployment**: Render (API), Netlify (frontend)

## Key Findings
- Brand is the strongest predictive signal — prediction error nearly doubled (₹4,791 vs ₹2,590 MAE) for listings in OLX's generic "Other Brands" bucket vs. listings with an identified brand
- Only 21% of listings had extractable bicycle age from free text (most mentions of "age" referred to the intended rider, not the bike) — handled via context-aware regex filtering and missing-value indicators rather than dropping incomplete rows

## Running Locally
\`\`\`bash
pip install -r requirements.txt
python app.py
\`\`\`
Then open \`index.html\` in a browser.

## Limitations & Next Steps
- Dataset limited to ~400 listings from one city cluster; scaling to 1000+ would improve reliability for less common brand/city combinations
- Brand extraction for the "Other Brands" bucket (51% of listings) could be improved via free-text keyword matching

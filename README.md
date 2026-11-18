# ML Customer Segmentation System

A web application that turns a CSV/XLSX file of customer transactions into customer segments. It covers the full workflow: user accounts, upload and data cleaning, RFM analysis, K-Means clustering, and charts.

## Features

- **User accounts** — registration, login/logout, salted password hashes, CSRF protection. Each user can only see their own datasets.
- **Upload and cleaning** — accepts `.csv` and `.xlsx` files up to 10 MB. Validates the structure, renames known column aliases, then removes or reports bad rows (missing IDs, unreadable or ambiguous dates, invalid amounts, duplicates, future dates). The results page shows exactly what was changed and why.
- **RFM analysis** — calculates Recency (days since last purchase), Frequency (number of transactions) and Monetary (total spent), plus 1–5 scores. The reference date defaults to the last transaction in the uploaded file, not today's date.
- **K-Means clustering** — groups customers by their RFM values. Tries k = 2–10 and shows inertia (elbow method) and silhouette scores. Features are log-transformed and standardized before clustering.
- **Results and charts** — dashboard, cluster profiles and visualisations rendered in the browser with Chart.js.

## Dataset format

Required columns (case and `space`/`_` /`-` spelling variants are accepted, e.g. `Customer ID`, `sales_amount`):

| Column | Meaning |
|---|---|
| `Customer_ID` | Unique customer code (several rows can share it) |
| `Transaction_Date` | Date of the purchase |
| `Transaction_Amount` | Amount spent (must be positive) |

Optional columns: `Customer_Name`, `Product`, `Category`, `Location`.

Accepted date examples: `2025-01-15`, `15/01/2025`, `15-Jan-2025`, `31st March 2025`, `15-Jan-25`, `20250115`, Excel serial numbers such as `45658`. Ambiguous dates like `05/03/2025` are kept only if the file itself proves the day/month order; otherwise the row is removed. Accepted amount examples: `1500`, `25,000`, `MK 25,000`, `$1,250.50`.

## Project structure

```text
app.py                 Flask application factory
config.py              Settings (database, uploads, SECRET_KEY from .env)
models/models.py       SQLAlchemy tables: users, datasets, customers, transactions, analysis runs, segments, RFM results
routes/                auth.py, dashboard.py, upload.py, rfm.py, clustering.py
services/              data_cleaning.py, rfm_analysis.py, clustering.py, behavioral_analytics.py
templates/             Jinja pages (base.html is the shared layout)
static/                style.css and script.js
tests/                 pytest suite
```

## Setup

Requires Python 3.10 or newer.

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Clustering also needs scikit-learn:

```powershell
pip install scikit-learn
```

Create a `.env` file in the project root with a real secret key (the file is gitignored):

```text
SECRET_KEY=replace-this-with-a-long-random-string
```

## Run

```powershell
python app.py
```

Open <http://127.0.0.1:5000>, register an account, upload a dataset, then run RFM analysis and clustering from the menu. The SQLite database (`database/app.db`) and upload folder (`uploads/`) are created automatically.

## Tests

```powershell
python -m pytest -q
```

## Notes

- Bootstrap 5.3.3 and Chart.js 4.4.3 are loaded from jsDelivr, so an internet connection is needed for styling and charts.
- Do not commit `database/app.db`, `uploads/`, `.env`, or `__pycache__/` — they are ignored in `.gitignore`.
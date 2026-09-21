# E-Commerce Revenue Root Cause & Forecasting System

A portfolio project that combines SQL analytics, machine learning forecasting, anomaly detection, and explainable root-cause analysis.

The application has two working modes:

1. **Built-in Demo Dataset** — Maven Fuzzy Factory e-commerce data for revenue forecasting, marketing analysis, product analysis, and revenue root-cause diagnostics.
2. **Upload Your CSV** — automatically detects supported product-month demand data and trains forecasting models on the uploaded file. Generic Date + Revenue CSV files are also supported through column mapping.

## Main Features

### Built-in e-commerce analysis

- Executive KPIs: revenue, gross profit, sessions, orders, conversion rate, AOV
- Weekly revenue forecasting with a time-based train/test split
- Revenue anomaly detection using a rolling baseline and z-score
- Revenue-change decomposition into traffic, conversion rate, AOV, and refunds
- Marketing-source, device, and product contribution analysis
- Optional OpenAI summary generated only from calculated evidence

### Product monthly demand upload mode

The app automatically detects this structure:

```text
StockCode
Description
Month
Quantity
Revenue
Transactions
Customers
Avg_Price
```

It then provides:

- Next-month **Quantity / Demand** forecasting
- Next-month **Revenue** forecasting
- Product selector for SKU-level analysis
- Time-based model validation
- Last-month benchmark vs ML models
- Product-level and panel-level WMAPE
- Product anomaly detection
- Demand-change decomposition
- Revenue-change decomposition
- Portfolio forecast table and CSV export
- Optional OpenAI diagnostic explanation

## Forecasting Methodology

For product-month data, the application first creates a complete calendar-month series for every product after its first appearance. Missing months are treated as zero demand so lag features represent actual calendar time.

Features include:

- Current quantity, revenue, transactions, customers, and price
- 1, 2, 3, and 12-month lags
- 3, 6, and 12-month rolling averages
- Units per transaction
- Transactions per customer
- Revenue per customer
- Price change
- Month seasonality
- Product age

Models compared:

- Last-Month Baseline
- Linear Regression
- Histogram Gradient Boosting

The ML model is selected using a **time-based holdout**, not a random split. The latest months are reserved for testing.

## Root-Cause Logic for Product Data

Product revenue is explained using the measurable relationship:

```text
Revenue = Transactions × Units per Transaction × Revenue per Unit
```

This allows the app to separate revenue change into:

- Transaction-volume impact
- Units-per-transaction impact
- Revenue-per-unit impact

Customer change is shown as supporting context.

## Project Structure

```text
app.py
requirements.txt
run_app.ps1
.env.example

src/
  analytics.py
  anomaly.py
  forecasting.py
  root_cause.py
  ai_summary.py
  upload_analysis.py
  product_demand.py
  database.py

sql/
  create_tables.sql
  create_indexes.sql
  kpi_queries.sql

scripts/
  01_build_database.py
  02_build_analytics_datasets.py
  03_train_revenue_model.py
  04_validate_project.py

data/
  raw/
  processed/
  models/
  database/

sample_data/
  product_monthly_demand.csv

INTERVIEW_PREP.md
RESUME_POINTS.md
```

## Run the Project on Windows

Open PowerShell in the project folder. `app.py` should be visible in that folder.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Or run:

```powershell
.\run_app.ps1
```

## OpenAI Setup — Optional

The core forecasting and root-cause calculations do **not** require OpenAI.

To enable AI-written diagnostic explanations, copy `.env.example` to `.env` and add your own API key:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5.6-luna
```

Never commit `.env` to GitHub.

## Example Product CSV

A sample file is included at:

```text
sample_data/product_monthly_demand.csv
```

Choose **Upload Your CSV** in the sidebar and upload this file. The app should automatically show **Detected dataset type: Product Monthly Demand**.

## Important Limitations

- Product demand is intermittent, so forecasting accuracy varies across SKUs.
- Product-specific validation metrics should be considered together with panel-level metrics.
- Forecasts are decision-support estimates, not guaranteed future outcomes.
- AI-generated explanations do not calculate the numbers; they only summarize already-calculated evidence.

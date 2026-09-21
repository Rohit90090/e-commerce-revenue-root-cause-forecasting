# Interview Prep — E-Commerce Revenue Root Cause & Forecasting System

## 30-Second Explanation

I built an e-commerce forecasting and root-cause analytics application using Python, SQL, Pandas, scikit-learn, Plotly, and Streamlit. The project can use a built-in relational e-commerce dataset or accept a product-level monthly CSV. It forecasts future revenue or demand, detects anomalies, and explains which measurable business drivers contributed to a change. I used time-based validation instead of a random train-test split because this is a forecasting problem.

## 60-Second Explanation

The project has two analysis modes. The built-in Maven Fuzzy Factory mode uses a SQLite database and SQL queries for business KPIs, weekly revenue forecasting, anomaly detection, marketing analysis, and revenue root-cause decomposition. The second mode accepts a product-month CSV and automatically detects fields such as StockCode, Month, Quantity, Revenue, Transactions, Customers, and Avg Price. It builds lag and rolling features, compares a baseline with Linear Regression and Histogram Gradient Boosting, and predicts the next month for each product. For root-cause analysis, I decompose revenue into transaction volume, units per transaction, and revenue per unit. OpenAI is optional and only converts calculated evidence into a short business explanation.

## Architecture

```text
CSV / SQLite
    ↓
Data preparation
    ↓
SQL + Pandas KPI layer
    ↓
Feature engineering
    ↓
Time-based model validation
    ↓
Forecast
    ↓
Anomaly detection
    ↓
Root-cause decomposition
    ↓
Streamlit dashboard
```

## Common Questions

### Why did you use a time-based split?

Random splitting would allow later observations to appear in training while earlier observations appear in testing. That can create leakage in forecasting. I reserve the latest time periods for testing so the evaluation is closer to a real future forecast.

### Why include a simple baseline?

A machine-learning model should improve on a simple approach. I compare the ML models with a last-month baseline so I can verify whether the extra model complexity adds value.

### Why use WMAPE for product demand?

Product demand contains low-volume and zero-demand months. Regular MAPE becomes unstable when actual values are close to zero. WMAPE gives a more practical portfolio-level percentage error.

### Which models do you compare?

For uploaded product-month data I compare Linear Regression and Histogram Gradient Boosting, plus a last-month baseline. The selected ML model is based on out-of-sample performance.

### What features are used?

Current business measures, multiple lag values, rolling averages, month seasonality, price change, units per transaction, transactions per customer, and product age.

### How does the root-cause analysis work?

For product-level analysis I use the identity Revenue = Transactions × Units per Transaction × Revenue per Unit. A sequential decomposition estimates how much of the revenue change came from each component. This is based on calculated business metrics rather than an LLM guessing the cause.

### What does OpenAI do?

OpenAI is optional. It does not calculate forecasts or root causes. Python and SQL calculate the evidence, and the model can turn that evidence into a concise business explanation.

### What happens when a user uploads a different CSV?

If the file matches the product-month schema, the advanced product forecasting workflow is automatically enabled. Otherwise, the generic upload workflow lets the user map Date, Revenue, and optional business columns.

### What is the biggest limitation?

Some products have intermittent demand and limited history, so SKU-level forecasts can be volatile. I show both product-specific and panel-level validation metrics instead of presenting a forecast without context.

## Live Demo Order

1. Show Executive Overview in the built-in dataset.
2. Show the built-in revenue forecast and anomaly chart.
3. Show revenue root-cause decomposition.
4. Switch to Upload Your CSV.
5. Upload `sample_data/product_monthly_demand.csv`.
6. Select a product and forecast target.
7. Explain the time-based model comparison.
8. Show product anomalies.
9. Show transaction / basket-size / unit-value root cause.
10. Download the product forecast CSV.

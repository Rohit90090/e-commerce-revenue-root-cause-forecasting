from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.analytics import (
    executive_kpis,
    monthly_kpis,
    marketing_performance,
    product_performance,
)
from src.forecasting import (
    model_metadata,
    test_predictions,
    next_week_forecast,
)
from src.anomaly import detect_weekly_anomalies
from src.root_cause import compare_months
from src.ai_summary import ai_summary

from src.upload_analysis import (
    UploadConfig,
    compare_months as compare_uploaded_months,
    detect_anomalies as detect_uploaded_anomalies,
    executive_kpis as uploaded_executive_kpis,
    monthly_kpis as uploaded_monthly_kpis,
    prepare_uploaded_data,
    train_forecast,
    weekly_kpis as uploaded_weekly_kpis,
)

from src.product_demand import (
    is_product_monthly_dataset,
    prepare_product_monthly,
    portfolio_monthly,
    eligible_products,
    train_product_forecast,
    product_history,
    product_anomalies,
    product_root_cause,
    local_product_summary,
    product_ai_summary,
)

load_dotenv()

st.set_page_config(
    page_title="E-Commerce Revenue Root Cause & Forecasting System",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        max-width: 1500px;
    }

    .hero {
        padding: 22px 24px;
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 18px;
        margin-bottom: 18px;
    }

    .hero h1 {
        margin: 0 0 4px 0;
        font-size: 2rem;
    }

    .hero p {
        margin: 0;
        opacity: .72;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.18);
        border-radius: 14px;
        padding: 12px;
    }

    .note {
        padding: 10px 12px;
        border-left: 4px solid #64748b;
        background: rgba(100,116,139,.08);
        border-radius: 7px;
        margin: 6px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>E-Commerce Revenue Root Cause & Forecasting System</h1>
      <p>
        Forecast revenue, detect unusual performance,
        and diagnose the business drivers behind change.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


def fmt_money(value):
    return "—" if pd.isna(value) else f"${float(value):,.0f}"


def fmt_compact_money(value):
    if pd.isna(value):
        return "—"

    value = float(value)
    absolute = abs(value)

    if absolute >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"

    if absolute >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"

    if absolute >= 1_000:
        return f"${value / 1_000:.1f}K"

    return f"${value:,.0f}"


def safe_markdown_text(text):
    return str(text).replace("$", r"\$")


def fmt_num(value):
    return "—" if pd.isna(value) else f"{int(round(float(value))):,}"


def fmt_pct(value):
    return "—" if pd.isna(value) else f"{float(value):.2f}%"


def pretty_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    rename = {
        "week": "Week",
        "net_revenue": "Net Revenue",
        "baseline": "Baseline",
        "deviation_pct": "Deviation %",
        "z_score": "Z-Score",
        "severity": "Severity",
        "driver": "Driver",
        "impact": "Impact",
        "share_abs_pct": "Contribution Share %",
        "dimension": "Dimension",
        "previous": "Previous",
        "current": "Current",
        "change": "Change",
        "change_pct": "Change %",
        "source": "Source",
        "sessions": "Sessions",
        "orders": "Orders",
        "gross_revenue": "Gross Revenue",
        "conversion_rate": "Conversion Rate %",
        "revenue_per_session": "Revenue / Session",
        "product_id": "Product ID",
        "product_name": "Product",
        "units": "Units",
        "gross_profit": "Gross Profit",
        "refunded_items": "Refunded Items",
        "refund_amount": "Refund Amount",
        "refund_rate": "Refund Rate %",
    }

    return out.rename(columns=rename)


def guess_column(columns, candidates):
    low = {
        str(c).lower().replace(" ", "").replace("_", ""): c
        for c in columns
    }

    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("_", "")
        if key in low:
            return low[key]

    for c in columns:
        compact = str(c).lower().replace(" ", "").replace("_", "")

        if any(
            candidate.lower().replace(" ", "").replace("_", "") in compact
            for candidate in candidates
        ):
            return c

    return None


def optional_select(label, columns, default=None, key=None):
    options = ["Not available"] + list(columns)

    index = options.index(default) if default in options else 0

    value = st.selectbox(
        label,
        options,
        index=index,
        key=key,
    )

    return None if value == "Not available" else value


@st.cache_data(show_spinner=False)
def cached_executive_kpis():
    return executive_kpis()


@st.cache_data(show_spinner=False)
def cached_monthly_kpis():
    return monthly_kpis()


@st.cache_data(show_spinner=False)
def cached_model_metadata():
    return model_metadata()


@st.cache_data(show_spinner=False)
def cached_test_predictions():
    return test_predictions()


@st.cache_data(show_spinner=False)
def cached_next_week_forecast():
    return next_week_forecast()


@st.cache_data(show_spinner=False)
def cached_anomalies():
    return detect_weekly_anomalies(
        threshold=2.5
    )


@st.cache_data(show_spinner=False)
def cached_root_cause(month):
    return compare_months(month)


@st.cache_data(show_spinner=False)
def cached_marketing():
    return marketing_performance()


@st.cache_data(show_spinner=False)
def cached_products():
    return product_performance()


with st.sidebar:

    st.header("Project Controls")

    dataset_mode = st.radio(
        "Dataset Mode",
        [
            "Built-in Demo Dataset",
            "Upload Your CSV",
        ],
        help=(
            "Use the Maven Fuzzy Factory demo "
            "or train the analysis on your own CSV."
        ),
    )

    ai_status = (
        "OpenAI enabled"
        if os.getenv("OPENAI_API_KEY")
        else "Local diagnostic summary"
    )

    st.caption(
        f"Summary mode: {ai_status}"
    )

    if dataset_mode == "Built-in Demo Dataset":

        st.caption(
            "Dataset: Maven Fuzzy Factory"
        )

        st.caption(
            "Core calculations: SQL + Pandas"
        )

        st.caption(
            "Forecasting: scikit-learn"
        )

    else:

        st.caption(
            "Dataset: User uploaded CSV"
        )

        st.caption(
            "Forecast model retrains on the uploaded data"
        )

        st.caption(
            "Product-month datasets are auto-detected; "
            "generic CSVs need Date + Revenue"
        )


def render_builtin():

    k = cached_executive_kpis()

    monthly = cached_monthly_kpis()

    (
        overview_tab,
        forecast_tab,
        root_tab,
        drivers_tab,
        summary_tab,
    ) = st.tabs(
        [
            "Executive Overview",
            "Forecast & Anomalies",
            "Root Cause Analysis",
            "Marketing & Product Drivers",
            "Diagnostic Summary",
        ]
    )

    with overview_tab:

        c1, c2, c3, c4, c5, c6 = st.columns(6)

        c1.metric(
            "Net Revenue",
            f"${k['net_revenue']/1_000_000:.2f}M",
        )

        c2.metric(
            "Gross Profit",
            f"${k['gross_profit']/1_000_000:.2f}M",
        )

        c3.metric(
            "Sessions",
            f"{k['sessions']:,}",
        )

        c4.metric(
            "Orders",
            f"{k['orders']:,}",
        )

        c5.metric(
            "Conversion Rate",
            f"{k['conversion_rate']:.2f}%",
        )

        c6.metric(
            "Avg Order Value",
            f"${k['aov']:.2f}",
        )

        left, right = st.columns(
            [1.4, 1]
        )

        with left:

            fig = px.line(
                monthly,
                x="month",
                y="net_revenue",
                markers=True,
                title="Monthly Net Revenue",
            )

            fig.update_layout(
                xaxis_title="",
                yaxis_title="Net Revenue ($)",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with right:

            fig = px.line(
                monthly,
                x="month",
                y="conversion_rate",
                markers=True,
                title="Monthly Conversion Rate",
            )

            fig.update_layout(
                xaxis_title="",
                yaxis_title="Conversion Rate (%)",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        c1, c2 = st.columns(2)

        with c1:

            fig = px.line(
                monthly,
                x="month",
                y="orders",
                markers=True,
                title="Monthly Orders",
            )

            fig.update_layout(
                xaxis_title="",
                yaxis_title="Orders",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with c2:

            fig = px.line(
                monthly,
                x="month",
                y="refund_amount",
                markers=True,
                title="Monthly Refund Amount",
            )

            fig.update_layout(
                xaxis_title="",
                yaxis_title="Refunds ($)",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    with forecast_tab:

        meta = cached_model_metadata()

        pred = cached_test_predictions()

        nxt = cached_next_week_forecast()

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Selected Model",
            meta["best_model"],
        )

        m2.metric(
            "Test MAE",
            f"${meta['best_model_mae']:,.0f}",
        )

        m3.metric(
            "Test MAPE",
            f"{meta['best_model_mape']:.1f}%",
        )

        m4.metric(
            "Next Week Forecast",
            f"${nxt['prediction']:,.0f}",
        )

        st.caption(
            f"Forecast range based on one test-set MAE: "
            f"${nxt['lower']:,.0f} to ${nxt['upper']:,.0f}. "
            "This is a practical uncertainty band, "
            "not a formal prediction interval."
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=pred["week"],
                y=pred["net_revenue"],
                mode="lines+markers",
                name="Actual",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=pred["week"],
                y=pred["predicted_revenue"],
                mode="lines+markers",
                name="Predicted",
            )
        )

        fig.update_layout(
            title=(
                "Out-of-Sample Weekly Revenue: "
                "Actual vs Predicted"
            ),
            xaxis_title="",
            yaxis_title="Net Revenue ($)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        anomalies = cached_anomalies()

        recent = (
            anomalies
            .sort_values("week")
            .tail(35)
        )

        fig = px.line(
            recent,
            x="week",
            y="net_revenue",
            markers=True,
            title=(
                "Recent Weekly Revenue "
                "with Anomaly Baseline"
            ),
        )

        fig.add_scatter(
            x=recent["week"],
            y=recent["baseline"],
            mode="lines",
            name="8-week baseline",
        )

        anomaly_rows = recent[
            recent["is_anomaly"]
        ]

        if not anomaly_rows.empty:

            fig.add_scatter(
                x=anomaly_rows["week"],
                y=anomaly_rows["net_revenue"],
                mode="markers",
                name="Anomaly",
                marker=dict(
                    size=12,
                    symbol="diamond",
                ),
            )

        fig.update_layout(
            xaxis_title="",
            yaxis_title="Net Revenue ($)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.write(
            "**Most recent detected anomalies**"
        )

        show = (
            anomalies[
                anomalies["is_anomaly"]
            ][
                [
                    "week",
                    "net_revenue",
                    "baseline",
                    "deviation_pct",
                    "z_score",
                    "severity",
                ]
            ]
            .sort_values(
                "week",
                ascending=False,
            )
            .head(10)
            .copy()
        )

        show["week"] = (
            pd.to_datetime(show["week"])
            .dt.strftime("%d %b %Y")
        )

        show["net_revenue"] = (
            show["net_revenue"]
            .map(lambda x: f"${x:,.0f}")
        )

        show["baseline"] = (
            show["baseline"]
            .map(lambda x: f"${x:,.0f}")
        )

        show["deviation_pct"] = (
            show["deviation_pct"]
            .map(lambda x: f"{x:.1f}%")
        )

        show["z_score"] = (
            show["z_score"]
            .map(lambda x: f"{x:.2f}")
        )

        st.dataframe(
            pretty_table(show),
            use_container_width=True,
            hide_index=True,
        )

    with root_tab:

        month_keys = (
            monthly["month"]
            .dt.strftime("%Y-%m")
            .tolist()[1:]
        )

        selected_month = st.selectbox(
            "Month to diagnose",
            month_keys,
            index=max(
                len(month_keys) - 1,
                0,
            ),
        )

        result = cached_root_cause(
            selected_month
        )

        current = result["current"]
        previous = result["previous"]

        d1, d2, d3, d4 = st.columns(4)

        d1.metric(
            "Net Revenue Change",
            f"${result['net_revenue_change']:,.0f}",
            f"{result['net_revenue_change_pct']:.1f}%",
        )

        d2.metric(
            "Sessions",
            f"{int(current['sessions']):,}",
            (
                f"{int(current['sessions'] - previous['sessions']):+,}"
            ),
        )

        d3.metric(
            "Conversion Rate",
            f"{current['conversion_rate']:.2f}%",
            (
                f"{current['conversion_rate'] - previous['conversion_rate']:+.2f} pp"
            ),
        )

        aov_delta = float(
            current["aov"] - previous["aov"]
        )

        d4.metric(
            "AOV",
            f"${current['aov']:.2f}",
            f"{aov_delta:+.2f} USD",
        )

        dec = (
            result["decomposition"]
            .sort_values("impact")
        )

        fig = px.bar(
            dec,
            x="impact",
            y="driver",
            orientation="h",
            title="Revenue Change Decomposition",
        )

        fig.update_layout(
            xaxis_title="Estimated Impact ($)",
            yaxis_title="",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.caption(
            "Traffic, conversion rate, and AOV impacts "
            "use an exact sequential decomposition of gross revenue. "
            "Refund impact adjusts the result from gross to net revenue."
        )

        c1, c2 = st.columns(2)

        with c1:

            src = (
                result["source_drivers"]
                .sort_values("change")
                .head(8)
            )

            fig = px.bar(
                src,
                x="change",
                y="dimension",
                orientation="h",
                title=(
                    "Source Contribution "
                    "to Revenue Change"
                ),
            )

            fig.update_layout(
                xaxis_title="Net Revenue Change ($)",
                yaxis_title="",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with c2:

            prod = (
                result["product_drivers"]
                .sort_values("change")
                .head(8)
            )

            fig = px.bar(
                prod,
                x="change",
                y="dimension",
                orientation="h",
                title=(
                    "Product Contribution "
                    "to Revenue Change"
                ),
            )

            fig.update_layout(
                xaxis_title="Net Revenue Change ($)",
                yaxis_title="",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    with drivers_tab:

        st.subheader(
            "Marketing Drivers"
        )

        marketing = cached_marketing()

        source = (
            marketing
            .groupby(
                "source",
                as_index=False,
            )
            .agg(
                sessions=("sessions", "sum"),
                orders=("orders", "sum"),
                gross_revenue=(
                    "gross_revenue",
                    "sum",
                ),
            )
        )

        source["conversion_rate"] = (
            source["orders"]
            / source["sessions"]
            * 100
        )

        source["revenue_per_session"] = (
            source["gross_revenue"]
            / source["sessions"]
        )

        source = source.sort_values(
            "gross_revenue",
            ascending=False,
        )

        st.dataframe(
            pretty_table(source),
            use_container_width=True,
            hide_index=True,
        )

        c1, c2 = st.columns(2)

        with c1:

            fig = px.bar(
                source,
                x="source",
                y="gross_revenue",
                title="Revenue by Traffic Source",
            )

            fig.update_layout(
                xaxis_title="",
                yaxis_title="Gross Revenue ($)",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with c2:

            device = (
                marketing
                .groupby(
                    "device_type",
                    as_index=False,
                )
                .agg(
                    sessions=("sessions", "sum"),
                    orders=("orders", "sum"),
                    gross_revenue=(
                        "gross_revenue",
                        "sum",
                    ),
                )
            )

            device["conversion_rate"] = (
                device["orders"]
                / device["sessions"]
                * 100
            )

            fig = px.bar(
                device,
                x="device_type",
                y="conversion_rate",
                title="Conversion Rate by Device",
            )

            fig.update_layout(
                xaxis_title="",
                yaxis_title="Conversion Rate (%)",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.subheader(
            "Product Drivers"
        )

        products = (
            cached_products()
            .sort_values(
                "net_revenue",
                ascending=False,
            )
        )

        st.dataframe(
            pretty_table(products),
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            products.sort_values(
                "net_revenue"
            ),
            x="net_revenue",
            y="product_name",
            orientation="h",
            title="Net Revenue by Product",
        )

        fig.update_layout(
            xaxis_title="Net Revenue ($)",
            yaxis_title="",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with summary_tab:

        month_keys = (
            monthly["month"]
            .dt.strftime("%Y-%m")
            .tolist()[1:]
        )

        selected_summary_month = st.selectbox(
            "Month for diagnostic summary",
            month_keys,
            index=max(
                len(month_keys) - 1,
                0,
            ),
            key="summary_month",
        )

        result = cached_root_cause(
            selected_summary_month
        )

        summary, engine = ai_summary(
            result
        )

        st.caption(
            f"Summary engine: {engine}"
        )

        st.markdown(
            safe_markdown_text(
                summary
            )
        )

        st.divider()

        st.write(
            "**Calculated evidence used by the summary**"
        )

        evidence = (
            result["decomposition"]
            .copy()
        )

        evidence["impact"] = (
            evidence["impact"]
            .map(
                lambda x: f"${x:,.0f}"
            )
        )

        evidence["share_abs_pct"] = (
            evidence["share_abs_pct"]
            .map(
                lambda x: f"{x:.1f}%"
            )
        )

        st.dataframe(
            pretty_table(evidence),
            use_container_width=True,
            hide_index=True,
        )

        worst_source = (
            result["source_drivers"]
            .sort_values("change")
            .head(5)
        )

        worst_product = (
            result["product_drivers"]
            .sort_values("change")
            .head(5)
        )

        c1, c2 = st.columns(2)

        with c1:

            st.write(
                "Largest source declines"
            )

            st.dataframe(
                pretty_table(
                    worst_source
                ),
                use_container_width=True,
                hide_index=True,
            )

        with c2:

            st.write(
                "Largest product declines"
            )

            st.dataframe(
                pretty_table(
                    worst_product
                ),
                use_container_width=True,
                hide_index=True,
            )


@st.cache_data(show_spinner=False)
def cached_prepare_product_monthly(
    raw: pd.DataFrame,
) -> pd.DataFrame:

    return prepare_product_monthly(
        raw
    )


@st.cache_data(show_spinner=False)
def cached_product_forecast(
    panel: pd.DataFrame,
    target: str,
) -> dict:

    return train_product_forecast(
        panel,
        target,
    )


def render_product_monthly_upload(
    raw: pd.DataFrame,
):

    st.success(
        "Detected dataset type: Product Monthly Demand"
    )

    st.caption(
        "This mode forecasts next-month product demand or revenue, "
        "detects product anomalies, and explains changes using "
        "transactions, units per transaction, customers, and unit value."
    )

    try:

        panel = (
            cached_prepare_product_monthly(
                raw
            )
        )

    except Exception as exc:

        st.error(
            f"Could not prepare the product-month dataset: {exc}"
        )

        return

    product_stats = eligible_products(
        panel,
        min_months=8,
    )

    if product_stats.empty:

        st.error(
            "No products have enough monthly history for forecasting. "
            "At least 8 months with 4 active months are required."
        )

        return

    monthly = portfolio_monthly(
        panel
    )

    start_month = panel[
        "Month"
    ].min()

    end_month = panel[
        "Month"
    ].max()

    with st.expander(
        "How this dataset is prepared",
        expanded=False,
    ):

        st.markdown(
            "- Duplicate StockCode + Month rows are combined into one product-month record.\n"
            "- Missing months after a product first appears are treated as zero demand.\n"
            "- Forecast validation uses the latest calendar months as the test set; rows are not randomly shuffled.\n"
            "- A last-month baseline is included so the ML models must beat a simple benchmark."
        )

    control1, control2 = st.columns(
        [1, 2]
    )

    with control1:

        target_label = st.radio(
            "Forecast target",
            [
                "Demand (Quantity)",
                "Revenue",
            ],
            horizontal=False,
            key="product_forecast_target",
        )

    target = (
        "Quantity"
        if target_label.startswith(
            "Demand"
        )
        else "Revenue"
    )

    product_options = (
        product_stats[
            "Product"
        ].tolist()
    )

    with control2:

        selected_product = st.selectbox(
            "Product",
            product_options,
            index=0,
            key="product_selector",
            help=(
                "Products are ordered "
                "by historical revenue."
            ),
        )

    selected_code = str(
        product_stats.loc[
            product_stats["Product"]
            == selected_product,
            "StockCode",
        ].iloc[0]
    )

    try:

        with st.spinner(
            "Training time-based forecasting models "
            "on the uploaded product history..."
        ):

            forecast = (
                cached_product_forecast(
                    panel,
                    target,
                )
            )

    except Exception as exc:

        st.error(
            f"Forecast training failed: {exc}"
        )

        return

    portfolio_forecasts = (
        forecast[
            "portfolio_forecasts"
        ].copy()
    )

    selected_forecast_rows = (
        portfolio_forecasts[
            portfolio_forecasts[
                "StockCode"
            ].astype(str)
            == selected_code
        ]
    )

    if selected_forecast_rows.empty:

        st.error(
            "A next-month forecast could not be generated "
            "for the selected product."
        )

        return

    selected_forecast = (
        selected_forecast_rows.iloc[0]
    )

    history = product_history(
        panel,
        selected_code,
    )

    active_months = int(
        (
            history["Quantity"] > 0
        ).sum()
    )

    st.caption(
        f"Detected monthly frequency · "
        f"{start_month.strftime('%b %Y')} to "
        f"{end_month.strftime('%b %Y')} · "
        f"{panel['StockCode'].nunique():,} products · "
        f"selected product has {len(history)} calendar months "
        f"({active_months} active)."
    )

    (
        overview_tab,
        forecast_tab,
        anomaly_tab,
        root_tab,
        portfolio_tab,
    ) = st.tabs(
        [
            "Portfolio Overview",
            "Product Forecast",
            "Anomalies",
            "Root Cause",
            "Portfolio Outlook",
        ]
    )

    with overview_tab:

        total_revenue = float(
            panel["Revenue"].sum()
        )

        total_quantity = float(
            panel["Quantity"].sum()
        )

        c1, c2, c3, c4, c5 = st.columns(
            5
        )

        c1.metric(
            "Uploaded Rows",
            f"{len(raw):,}",
        )

        c2.metric(
            "Products",
            f"{panel['StockCode'].nunique():,}",
        )

        c3.metric(
            "Calendar Months",
            f"{panel['Month'].nunique():,}",
        )

        c4.metric(
            "Total Revenue",
            fmt_compact_money(
                total_revenue
            ),
        )

        c5.metric(
            "Units",
            f"{total_quantity:,.0f}",
        )

        left, right = st.columns(
            2
        )

        with left:

            fig = px.line(
                monthly,
                x="Month",
                y="Revenue",
                markers=True,
                title="Monthly Portfolio Revenue",
            )

            fig.update_layout(
                xaxis_title="",
                yaxis_title="Revenue ($)",
            )

            fig.update_xaxes(
                tickformat="%b %Y"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with right:

            fig = px.line(
                monthly,
                x="Month",
                y="Quantity",
                markers=True,
                title="Monthly Portfolio Demand",
            )

            fig.update_layout(
                xaxis_title="",
                yaxis_title="Units",
            )

            fig.update_xaxes(
                tickformat="%b %Y"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        top_products = (
            panel
            .groupby(
                [
                    "StockCode",
                    "Description",
                ],
                as_index=False,
            )[
                [
                    "Revenue",
                    "Quantity",
                ]
            ]
            .sum()
            .sort_values(
                "Revenue",
                ascending=False,
            )
            .head(12)
        )

        top_products[
            "Product"
        ] = (
            top_products[
                "StockCode"
            ]
            + " — "
            + top_products[
                "Description"
            ]
        )

        fig = px.bar(
            top_products.sort_values(
                "Revenue"
            ),
            x="Revenue",
            y="Product",
            orientation="h",
            title=(
                "Top Products by Historical Revenue"
            ),
        )

        fig.update_layout(
            xaxis_title="Revenue ($)",
            yaxis_title="",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with forecast_tab:

        ml_metrics = (
            forecast[
                "metrics"
            ][
                forecast[
                    "metrics"
                ]["Type"] == "ML"
            ]
            .copy()
        )

        best = (
            ml_metrics[
                ml_metrics["Model"]
                == forecast[
                    "best_model"
                ]
            ]
            .iloc[0]
        )

        baseline = (
            forecast[
                "metrics"
            ][
                forecast[
                    "metrics"
                ]["Type"]
                == "Baseline"
            ]
            .iloc[0]
        )

        if target == "Quantity":

            forecast_value = (
                f"{selected_forecast['Forecast']:,.0f} units"
            )

            mae_value = (
                f"{best['MAE']:,.0f} units"
            )

        else:

            forecast_value = (
                f"${selected_forecast['Forecast']:,.0f}"
            )

            mae_value = (
                f"${best['MAE']:,.0f}"
            )

        selected_test = (
            forecast[
                "predictions"
            ][
                forecast[
                    "predictions"
                ]["StockCode"].astype(str)
                == selected_code
            ]
            .copy()
        )

        if (
            not selected_test.empty
            and selected_test[
                "Target"
            ].abs().sum() > 0
        ):

            product_wmape = (
                (
                    selected_test[
                        "Target"
                    ]
                    - selected_test[
                        "Predicted"
                    ]
                )
                .abs()
                .sum()
                / selected_test[
                    "Target"
                ]
                .abs()
                .sum()
                * 100
            )

            product_wmape_text = (
                f"{product_wmape:.1f}%"
            )

        else:

            product_wmape_text = "—"

        model_display = {
            "Histogram Gradient Boosting":
                "Gradient Boosting",
            "Linear Regression":
                "Linear Regression",
        }.get(
            forecast[
                "best_model"
            ],
            forecast[
                "best_model"
            ],
        )

        m1, m2, m3, m4 = st.columns(
            4
        )

        m1.metric(
            "Selected ML Model",
            model_display,
        )

        m2.metric(
            "Selected Product WMAPE",
            product_wmape_text,
        )

        m3.metric(
            "All-SKU Test WMAPE",
            f"{best['WMAPE']:.1f}%",
        )

        m4.metric(
            f"{forecast['next_month'].strftime('%b %Y')} Forecast",
            forecast_value,
        )

        st.caption(
            f"Time-based validation uses the latest "
            f"{len(forecast['test_months'])} calendar months "
            f"as the test period. "
            f"Overall MAE is {mae_value}. "
            f"The last-month baseline WMAPE is "
            f"{baseline['WMAPE']:.1f}%. "
            "The selected-product error is the more relevant "
            "metric for an individual SKU. "
            "The all-SKU metric is higher because the portfolio "
            "contains many intermittent and low-volume products."
        )

        if target == "Quantity":

            recent_rpu = (
                history.loc[
                    history[
                        "Revenue_Per_Unit"
                    ] > 0,
                    "Revenue_Per_Unit",
                ]
            )

            if not recent_rpu.empty:

                estimated_revenue = (
                    float(
                        selected_forecast[
                            "Forecast"
                        ]
                    )
                    * float(
                        recent_rpu.iloc[-1]
                    )
                )

                st.info(
                    "At the latest observed revenue per unit, "
                    "the demand forecast corresponds to approximately "
                    f"${estimated_revenue:,.0f} revenue. "
                    "This is a scenario estimate, "
                    "not a separate revenue model forecast."
                )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=history["Month"],
                y=history[target],
                mode="lines+markers",
                name="Actual",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[
                    forecast[
                        "next_month"
                    ]
                ],
                y=[
                    selected_forecast[
                        "Forecast"
                    ]
                ],
                mode="markers",
                name="Next-month forecast",
                marker=dict(
                    size=13,
                    symbol="diamond",
                ),
            )
        )

        fig.update_layout(
            title=(
                f"{selected_product}: "
                f"Historical {target} "
                f"and Next-Month Forecast"
            ),
            xaxis_title="",
            yaxis_title=(
                "Units"
                if target == "Quantity"
                else "Revenue ($)"
            ),
        )

        fig.update_xaxes(
            dtick="M3",
            tickformat="%b %Y",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        pred = (
            forecast[
                "predictions"
            ]
        )

        pred = (
            pred[
                pred[
                    "StockCode"
                ].astype(str)
                == selected_code
            ]
            .copy()
        )

        if not pred.empty:

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=pred[
                        "target_month"
                    ],
                    y=pred[
                        "Target"
                    ],
                    mode="lines+markers",
                    name="Actual",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=pred[
                        "target_month"
                    ],
                    y=pred[
                        "Predicted"
                    ],
                    mode="lines+markers",
                    name="ML Predicted",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=pred[
                        "target_month"
                    ],
                    y=pred[
                        "Baseline"
                    ],
                    mode="lines+markers",
                    name="Last-Month Baseline",
                )
            )

            fig.update_layout(
                title=(
                    "Out-of-Sample Test Months: "
                    "Actual vs Predicted"
                ),
                xaxis_title="",
                yaxis_title=(
                    "Units"
                    if target == "Quantity"
                    else "Revenue ($)"
                ),
            )

            fig.update_xaxes(
                dtick="M1",
                tickformat="%b %Y",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with st.expander(
            "Model comparison"
        ):

            table = (
                forecast[
                    "metrics"
                ][
                    [
                        "Model",
                        "Type",
                        "MAE",
                        "RMSE",
                        "R2",
                        "WMAPE",
                    ]
                ]
                .copy()
            )

            table[
                "MAE"
            ] = table[
                "MAE"
            ].round(2)

            table[
                "RMSE"
            ] = table[
                "RMSE"
            ].round(2)

            table[
                "R2"
            ] = table[
                "R2"
            ].round(3)

            table[
                "WMAPE"
            ] = (
                table[
                    "WMAPE"
                ]
                .round(1)
                .astype(str)
                + "%"
            )

            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
            )

    with anomaly_tab:

        anomalies = product_anomalies(
            panel,
            selected_code,
            target,
        )

        valid = (
            anomalies[
                anomalies[
                    "Baseline"
                ].notna()
            ]
            .copy()
        )

        if valid.empty:

            st.info(
                "Not enough history to calculate "
                "a rolling anomaly baseline for this product."
            )

        else:

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=valid[
                        "Month"
                    ],
                    y=valid[
                        target
                    ],
                    mode="lines+markers",
                    name="Actual",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=valid[
                        "Month"
                    ],
                    y=valid[
                        "Baseline"
                    ],
                    mode="lines",
                    name="6-month baseline",
                )
            )

            flagged = (
                valid[
                    valid[
                        "Is_Anomaly"
                    ]
                ]
            )

            if not flagged.empty:

                fig.add_trace(
                    go.Scatter(
                        x=flagged[
                            "Month"
                        ],
                        y=flagged[
                            target
                        ],
                        mode="markers",
                        name="Anomaly",
                        marker=dict(
                            size=13,
                            symbol="diamond",
                        ),
                    )
                )

            fig.update_layout(
                title=(
                    f"{selected_product}: "
                    f"{target} Anomaly Monitor"
                ),
                xaxis_title="",
                yaxis_title=(
                    "Units"
                    if target == "Quantity"
                    else "Revenue ($)"
                ),
            )

            fig.update_xaxes(
                dtick="M3",
                tickformat="%b %Y",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            show = (
                flagged[
                    [
                        "Month",
                        target,
                        "Baseline",
                        "Deviation_Pct",
                        "Z_Score",
                        "Severity",
                    ]
                ]
                .copy()
            )

            if show.empty:

                st.success(
                    "No anomalies were flagged "
                    "for this product under the current rules."
                )

            else:

                show[
                    "Month"
                ] = (
                    show[
                        "Month"
                    ]
                    .dt.strftime(
                        "%b %Y"
                    )
                )

                show[
                    "Baseline"
                ] = show[
                    "Baseline"
                ].round(2)

                show[
                    "Deviation_Pct"
                ] = show[
                    "Deviation_Pct"
                ].round(1)

                show[
                    "Z_Score"
                ] = show[
                    "Z_Score"
                ].round(2)

                show = show.rename(
                    columns={
                        "Deviation_Pct":
                            "Deviation %",
                        "Z_Score":
                            "Z-Score",
                    }
                )

                st.dataframe(
                    show,
                    use_container_width=True,
                    hide_index=True,
                )

    with root_tab:

        month_options = (
            history[
                "Month"
            ]
            .iloc[1:]
            .tolist()
        )

        selected_month = st.selectbox(
            "Month to diagnose",
            month_options,
            index=(
                len(
                    month_options
                )
                - 1
            ),
            format_func=lambda x: (
                pd.Timestamp(x)
                .strftime("%b %Y")
            ),
            key="product_root_month",
        )

        result = product_root_cause(
            panel,
            selected_code,
            selected_month,
        )

        curr = result[
            "current"
        ]

        prev = result[
            "previous"
        ]

        q_delta = result[
            "quantity_change"
        ]

        r_delta = result[
            "revenue_change"
        ]

        c1, c2, c3, c4, c5 = st.columns(
            5
        )

        c1.metric(
            "Demand",
            f"{curr['Quantity']:,.0f} units",
            f"{q_delta:+,.0f} units",
        )

        c2.metric(
            "Revenue",
            f"${curr['Revenue']:,.0f}",
            f"{r_delta:+,.0f} USD",
        )

        c3.metric(
            "Transactions",
            f"{curr['Transactions']:,.0f}",
            f"{result['transaction_change']:+,.0f}",
        )

        c4.metric(
            "Customers",
            f"{curr['Customers']:,.0f}",
            f"{result['customer_change']:+,.0f}",
        )

        c5.metric(
            "Avg Price",
            f"${curr['Avg_Price']:.2f}",
            f"{result['avg_price_change']:+.2f} USD",
        )

        left, right = st.columns(
            2
        )

        with left:

            dec = (
                result[
                    "demand_decomposition"
                ]
                .copy()
            )

            fig = px.bar(
                dec,
                x="Impact",
                y="Driver",
                orientation="h",
                title=(
                    "Demand Change Decomposition"
                ),
            )

            fig.update_layout(
                xaxis_title=(
                    "Estimated Unit Impact"
                ),
                yaxis_title="",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with right:

            dec = (
                result[
                    "revenue_decomposition"
                ]
                .copy()
            )

            fig = px.bar(
                dec,
                x="Impact",
                y="Driver",
                orientation="h",
                title=(
                    "Revenue Change Decomposition"
                ),
            )

            fig.update_layout(
                xaxis_title=(
                    "Estimated Revenue Impact ($)"
                ),
                yaxis_title="",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.caption(
            "The decomposition is sequential and reconciles "
            "the measured change through transaction volume, "
            "units per transaction, and revenue per unit. "
            "Customer counts are shown as supporting context."
        )

        local_text = (
            local_product_summary(
                result,
                selected_forecast,
                target,
            )
        )

        st.subheader(
            "Diagnostic Summary"
        )

        st.markdown(
            safe_markdown_text(
                local_text
            )
        )

        if os.getenv(
            "OPENAI_API_KEY"
        ):

            if st.button(
                "Generate AI explanation from calculated evidence",
                key="product_ai_button",
            ):

                with st.spinner(
                    "Generating explanation..."
                ):

                    ai_text, mode = (
                        product_ai_summary(
                            result,
                            selected_forecast,
                            target,
                        )
                    )

                st.caption(
                    f"Summary engine: {mode}"
                )

                st.markdown(
                    safe_markdown_text(
                        ai_text
                    )
                )

        else:

            st.caption(
                "Add OPENAI_API_KEY to .env "
                "if you want an optional AI-written explanation. "
                "Core calculations do not require AI."
            )

    with portfolio_tab:

        pf = (
            portfolio_forecasts
            .merge(
                product_stats[
                    [
                        "StockCode",
                        "Months",
                        "Active_Months",
                        "Total_Revenue",
                    ]
                ],
                on="StockCode",
                how="inner",
            )
        )

        pf[
            "Activity_Ratio"
        ] = np.where(
            pf[
                "Months"
            ] > 0,
            (
                pf[
                    "Active_Months"
                ]
                / pf[
                    "Months"
                ]
            ),
            0.0,
        )

        pf = (
            pf[
                (
                    pf[
                        "Active_Months"
                    ]
                    >= 8
                )
                & (
                    pf[
                        "Activity_Ratio"
                    ]
                    >= 0.50
                )
                & (
                    pf[
                        "Current_Value"
                    ]
                    > 0
                )
            ]
            .copy()
        )

        pf = pf.sort_values(
            "Forecast",
            ascending=False,
        )

        st.subheader(
            f"{forecast['next_month'].strftime('%b %Y')} Portfolio Forecast"
        )

        st.caption(
            "This table uses the same pooled ML model "
            "for every eligible product. "
            "Use it to prioritize products for review "
            "rather than as an inventory commitment."
        )

        chart_df = (
            pf
            .head(15)
            .sort_values(
                "Forecast"
            )
        )

        fig = px.bar(
            chart_df,
            x="Forecast",
            y="Product",
            orientation="h",
            title=(
                f"Top Predicted Products by {target}"
            ),
        )

        fig.update_layout(
            xaxis_title=(
                "Units"
                if target == "Quantity"
                else "Revenue ($)"
            ),
            yaxis_title="",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        left, right = st.columns(
            2
        )

        if not pf.empty:

            minimum_current_value = (
                pf[
                    "Current_Value"
                ]
                .quantile(0.25)
            )

        else:

            minimum_current_value = 0

        nonzero = (
            pf[
                pf[
                    "Current_Value"
                ]
                >= minimum_current_value
            ]
            .copy()
        )

        with left:

            growth = (
                nonzero
                .sort_values(
                    "Forecast_Change_Pct",
                    ascending=False,
                )
                .head(10)
            )

            st.write(
                "**Largest predicted increases**"
            )

            show = (
                growth[
                    [
                        "Product",
                        "Current_Value",
                        "Forecast",
                        "Forecast_Change_Pct",
                    ]
                ]
                .copy()
            )

            show = show.rename(
                columns={
                    "Current_Value":
                        "Current",
                    "Forecast":
                        "Forecast",
                    "Forecast_Change_Pct":
                        "Forecast Change %",
                }
            )

            show[
                "Current"
            ] = show[
                "Current"
            ].round(1)

            show[
                "Forecast"
            ] = show[
                "Forecast"
            ].round(1)

            show[
                "Forecast Change %"
            ] = show[
                "Forecast Change %"
            ].round(1)

            st.dataframe(
                show,
                use_container_width=True,
                hide_index=True,
            )

        with right:

            decline = (
                nonzero
                .sort_values(
                    "Forecast_Change_Pct"
                )
                .head(10)
            )

            st.write(
                "**Largest predicted declines**"
            )

            show = (
                decline[
                    [
                        "Product",
                        "Current_Value",
                        "Forecast",
                        "Forecast_Change_Pct",
                    ]
                ]
                .copy()
            )

            show = show.rename(
                columns={
                    "Current_Value":
                        "Current",
                    "Forecast":
                        "Forecast",
                    "Forecast_Change_Pct":
                        "Forecast Change %",
                }
            )

            show[
                "Current"
            ] = show[
                "Current"
            ].round(1)

            show[
                "Forecast"
            ] = show[
                "Forecast"
            ].round(1)

            show[
                "Forecast Change %"
            ] = show[
                "Forecast Change %"
            ].round(1)

            st.dataframe(
                show,
                use_container_width=True,
                hide_index=True,
            )

        export = (
            pf[
                [
                    "StockCode",
                    "Description",
                    "Forecast_Month",
                    "Current_Value",
                    "Forecast",
                    "Forecast_Change",
                    "Forecast_Change_Pct",
                    "Active_Months",
                ]
            ]
            .copy()
        )

        export[
            "Forecast_Month"
        ] = (
            pd.to_datetime(
                export[
                    "Forecast_Month"
                ]
            )
            .dt.strftime(
                "%Y-%m"
            )
        )

        st.download_button(
            "Download product forecasts (CSV)",
            data=(
                export
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8"
                )
            ),
            file_name=(
                f"product_{target.lower()}_forecast_"
                f"{forecast['next_month'].strftime('%Y_%m')}.csv"
            ),
            mime="text/csv",
        )


def render_upload():

    st.subheader(
        "Upload Your Business CSV"
    )

    st.caption(
        "The forecasting model is retrained on the uploaded file. "
        "Date and Revenue are required. "
        "Sessions, Orders, Refunds, Source, Product, and Device "
        "are optional and make root-cause analysis richer."
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=[
            "csv"
        ],
        key="business_csv",
    )

    if uploaded is None:

        st.info(
            "Upload a CSV to begin. "
            "The file should preferably contain at least "
            "20 weeks of historical data; "
            "30+ weeks is recommended."
        )

        st.code(
            "date,revenue,sessions,orders,refunds,source,product,device\n"
            "2025-01-01,12500,2100,180,320,google,Product A,desktop",
            language="text",
        )

        return

    try:

        raw = pd.read_csv(
            uploaded,
            low_memory=False,
        )

    except Exception as exc:

        st.error(
            f"Could not read the CSV: {exc}"
        )

        return

    if (
        raw.empty
        or len(
            raw.columns
        ) < 2
    ):

        st.error(
            "The CSV must contain data "
            "and at least two columns."
        )

        return

    st.success(
        f"Loaded {len(raw):,} rows "
        f"and {len(raw.columns)} columns."
    )

    with st.expander(
        "Preview uploaded data",
        expanded=False,
    ):

        st.dataframe(
            raw.head(50),
            use_container_width=True,
            hide_index=True,
        )

    if is_product_monthly_dataset(
        raw
    ):

        render_product_monthly_upload(
            raw
        )

        return

    st.info(
        "Generic CSV mode: map your business columns below. "
        "For the richest root-cause analysis, "
        "include Sessions and Orders "
        "in addition to Date and Revenue."
    )

    columns = list(
        raw.columns
    )

    date_guess = guess_column(
        columns,
        [
            "date",
            "created_at",
            "order_date",
            "week",
            "month",
            "timestamp",
        ],
    )

    revenue_guess = guess_column(
        columns,
        [
            "revenue",
            "sales",
            "net_revenue",
            "gross_revenue",
            "amount",
            "price_usd",
        ],
    )

    sessions_guess = guess_column(
        columns,
        [
            "sessions",
            "visits",
            "traffic",
        ],
    )

    orders_guess = guess_column(
        columns,
        [
            "orders",
            "order_count",
            "purchases",
        ],
    )

    refunds_guess = guess_column(
        columns,
        [
            "refunds",
            "refund_amount",
            "refund_amount_usd",
        ],
    )

    source_guess = guess_column(
        columns,
        [
            "source",
            "utm_source",
            "channel",
            "marketing_channel",
        ],
    )

    product_guess = guess_column(
        columns,
        [
            "product",
            "product_name",
            "category",
        ],
    )

    device_guess = guess_column(
        columns,
        [
            "device",
            "device_type",
        ],
    )

    st.write(
        "**Map your columns**"
    )

    c1, c2 = st.columns(
        2
    )

    with c1:

        date_index = (
            columns.index(
                date_guess
            )
            if date_guess
            in columns
            else 0
        )

        date_col = st.selectbox(
            "Date column *",
            columns,
            index=date_index,
        )

        revenue_index = (
            columns.index(
                revenue_guess
            )
            if revenue_guess
            in columns
            else min(
                1,
                len(columns) - 1,
            )
        )

        revenue_col = st.selectbox(
            "Revenue column *",
            columns,
            index=revenue_index,
        )

        sessions_col = optional_select(
            "Sessions column",
            columns,
            sessions_guess,
            "map_sessions",
        )

        orders_col = optional_select(
            "Orders column",
            columns,
            orders_guess,
            "map_orders",
        )

    with c2:

        refunds_col = optional_select(
            "Refund amount column",
            columns,
            refunds_guess,
            "map_refunds",
        )

        source_col = optional_select(
            "Traffic / marketing source column",
            columns,
            source_guess,
            "map_source",
        )

        product_col = optional_select(
            "Product / category column",
            columns,
            product_guess,
            "map_product",
        )

        device_col = optional_select(
            "Device column",
            columns,
            device_guess,
            "map_device",
        )

    revenue_includes_refunds = True

    if refunds_col:

        revenue_includes_refunds = st.checkbox(
            "Revenue already includes refunds",
            value=True,
            help=(
                "Keep this checked when the Revenue column "
                "is already net of refunds. "
                "Uncheck it only when refunds should be "
                "subtracted from Revenue."
            ),
        )

    cfg = UploadConfig(
        date_col=date_col,
        revenue_col=revenue_col,
        sessions_col=sessions_col,
        orders_col=orders_col,
        refunds_col=refunds_col,
        source_col=source_col,
        product_col=product_col,
        device_col=device_col,
        revenue_includes_refunds=revenue_includes_refunds,
    )

    try:

        data = prepare_uploaded_data(
            raw,
            cfg,
        )

        monthly = uploaded_monthly_kpis(
            data,
            cfg,
        )

        weekly = uploaded_weekly_kpis(
            data,
            cfg,
        )

        k = uploaded_executive_kpis(
            data,
            cfg,
        )

    except Exception as exc:

        st.error(
            f"Could not prepare the uploaded data: {exc}"
        )

        return

    st.caption(
        f"Valid date range: "
        f"{k['start_date'].date()} to "
        f"{k['end_date'].date()} · "
        f"{len(weekly)} weekly periods · "
        f"{len(monthly)} monthly periods"
    )

    (
        overview_tab,
        forecast_tab,
        root_tab,
        drivers_tab,
    ) = st.tabs(
        [
            "Executive Overview",
            "Forecast & Anomalies",
            "Root Cause Analysis",
            "Segment Drivers",
        ]
    )

    with overview_tab:

        metrics = st.columns(
            6
        )

        metrics[0].metric(
            "Net Revenue",
            fmt_money(
                k[
                    "net_revenue"
                ]
            ),
        )

        metrics[1].metric(
            "Rows",
            f"{k['rows']:,}",
        )

        metrics[2].metric(
            "Sessions",
            fmt_num(
                k[
                    "sessions"
                ]
            ),
        )

        metrics[3].metric(
            "Orders",
            fmt_num(
                k[
                    "orders"
                ]
            ),
        )

        metrics[4].metric(
            "Conversion Rate",
            fmt_pct(
                k[
                    "conversion_rate"
                ]
            ),
        )

        metrics[5].metric(
            "Avg Order Value",
            (
                "—"
                if pd.isna(
                    k[
                        "aov"
                    ]
                )
                else f"${k['aov']:.2f}"
            ),
        )

        c1, c2 = st.columns(
            2
        )

        with c1:

            fig = px.line(
                monthly,
                x="month",
                y="net_revenue",
                markers=True,
                title=(
                    "Monthly Net Revenue"
                ),
            )

            fig.update_layout(
                xaxis_title="",
                yaxis_title=(
                    "Net Revenue"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with c2:

            if monthly[
                "orders"
            ].notna().any():

                fig = px.line(
                    monthly,
                    x="month",
                    y="orders",
                    markers=True,
                    title=(
                        "Monthly Orders"
                    ),
                )

                fig.update_layout(
                    xaxis_title="",
                    yaxis_title=(
                        "Orders"
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:

                st.info(
                    "Map an Orders column to show "
                    "order trends and conversion/AOV metrics."
                )

    with forecast_tab:

        try:

            forecast = train_forecast(
                weekly
            )

            m1, m2, m3, m4 = st.columns(
                4
            )

            best_row = (
                forecast[
                    "metrics"
                ]
                .iloc[0]
            )

            m1.metric(
                "Selected Model",
                forecast[
                    "best_model"
                ],
            )

            m2.metric(
                "Test MAE",
                f"${best_row['MAE']:,.0f}",
            )

            m3.metric(
                "Test MAPE",
                (
                    "—"
                    if pd.isna(
                        best_row[
                            "MAPE"
                        ]
                    )
                    else f"{best_row['MAPE']:.1f}%"
                ),
            )

            m4.metric(
                "Next Week Forecast",
                f"${forecast['next_prediction']:,.0f}",
            )

            st.caption(
                f"Forecast for week beginning "
                f"{forecast['next_week'].strftime('%d %b %Y')}. "
                f"Practical MAE band: "
                f"${forecast['lower']:,.0f} to "
                f"${forecast['upper']:,.0f}."
            )

            pred = forecast[
                "predictions"
            ]

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=pred[
                        "week"
                    ],
                    y=pred[
                        "net_revenue"
                    ],
                    mode="lines+markers",
                    name="Actual",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=pred[
                        "week"
                    ],
                    y=pred[
                        "predicted_revenue"
                    ],
                    mode="lines+markers",
                    name="Predicted",
                )
            )

            fig.update_layout(
                title=(
                    "Uploaded Data: "
                    "Out-of-Sample Actual vs Predicted"
                ),
                xaxis_title="",
                yaxis_title="Revenue",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            with st.expander(
                "Model comparison"
            ):

                metrics_table = (
                    forecast[
                        "metrics"
                    ]
                    .copy()
                )

                for col in [
                    "MAE",
                    "RMSE",
                ]:

                    metrics_table[
                        col
                    ] = (
                        metrics_table[
                            col
                        ]
                        .map(
                            lambda x: f"${x:,.0f}"
                        )
                    )

                metrics_table[
                    "R2"
                ] = (
                    metrics_table[
                        "R2"
                    ]
                    .map(
                        lambda x: (
                            "—"
                            if pd.isna(x)
                            else f"{x:.3f}"
                        )
                    )
                )

                metrics_table[
                    "MAPE"
                ] = (
                    metrics_table[
                        "MAPE"
                    ]
                    .map(
                        lambda x: (
                            "—"
                            if pd.isna(x)
                            else f"{x:.1f}%"
                        )
                    )
                )

                st.dataframe(
                    metrics_table.rename(
                        columns={
                            "model":
                                "Model"
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        except ValueError as exc:

            st.warning(
                str(exc)
            )

        anomalies = (
            detect_uploaded_anomalies(
                weekly
            )
        )

        valid = (
            anomalies[
                anomalies[
                    "baseline"
                ].notna()
            ]
            .copy()
        )

        if not valid.empty:

            recent = valid.tail(
                40
            )

            fig = px.line(
                recent,
                x="week",
                y="net_revenue",
                markers=True,
                title=(
                    "Weekly Revenue "
                    "with 8-Week Baseline"
                ),
            )

            fig.add_scatter(
                x=recent[
                    "week"
                ],
                y=recent[
                    "baseline"
                ],
                mode="lines",
                name="8-week baseline",
            )

            anomalous = (
                recent[
                    recent[
                        "is_anomaly"
                    ]
                ]
            )

            if not anomalous.empty:

                fig.add_scatter(
                    x=anomalous[
                        "week"
                    ],
                    y=anomalous[
                        "net_revenue"
                    ],
                    mode="markers",
                    name="Anomaly",
                    marker=dict(
                        size=12,
                        symbol="diamond",
                    ),
                )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            show = (
                valid[
                    valid[
                        "is_anomaly"
                    ]
                ][
                    [
                        "week",
                        "net_revenue",
                        "baseline",
                        "deviation_pct",
                        "z_score",
                        "severity",
                    ]
                ]
                .sort_values(
                    "week",
                    ascending=False,
                )
                .head(10)
                .copy()
            )

            if not show.empty:

                show[
                    "week"
                ] = (
                    show[
                        "week"
                    ]
                    .dt.strftime(
                        "%d %b %Y"
                    )
                )

                show[
                    "net_revenue"
                ] = (
                    show[
                        "net_revenue"
                    ]
                    .map(
                        lambda x: f"${x:,.0f}"
                    )
                )

                show[
                    "baseline"
                ] = (
                    show[
                        "baseline"
                    ]
                    .map(
                        lambda x: f"${x:,.0f}"
                    )
                )

                show[
                    "deviation_pct"
                ] = (
                    show[
                        "deviation_pct"
                    ]
                    .map(
                        lambda x: f"{x:.1f}%"
                    )
                )

                show[
                    "z_score"
                ] = (
                    show[
                        "z_score"
                    ]
                    .map(
                        lambda x: f"{x:.2f}"
                    )
                )

                st.write(
                    "**Detected anomalies**"
                )

                st.dataframe(
                    pretty_table(
                        show
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    with root_tab:

        month_keys = (
            monthly[
                "month"
            ]
            .dt.strftime(
                "%Y-%m"
            )
            .tolist()[1:]
        )

        if not month_keys:

            st.warning(
                "At least two months of data are required "
                "for month-over-month root-cause analysis."
            )

        else:

            selected = st.selectbox(
                "Month to diagnose",
                month_keys,
                index=(
                    len(
                        month_keys
                    )
                    - 1
                ),
                key="upload_root_month",
            )

            result = (
                compare_uploaded_months(
                    data,
                    cfg,
                    selected,
                )
            )

            current = result[
                "current"
            ]

            previous = result[
                "previous"
            ]

            cols = st.columns(
                4
            )

            cols[0].metric(
                "Net Revenue Change",
                f"${result['net_revenue_change']:,.0f}",
                f"{result['net_revenue_change_pct']:.1f}%",
            )

            cols[1].metric(
                "Current Revenue",
                f"${current['net_revenue']:,.0f}",
            )

            if pd.notna(
                current[
                    "conversion_rate"
                ]
            ):

                cols[2].metric(
                    "Conversion Rate",
                    f"{current['conversion_rate']:.2f}%",
                    (
                        f"{current['conversion_rate'] - previous['conversion_rate']:+.2f} pp"
                    ),
                )

            else:

                cols[2].metric(
                    "Conversion Rate",
                    "—",
                )

            if pd.notna(
                current[
                    "aov"
                ]
            ):

                aov_delta = (
                    current[
                        "aov"
                    ]
                    - previous[
                        "aov"
                    ]
                )

                cols[3].metric(
                    "AOV",
                    f"${current['aov']:.2f}",
                    f"{aov_delta:+.2f} USD",
                )

            else:

                cols[3].metric(
                    "AOV",
                    "—",
                )

            dec = (
                result[
                    "decomposition"
                ]
                .sort_values(
                    "impact"
                )
            )

            fig = px.bar(
                dec,
                x="impact",
                y="driver",
                orientation="h",
                title=(
                    "Revenue Change Decomposition"
                ),
            )

            fig.update_layout(
                xaxis_title=(
                    "Estimated Impact"
                ),
                yaxis_title="",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            if not (
                cfg.sessions_col
                and cfg.orders_col
            ):

                st.info(
                    "For Traffic / Conversion / AOV decomposition, "
                    "map both Sessions and Orders columns. "
                    "Revenue change and segment drivers "
                    "still work without them."
                )

            driver_frames = [
                (
                    "Source Contribution",
                    result[
                        "source_drivers"
                    ],
                ),
                (
                    "Product Contribution",
                    result[
                        "product_drivers"
                    ],
                ),
                (
                    "Device Contribution",
                    result[
                        "device_drivers"
                    ],
                ),
            ]

            available = [
                (
                    title,
                    frame,
                )
                for title, frame
                in driver_frames
                if not frame.empty
            ]

            if available:

                for i in range(
                    0,
                    len(available),
                    2,
                ):

                    pair = (
                        available[
                            i:i + 2
                        ]
                    )

                    cols2 = st.columns(
                        len(
                            pair
                        )
                    )

                    for (
                        col_area,
                        (
                            title,
                            frame,
                        ),
                    ) in zip(
                        cols2,
                        pair,
                    ):

                        with col_area:

                            chart_df = (
                                frame
                                .sort_values(
                                    "change"
                                )
                                .tail(12)
                            )

                            fig = px.bar(
                                chart_df,
                                x="change",
                                y="dimension",
                                orientation="h",
                                title=title,
                            )

                            fig.update_layout(
                                xaxis_title=(
                                    "Revenue Change"
                                ),
                                yaxis_title="",
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                            )

            else:

                st.info(
                    "Map Source, Product, or Device columns "
                    "to identify which segments drove "
                    "the revenue change."
                )

            direction = (
                "increased"
                if result[
                    "net_revenue_change"
                ] >= 0
                else "decreased"
            )

            summary = (
                f"Net revenue {direction} by "
                f"{abs(result['net_revenue_change_pct']):.1f}% "
                f"versus {result['previous_month']}."
            )

            negative = (
                dec[
                    dec[
                        "impact"
                    ] < 0
                ]
                .sort_values(
                    "impact"
                )
            )

            positive = (
                dec[
                    dec[
                        "impact"
                    ] > 0
                ]
                .sort_values(
                    "impact",
                    ascending=False,
                )
            )

            if not negative.empty:

                row = (
                    negative
                    .iloc[0]
                )

                summary += (
                    " The largest negative measured driver was "
                    f"{row['driver'].lower()} "
                    f"({row['impact']:,.0f})."
                )

            if (
                cfg.source_col
                and not result[
                    "source_drivers"
                ].empty
            ):

                row = (
                    result[
                        "source_drivers"
                    ]
                    .sort_values(
                        "change"
                    )
                    .iloc[0]
                )

                summary += (
                    f" The weakest {cfg.source_col} segment was "
                    f"{row['dimension']} "
                    f"({row['change']:,.0f})."
                )

            if (
                cfg.product_col
                and not result[
                    "product_drivers"
                ].empty
            ):

                row = (
                    result[
                        "product_drivers"
                    ]
                    .sort_values(
                        "change"
                    )
                    .iloc[0]
                )

                summary += (
                    f" The weakest {cfg.product_col} segment was "
                    f"{row['dimension']} "
                    f"({row['change']:,.0f})."
                )

            if not positive.empty:

                row = (
                    positive
                    .iloc[0]
                )

                summary += (
                    " The strongest positive offset was "
                    f"{row['driver'].lower()} "
                    f"({row['impact']:,.0f})."
                )

            st.subheader(
                "Diagnostic Summary"
            )

            st.markdown(
                safe_markdown_text(
                    summary
                )
            )

    with drivers_tab:

        dimension_options = [
            (
                "Source",
                cfg.source_col,
            ),
            (
                "Product",
                cfg.product_col,
            ),
            (
                "Device",
                cfg.device_col,
            ),
        ]

        available_dims = [
            (
                label,
                col,
            )
            for label, col
            in dimension_options
            if col
        ]

        if not available_dims:

            st.info(
                "Map Source, Product, or Device "
                "to unlock segment performance tables and charts."
            )

        else:

            data_for_dims = (
                data.copy()
            )

            for label, col in available_dims:

                st.subheader(
                    f"{label} Performance"
                )

                grouped = (
                    data_for_dims
                    .groupby(
                        col,
                        dropna=False,
                    )
                    .agg(
                        revenue=(
                            "_net_revenue",
                            "sum",
                        ),
                        rows=(
                            "_net_revenue",
                            "size",
                        ),
                    )
                    .reset_index()
                    .sort_values(
                        "revenue",
                        ascending=False,
                    )
                )

                grouped[
                    col
                ] = (
                    grouped[
                        col
                    ]
                    .fillna(
                        "Unknown"
                    )
                    .astype(str)
                )

                st.dataframe(
                    grouped.head(25),
                    use_container_width=True,
                    hide_index=True,
                )

                chart = (
                    grouped
                    .head(15)
                    .sort_values(
                        "revenue"
                    )
                )

                fig = px.bar(
                    chart,
                    x="revenue",
                    y=col,
                    orientation="h",
                    title=(
                        f"Revenue by {label}"
                    ),
                )

                fig.update_layout(
                    xaxis_title="Revenue",
                    yaxis_title="",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )


if dataset_mode == "Built-in Demo Dataset":

    render_builtin()

else:

    render_upload()
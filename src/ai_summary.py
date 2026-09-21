from __future__ import annotations

import json
import os
import pandas as pd


def local_summary(result: dict) -> str:
    change = result["net_revenue_change"]
    pct = result["net_revenue_change_pct"]
    direction = "increased" if change >= 0 else "decreased"

    dec = result["decomposition"].copy()
    negative_rows = dec[dec["impact"] < 0].sort_values("impact")
    positive_rows = dec[dec["impact"] > 0].sort_values("impact", ascending=False)

    negative = negative_rows.iloc[0] if not negative_rows.empty else None
    positive = positive_rows.iloc[0] if not positive_rows.empty else None

    source_table = result["source_drivers"].sort_values("change")
    product_table = result["product_drivers"].sort_values("change")
    source_negative = source_table[source_table["change"] < 0]
    product_negative = product_table[product_table["change"] < 0]

    text = f"Net revenue {direction} by {abs(pct):.1f}% versus the previous month. "

    if negative is not None:
        text += (
            f"The largest negative funnel driver was {negative['driver'].lower()} "
            f"with an estimated impact of ${negative['impact']:,.0f}. "
        )
    elif positive is not None:
        text += (
            f"All measured funnel components were non-negative; the strongest driver was "
            f"{positive['driver'].lower()} (${positive['impact']:,.0f}). "
        )

    if not source_negative.empty:
        row = source_negative.iloc[0]
        text += f"The largest source-level decline came from {row['dimension']} (${row['change']:,.0f}). "
    else:
        row = source_table.sort_values("change").iloc[0]
        text += f"No traffic source declined; the smallest source gain was {row['dimension']} (${row['change']:,.0f}). "

    if not product_negative.empty:
        row = product_negative.iloc[0]
        text += f"The largest product-level decline came from {row['dimension']} (${row['change']:,.0f}). "
    else:
        row = product_table.sort_values("change").iloc[0]
        text += f"No product declined; the smallest product gain was {row['dimension']} (${row['change']:,.0f}). "

    if positive is not None and (negative is None or positive["driver"] != negative["driver"]):
        text += (
            f"The strongest positive offset was {positive['driver'].lower()} "
            f"(${positive['impact']:,.0f})."
        )

    return text


def ai_summary(result: dict) -> tuple[str, str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return local_summary(result), "Local diagnostic summary"

    compact = {
        "current_month": result["current_month"],
        "previous_month": result["previous_month"],
        "net_revenue_change": result["net_revenue_change"],
        "net_revenue_change_pct": result["net_revenue_change_pct"],
        "funnel_impacts": result["decomposition"].to_dict(orient="records"),
        "worst_sources": result["source_drivers"].head(3).to_dict(orient="records"),
        "worst_campaigns": result["campaign_drivers"].head(3).to_dict(orient="records"),
        "worst_devices": result["device_drivers"].head(3).to_dict(orient="records"),
        "worst_products": result["product_drivers"].head(3).to_dict(orient="records"),
    }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a business data analyst. Explain only the supplied calculated results. "
                        "Do not invent causes, data, or recommendations unsupported by the inputs. "
                        "Write 4 concise bullet points: overall change, main funnel driver, main segment/product driver, and one practical investigation action."
                    ),
                },
                {"role": "user", "content": json.dumps(compact, default=str)},
            ],
        )
        return response.output_text.strip(), f"OpenAI ({model})"
    except Exception:
        return local_summary(result), "Local diagnostic summary (AI unavailable)"

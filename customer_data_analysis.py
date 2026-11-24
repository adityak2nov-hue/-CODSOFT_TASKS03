"""Customer segmentation and reporting example for an internship task.

By default, this creates a reproducible synthetic dataset so the project can
run without access to real customer records. To analyze an approved CSV, pass
--input path/to/customers.csv. Expected columns: customer_id, age, location,
orders, total_spend, and optionally last_purchase_days_ago.

Install:
    python -m pip install pandas matplotlib openpyxl

Run:
    python customer_data_analysis.py
    python customer_data_analysis.py --input customers.csv --output-dir report
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
REQUIRED_COLUMNS = {"customer_id", "age", "location", "orders", "total_spend"}


def make_sample_data(count: int = 600, seed: int = 42) -> pd.DataFrame:
    """Generate fake customer data for a safe, repeatable demonstration."""
    rng = np.random.default_rng(seed)
    locations = ["North", "South", "East", "West", "Central"]
    ages = rng.integers(18, 71, count)
    orders = rng.poisson(4, count) + 1
    average_order = rng.gamma(shape=4.0, scale=18.0, size=count)
    spend = (orders * average_order).round(2)
    # Add a small high-value group to make segmentation insights visible.
    premium = rng.random(count) < 0.12
    spend[premium] *= rng.uniform(1.8, 3.0, premium.sum())
    orders[premium] += rng.integers(3, 10, premium.sum())
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, count + 1)],
        "age": ages,
        "location": rng.choice(locations, count),
        "orders": orders,
        "total_spend": spend,
        "last_purchase_days_ago": rng.integers(0, 181, count),
    })


def load_data(input_path: Path | None) -> pd.DataFrame:
    if input_path is None:
        logging.info("No CSV supplied; using synthetic demonstration data.")
        frame = make_sample_data()
    else:
        frame = pd.read_csv(input_path)
        if "last_purchase_days_ago" not in frame.columns:
            frame["last_purchase_days_ago"] = pd.NA

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(sorted(missing))}")

    frame = frame.copy()
    for col in ("age", "orders", "total_spend", "last_purchase_days_ago"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["location"] = frame["location"].astype("string").str.strip().replace("", pd.NA)
    frame = frame.drop_duplicates(subset="customer_id")
    frame = frame.dropna(subset=["customer_id", "age", "location", "orders", "total_spend"])
    frame = frame[(frame["age"] >= 0) & (frame["orders"] >= 0) & (frame["total_spend"] >= 0)]
    frame["age_group"] = pd.cut(
        frame["age"], bins=[0, 24, 34, 44, 54, 64, float("inf")],
        labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"], right=True,
    )
    frame["average_order_value"] = frame["total_spend"] / frame["orders"].where(frame["orders"] > 0)
    return frame.reset_index(drop=True)


def summarize(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    by_age = frame.groupby("age_group", observed=False).agg(
        customers=("customer_id", "nunique"),
        average_spend=("total_spend", "mean"),
        total_revenue=("total_spend", "sum"),
        average_orders=("orders", "mean"),
    ).reset_index()
    by_location = frame.groupby("location", dropna=False).agg(
        customers=("customer_id", "nunique"),
        average_spend=("total_spend", "mean"),
        total_revenue=("total_spend", "sum"),
        average_orders=("orders", "mean"),
    ).reset_index().sort_values("total_revenue", ascending=False)

    # Value segments are defined relative to the dataset's 80th percentile.
    value_cutoff = frame["total_spend"].quantile(0.80)
    frame["value_segment"] = frame["total_spend"].ge(value_cutoff).map({True: "High value", False: "Other customers"})
    value_segments = frame.groupby("value_segment").agg(
        customers=("customer_id", "nunique"),
        total_revenue=("total_spend", "sum"),
        average_spend=("total_spend", "mean"),
        average_orders=("orders", "mean"),
    ).reset_index()
    value_segments["revenue_share_pct"] = (value_segments["total_revenue"] / frame["total_spend"].sum() * 100).round(1)
    return {"age": by_age, "location": by_location, "value": value_segments}


def make_recommendations(frame: pd.DataFrame, summaries: dict[str, pd.DataFrame]) -> list[str]:
    recs: list[str] = []
    top_location = summaries["location"].iloc[0]
    high_value = frame[frame["value_segment"] == "High value"]
    recs.append(
        f"Prioritize retention offers for high-value customers ({len(high_value)} records; "
        f"they contribute {high_value['total_spend'].sum() / frame['total_spend'].sum():.0%} of recorded spend)."
    )
    recs.append(
        f"Test a region-specific campaign in {top_location['location']}, which has the highest total spend in this dataset."
    )
    if frame["last_purchase_days_ago"].notna().any():
        inactive = frame[frame["last_purchase_days_ago"] >= 90]
        if len(inactive):
            recs.append(f"Consider a win-back test for the {len(inactive)} customers with no purchase in 90 or more days.")
    top_age = summaries["age"].sort_values("total_revenue", ascending=False).iloc[0]
    recs.append(f"Compare messaging for age group {top_age['age_group']}, the highest-revenue age band here.")
    recs.append("Run small A/B tests and compare incremental revenue before scaling any campaign.")
    return recs


def create_charts(frame: pd.DataFrame, summaries: dict[str, pd.DataFrame], output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    age = summaries["age"].dropna(subset=["age_group"])
    axes[0, 0].bar(age["age_group"].astype(str), age["customers"], color="#2389a8")
    axes[0, 0].set(title="Customers by age group", xlabel="Age group", ylabel="Customers")
    loc = summaries["location"].sort_values("total_revenue", ascending=False)
    axes[0, 1].bar(loc["location"].astype(str), loc["total_revenue"], color="#244d73")
    axes[0, 1].set(title="Revenue by location", xlabel="Location", ylabel="Total spend")
    axes[0, 1].tick_params(axis="x", rotation=25)
    axes[1, 0].hist(frame["total_spend"], bins=24, color="#36a9bf", edgecolor="white")
    axes[1, 0].set(title="Customer spend distribution", xlabel="Total spend", ylabel="Customers")
    axes[1, 1].scatter(frame["orders"], frame["total_spend"], alpha=0.55, color="#244d73")
    axes[1, 1].set(title="Orders and total spend", xlabel="Number of orders", ylabel="Total spend")
    fig.suptitle("Customer Data Analysis (aggregated view)", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "customer_insights.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_report(frame: pd.DataFrame, summaries: dict[str, pd.DataFrame], recommendations: list[str], output_dir: Path) -> None:
    total_revenue = frame["total_spend"].sum()
    avg_spend = frame["total_spend"].mean()
    high_value_share = (frame["value_segment"] == "High value").mean() * 100
    lines = [
        "# Customer Data Analysis Report", "",
        "> Results are descriptive and depend on the supplied dataset. Marketing ideas are hypotheses to test, not causal findings.", "",
        "## Overview", "",
        f"- Valid unique customer records: {len(frame):,}",
        f"- Total recorded spend: {total_revenue:,.2f}",
        f"- Average spend per customer: {avg_spend:,.2f}",
        f"- High-value segment: top 20% spend threshold; {high_value_share:.1f}% of records", "",
        "## Highest revenue location", "",
        "```text",
        summaries["location"].head(1).to_string(index=False),
        "```", "",
        "## Suggested experiments", "",
    ]
    lines.extend(f"- {item}" for item in recommendations)
    lines.extend(["", "## Data handling", "", "Use only approved data. Remove direct identifiers before analysis, limit access to source files, and follow your internship's privacy and retention rules."])
    (output_dir / "customer_insights_report.md").write_text("\n".join(lines), encoding="utf-8")


def save_outputs(frame: pd.DataFrame, summaries: dict[str, pd.DataFrame], recommendations: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Export aggregates and a de-identified analysis table, not customer IDs.
    export_frame = frame.drop(columns=["customer_id"], errors="ignore")
    export_frame.to_csv(output_dir / "cleaned_customer_data.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(output_dir / "customer_analysis.xlsx", engine="openpyxl") as writer:
        export_frame.to_excel(writer, sheet_name="Cleaned Data", index=False)
        summaries["age"].to_excel(writer, sheet_name="Age Summary", index=False)
        summaries["location"].to_excel(writer, sheet_name="Location Summary", index=False)
        summaries["value"].to_excel(writer, sheet_name="Value Segments", index=False)
        pd.DataFrame({"recommendation": recommendations}).to_excel(writer, sheet_name="Experiments", index=False)
    create_charts(frame, summaries, output_dir)
    write_report(frame, summaries, recommendations, output_dir)
    logging.info("Reports saved to %s", output_dir.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Approved customer CSV; omit to use synthetic sample data")
    parser.add_argument("--output-dir", type=Path, default=Path("customer_report"))
    args = parser.parse_args()
    try:
        frame = load_data(args.input)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    if frame.empty:
        parser.error("No valid customer rows remain after cleaning.")
    frame["value_segment"] = frame["total_spend"].ge(frame["total_spend"].quantile(0.80)).map({True: "High value", False: "Other customers"})
    summaries = summarize(frame)
    recommendations = make_recommendations(frame, summaries)
    print(f"\nCustomers analyzed: {len(frame)}")
    print(f"Average spend: {frame['total_spend'].mean():.2f}")
    print("\nRevenue by location:\n" + summaries["location"].round(2).to_string(index=False))
    print("\nSuggested marketing experiments:")
    for rec in recommendations:
        print(f"- {rec}")
    save_outputs(frame, summaries, recommendations, args.output_dir)


if __name__ == "__main__":
    main()

"""Create an interactive scatter map of internal migration in Kazakhstan.

The script reads tidy CSV data stored under ``data/`` and uses Plotly to
render a bubble map that encodes the magnitude and direction of net
migration by region.  The underlying CSVs are plain-text assets so the
repository can be shared without tracking large binary files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "migration_map.html"

MONTH_CHOICES = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "all",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--month",
        default="all",
        choices=MONTH_CHOICES,
        help=(
            "Month to visualise. Use 'all' to aggregate January–December 2024 "
            "data into a single annual view."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the HTML file that will be generated.",
    )
    return parser.parse_args()


def _load_data(month: str) -> Tuple[pd.DataFrame, str]:
    data = pd.read_csv(DATA_DIR / "internal_migration_2024.csv")
    centroids = pd.read_csv(DATA_DIR / "region_centroids.csv")

    month = month.lower()
    if month == "all":
        grouped = (
            data.groupby("region", as_index=False)[["arrivals", "departures", "net"]]
            .sum()
            .assign(month_label="January–December 2024")
        )
    else:
        grouped = (
            data.loc[data["month"].str.lower() == month]
            .copy()
            .assign(month_label=lambda df: df["month"].str.title() + " 2024")
        )

    merged = grouped.merge(centroids, on="region", how="left")
    if merged[["latitude", "longitude"]].isna().any().any():
        missing = merged.loc[merged["latitude"].isna(), "region"].tolist()
        raise ValueError(
            "Missing coordinates for the following regions: " + ", ".join(missing)
        )

    title = f"Internal migration flows – {merged['month_label'].iloc[0]}"
    return merged, title


def _compute_symbol_size(values: pd.Series) -> pd.Series:
    baseline = 8
    scale = 0.25
    return baseline + scale * np.sqrt(values.abs())


def build_figure(data: pd.DataFrame, title: str):
    data = data.assign(symbol_size=_compute_symbol_size(data["net"]))

    hover_template = (
        "<b>%{customdata[0]}</b><br>"
        "Arrivals: %{customdata[1]:,}<br>"
        "Departures: %{customdata[2]:,}<br>"
        "Net: %{z:,}"
    )

    fig = px.scatter_geo(
        data,
        lat="latitude",
        lon="longitude",
        size="symbol_size",
        color="net",
        color_continuous_scale=["#b30000", "#fef0d9", "#31a354"],
        hover_name="region",
        hover_data={"region": False, "arrivals": True, "departures": True, "net": True},
        projection="mercator",
    )

    fig.update_traces(
        marker=dict(line=dict(width=0.5, color="#333")),
        customdata=data[["region", "arrivals", "departures"]].to_numpy(),
        hovertemplate=hover_template,
    )

    fig.update_layout(
        title=title,
        geo=dict(
            scope="asia",
            showcountries=True,
            showland=True,
            landcolor="#f7f7f7",
            countrycolor="#b0b0b0",
            lonaxis=dict(range=[45, 90]),
            lataxis=dict(range=[40, 56]),
        ),
        coloraxis_colorbar=dict(
            title="Net migration",
            ticksuffix=" people",
        ),
    )

    return fig


def main() -> None:
    args = parse_args()
    data, title = _load_data(args.month)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = build_figure(data, title)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Map saved to {output_path}")


if __name__ == "__main__":
    main()

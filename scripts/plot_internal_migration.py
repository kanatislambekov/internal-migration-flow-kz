"""Create an interactive scatter map of internal migration in Kazakhstan.

The script reads tidy CSV data stored under ``data/`` and uses Plotly to
render a bubble map that encodes the magnitude and direction of net
migration by region.  The underlying CSVs are plain-text assets so the
repository can be shared without tracking large binary files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "migration_map.html"
GEOJSON_PATH = DATA_DIR / "kz.json"
GEOJSON_FEATURE_KEY = "properties.name"

GEOJSON_NAME_OVERRIDES = {
    "Abay": "Abai",
    "Aqmola": "Akmola",
    "Aqtobe": "Aktobe",
    "Almaty region": "Almaty",
    "Almaty city": "Almaty (city)",
    "Astana city": "Astana",
    "Qaraganda": "Karaganda",
    "Qostanay": "Kostanay",
    "Qyzylorda": "Kyzylorda",
    "Shymkent city": "Shymkent (city)",
    "Turkistan": "Turkestan",
    "Ulutay": "Ulytau",
}

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
]
ALL_MONTH_CHOICE = "all"

CITY_REGIONS = {"Astana city", "Almaty city", "Shymkent city"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--month",
        default=ALL_MONTH_CHOICE,
        choices=MONTH_CHOICES + [ALL_MONTH_CHOICE],
        help=(
            "Month to visualise. Use 'all' to aggregate January–December 2024 "
            "data into a single annual view."
        ),
    )
    parser.add_argument(
        "--months",
        nargs="+",
        choices=MONTH_CHOICES,
        metavar="MONTH",
        help=(
            "One or more months to aggregate. Specify multiple values (e.g. "
            "'--months january february') to pool several months together."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the HTML file that will be generated.",
    )
    args = parser.parse_args()
    if args.months and args.month != ALL_MONTH_CHOICE:
        parser.error("Use either --month or --months, not both.")
    return args


def _format_multi_month_label(months: list[str]) -> str:
    titled = [month.title() for month in months]
    if len(titled) == 1:
        return f"{titled[0]} 2024"
    if len(titled) == 2:
        return f"{titled[0]} and {titled[1]} 2024"
    return ", ".join(titled[:-1]) + f", and {titled[-1]} 2024"


def _load_data(month_selection: str | list[str]) -> Tuple[pd.DataFrame, str]:
    data = pd.read_csv(DATA_DIR / "internal_migration_2024.csv")
    centroids = pd.read_csv(DATA_DIR / "region_centroids.csv")

    if isinstance(month_selection, str):
        month = month_selection.lower()
        if month == ALL_MONTH_CHOICE:
            grouped = (
                data.groupby("region", as_index=False)[["arrivals", "departures"]]
                .sum()
                .assign(month_label="January–December 2024")
            )
        else:
            grouped = (
                data.loc[data["month"].str.lower() == month]
                .copy()
                .assign(month_label=lambda df: df["month"].str.title() + " 2024")
            )
    else:
        month_lookup = {name: idx for idx, name in enumerate(MONTH_CHOICES)}
        selected = [month.lower() for month in month_selection]
        unique_months: list[str] = []
        seen = set()
        for month in selected:
            if month not in month_lookup:
                raise ValueError(f"Unknown month provided: {month}")
            if month not in seen:
                unique_months.append(month)
                seen.add(month)

        sorted_months = sorted(unique_months, key=month_lookup.get)
        mask = data["month"].str.lower().isin(sorted_months)
        filtered = data.loc[mask].copy()
        if filtered.empty:
            raise ValueError("No data available for the selected months.")
        label = _format_multi_month_label(sorted_months)
        grouped = (
            filtered.groupby("region", as_index=False)[["arrivals", "departures"]]
            .sum()
            .assign(month_label=label)
        )

    grouped = grouped.assign(net=grouped["arrivals"] - grouped["departures"])
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


def _load_geojson() -> dict:
    with GEOJSON_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalise_regions_for_geojson(data: pd.DataFrame, geojson: dict) -> pd.DataFrame:
    feature_names = {feature["properties"]["name"] for feature in geojson["features"]}
    geo_regions = data["region"].map(GEOJSON_NAME_OVERRIDES).fillna(data["region"])

    missing = sorted(set(geo_regions) - feature_names)
    if missing:
        raise ValueError(
            "The GeoJSON file does not contain the following regions: "
            + ", ".join(missing)
        )

    return data.assign(geo_region=geo_regions)


def build_figure(data: pd.DataFrame, title: str, geojson: dict):
    data = _normalise_regions_for_geojson(data, geojson)
    data = data.assign(symbol_size=_compute_symbol_size(data["net"]))

    customdata = data[["region", "arrivals", "departures", "net"]].to_numpy()
    hover_template = (
        "<b>%{customdata[0]}</b><br>"
        "Arrivals: %{customdata[1]:,}<br>"
        "Departures: %{customdata[2]:,}<br>"
        "Net: %{customdata[3]:,}<extra></extra>"
    )

    city_mask = data["region"].isin(CITY_REGIONS)
    city_data = data.loc[city_mask]
    city_customdata = city_data[["region", "arrivals", "departures", "net"]].to_numpy()

    max_abs_net = data["net"].abs().max()
    if max_abs_net == 0:
        max_abs_net = 1

    fig = go.Figure()
    fig.add_choropleth(
        geojson=geojson,
        locations=data["geo_region"],
        featureidkey=GEOJSON_FEATURE_KEY,
        z=data["net"],
        customdata=customdata,
        hovertemplate=hover_template,
        coloraxis="coloraxis",
        marker=dict(line=dict(width=0.8, color="#4c4c4c")),
        name="",
        showscale=False,
    )

    if not city_data.empty:
        fig.add_scattergeo(
            lon=city_data["longitude"],
            lat=city_data["latitude"],
            customdata=city_customdata,
            marker=dict(
                size=city_data["symbol_size"],
                color=city_data["net"],
                coloraxis="coloraxis",
                line=dict(width=0.6, color="#333"),
            ),
            hovertemplate=hover_template,
            name="",
            showlegend=False,
        )

    fig.update_geos(
        projection_type="mercator",
        fitbounds="locations",
        visible=False,
    )

    fig.update_layout(
        title=title,
        coloraxis=dict(
            colorscale=["#b30000", "#fef0d9", "#31a354"],
            cmin=-max_abs_net,
            cmax=max_abs_net,
            colorbar=dict(title="Net migration", ticksuffix=" people"),
        ),
        margin=dict(l=0, r=0, t=70, b=0),
    )

    return fig


def main() -> None:
    args = parse_args()
    month_selection: str | list[str]
    if args.months:
        month_selection = args.months
    else:
        month_selection = args.month

    data, title = _load_data(month_selection)
    geojson = _load_geojson()

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = build_figure(data, title, geojson)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Map saved to {output_path}")


if __name__ == "__main__":
    main()

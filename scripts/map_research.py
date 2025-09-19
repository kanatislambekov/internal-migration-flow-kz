"""
Internal Migration in Kazakhstan — Research Dashboard (2024, optional 2010)

Sections:
1) Opening Hook: hero map + one-line kicker.
2) Context & Critical Insight: provocative question + placeholder for lit review.
3) Data Storytelling: interactive map (hover), year toggle (2022–2024, 2010 if available),
   OD Sankey (if data present), small charts (age/sex).
4) Critical Themes: short commentary bullets.
5) What-If: simple projection to 2030 with adjustable annual trend.
6) Concluding Insight: compact takeaway.

Data files (CSV, tidy):
- data/internal_migration_2024.csv : region,month,arrivals,departures
- data/internal_migration_2010.csv : region,month,arrivals,departures   (optional)
- data/region_centroids.csv        : region,latitude,longitude
- data/kz.json                     : GeoJSON regions
- data/od_flows_2024.csv           : origin,destination,count            (optional)
- data/od_flows_2010.csv           : origin,destination,count            (optional)
- data/migration_by_age_2024.csv   : region,age_group,value              (optional)
- data/migration_by_sex_2024.csv   : region,sex,value                    (optional)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px  # only for simple bars

# -------------------------
# Paths & constants
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "outputs" / "migration_dashboard.html"
GEOJSON_PATH = DATA / "kz.json"
GEOJSON_FEATURE_KEY = "properties.name"

CITY_REGIONS = {"Astana city", "Almaty city", "Shymkent city"}
MONTHS = ["january","february","march","april","may","june","july","august","september","october","november","december"]

NAME_OVERRIDES = {
    "Abay": "Abai","Aqmola": "Akmola","Aqtobe": "Aktobe","Almaty region": "Almaty",
    "Almaty city": "Almaty (city)","Astana city": "Astana","Qaraganda": "Karaganda",
    "Qostanay": "Kostanay","Qyzylorda": "Kyzylorda","Shymkent city": "Shymkent (city)",
    "Turkistan": "Turkestan","Ulutay": "Ulytau",
}

MAP_CONFIG = dict(
    scrollZoom=True,
    doubleClick="reset",
    modeBarButtonsToRemove=["pan2d", "select2d", "lasso2d", "autoScale2d"],
    displaylogo=False,
)

# -------------------------
# Utilities
# -------------------------
def read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None

def load_geojson() -> dict:
    with GEOJSON_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def norm_regions_for_geojson(df: pd.DataFrame, geo: dict) -> pd.DataFrame:
    feature_names = {f["properties"]["name"] for f in geo["features"]}
    geo_regions = df["region"].map(NAME_OVERRIDES).fillna(df["region"])
    missing = sorted(set(geo_regions) - feature_names)
    if missing:
        raise ValueError("GeoJSON missing regions: " + ", ".join(missing))
    return df.assign(geo_region=geo_regions)

def symbol_size(values: pd.Series) -> pd.Series:
    """Return a smaller point size for city markers on the map."""
    return 6 + 0.18 * np.sqrt(values.abs())

def aggregate_months(df: pd.DataFrame, months: List[str]) -> pd.DataFrame:
    mask = df["month"].str.lower().isin(months)
    agg = (df.loc[mask]
             .groupby("region", as_index=False)[["arrivals","departures"]].sum())
    agg["net"] = agg["arrivals"] - agg["departures"]
    return agg

def total_movers_text(df_annual: pd.DataFrame) -> str:
    # A soft estimate: sum of arrivals across regions (each mover counted once)
    movers = int(df_annual["arrivals"].sum())
    return f"Every year tens of thousands move within Kazakhstan. (≈{movers:,} arrivals recorded in 2024.)"

# -------------------------
# MAP with Year Toggle (frames)
# -------------------------
def build_map_with_year_toggle(
    datasets: Dict[str, pd.DataFrame], centroids: pd.DataFrame, geo: dict
) -> go.Figure:
    if not datasets:
        raise ValueError("At least one annual dataset is required to render the map.")

    def prep(df_raw: pd.DataFrame, label: str) -> pd.DataFrame:
        annual = aggregate_months(df_raw, MONTHS).merge(centroids, on="region", how="left")
        if annual[["latitude", "longitude"]].isna().any().any():
            miss = annual.loc[annual["latitude"].isna(), "region"].unique().tolist()
            raise ValueError("Missing coordinates for: " + ", ".join(miss))
        annual = norm_regions_for_geojson(annual, geo)
        annual["symbol_size"] = symbol_size(annual["net"])
        annual["year_label"] = label
        return annual

    views = {label: prep(df, label) for label, df in datasets.items()}

    max_abs = max(v["net"].abs().max() for v in views.values())
    max_abs = float(max(1, max_abs))

    def frame_traces(df: pd.DataFrame):
        choro = go.Choropleth(
            geojson=geo,
            locations=df["geo_region"],
            featureidkey=GEOJSON_FEATURE_KEY,
            z=df["net"],
            customdata=df[["region", "arrivals", "departures", "net"]].to_numpy(),
            hovertemplate="<b>%{customdata[0]}</b><br>Arrivals: %{customdata[1]:,}"
                          "<br>Departures: %{customdata[2]:,}<br>Net: %{customdata[3]:,}<extra></extra>",
            coloraxis="coloraxis",
            marker=dict(line=dict(width=0.8, color="#4c4c4c")),
            name="",
            showscale=False,
        )
        cities = df.loc[df["region"].isin(CITY_REGIONS)]
        scatter = go.Scattergeo(
            lon=cities["longitude"],
            lat=cities["latitude"],
            customdata=cities[["region", "arrivals", "departures", "net"]].to_numpy(),
            marker=dict(
                size=cities["symbol_size"],
                color=cities["net"],
                coloraxis="coloraxis",
                line=dict(width=0.6, color="#333"),
            ),
            hovertemplate="<b>%{customdata[0]}</b><br>Arrivals: %{customdata[1]:,}"
                          "<br>Departures: %{customdata[2]:,}<br>Net: %{customdata[3]:,}<extra></extra>",
            name="",
            showlegend=False,
        )
        return [choro, scatter]

    init_year = next(iter(views))
    fig = go.Figure(data=frame_traces(views[init_year]))
    fig.frames = [
        go.Frame(
            name=year,
            data=frame_traces(df),
            layout=go.Layout(title_text=f"Internal migration — {year}"),
        )
        for year, df in views.items()
    ]

    fig.update_geos(projection_type="mercator", fitbounds="locations", visible=False)
    fig.update_layout(
        title=f"Internal migration — {init_year}",
        coloraxis=dict(
            colorscale=["#b30000", "#fef0d9", "#31a354"],
            cmin=-max_abs,
            cmax=max_abs,
            colorbar=dict(title="Net migration", ticksuffix=" people"),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        dragmode="zoom",
        updatemenus=[
            dict(
                type="dropdown",
                x=0.02,
                y=0.97,
                xanchor="left",
                yanchor="top",
                pad=dict(t=6, l=4, r=4, b=4),
                buttons=[
                    dict(
                        label=yr,
                        method="animate",
                        args=[[yr], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    )
                    for yr in views.keys()
                ],
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#ddd",
            )
        ],
    )
    return fig

# -------------------------
# Sankey (OD flows)
# -------------------------
def build_sankey(df_od: pd.DataFrame, title: str) -> go.Figure:
    # Build node list
    regions = sorted(set(df_od["origin"]).union(set(df_od["destination"])))
    idx = {r:i for i,r in enumerate(regions)}
    link = dict(
        source=df_od["origin"].map(idx),
        target=df_od["destination"].map(idx),
        value=df_od["count"]
    )
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=regions, pad=12, thickness=12),
        link=link
    ))
    fig.update_layout(title=title, margin=dict(l=10,r=10,t=40,b=10))
    return fig

# -------------------------
# Small charts (age/sex)
# -------------------------
def build_age_bar(df_age: pd.DataFrame) -> go.Figure:
    # Expect columns: region, age_group, value (net or arrivals)
    order = ["0-14","15-24","25-34","35-44","45-54","55-64","65+"]
    df = df_age.copy()
    if "age_group" in df and set(order).issuperset(set(df["age_group"].unique())):
        df["age_group"] = pd.Categorical(df["age_group"], order, ordered=True)
    fig = px.bar(df, x="age_group", y="value", color="region", barmode="group",
                 labels={"value":"Migrants","age_group":"Age group"},
                 title="Migration by age group (2024)")
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=20), legend_title="")
    return fig

def build_sex_bar(df_sex: pd.DataFrame) -> go.Figure:
    # Expect columns: region, sex, value
    df = df_sex.copy()
    fig = px.bar(df, x="sex", y="value", color="region", barmode="group",
                 labels={"value":"Migrants","sex":"Sex"},
                 title="Migration by sex (2024)")
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=20), legend_title="")
    return fig

# -------------------------
# What-If projection (to 2030)
# -------------------------
def build_projection(df24_annual: pd.DataFrame, region_default: str = "Astana") -> go.Figure:
    """
    Very simple scenario: assume an annual percentage change in net migration (slider via HTML),
    but here we prep a base trace for 0% change; HTML will add a little JS to update.
    """
    # Base: cumulative net = constant 2024 net each year
    years = list(range(2024, 2031))
    base = df24_annual.set_index("region")["net"].to_dict()
    region = region_default if region_default in base else next(iter(base.keys()))
    y = [base[region]*(i-2023) for i in years]  # cumulative
    fig = go.Figure(go.Scatter(x=years, y=y, mode="lines+markers", name=region))
    fig.update_layout(
        title=f"Simple projection of cumulative net migration — {region} (to 2030)",
        xaxis_title="Year", yaxis_title="Cumulative net migration since 2024",
        margin=dict(l=40,r=10,t=50,b=40)
    )
    # store per-region base in figure JSON for small JS updater
    fig.update_layout(meta=dict(base_net=base))
    return fig

# -------------------------
# Assemble HTML shell
# -------------------------
def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Load core datasets
    df24 = pd.read_csv(DATA / "internal_migration_2024.csv")
    df23 = read_csv_safe(DATA / "internal_migration_2023.csv")
    df22 = read_csv_safe(DATA / "internal_migration_2022.csv")
    df10 = read_csv_safe(DATA / "internal_migration_2010.csv")
    centroids = pd.read_csv(DATA / "region_centroids.csv")
    geo = load_geojson()

    # Hook text (uses arrivals sum across 2024)
    annual24 = aggregate_months(df24, MONTHS)
    hook_line = total_movers_text(annual24)

    # Build figures
    datasets = {"2024": df24}
    if df23 is not None:
        datasets["2023"] = df23
    if df22 is not None:
        datasets["2022"] = df22
    if df10 is not None:
        datasets["2010"] = df10

    fig_map = build_map_with_year_toggle(datasets, centroids, geo)
    map_html = fig_map.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        div_id="map_hero",
        config=MAP_CONFIG,
    )

    # Data storytelling: optional Sankey & small charts
    sankey_html_blocks = []
    od24 = read_csv_safe(DATA / "od_flows_2024.csv")
    if od24 is not None and not od24.empty:
        sankey_html_blocks.append(build_sankey(od24, "Origin–destination flows (2024)")
                                  .to_html(full_html=False, include_plotlyjs=False, div_id="sankey24"))
    od10 = read_csv_safe(DATA / "od_flows_2010.csv")
    if od10 is not None and not od10.empty:
        sankey_html_blocks.append(build_sankey(od10, "Origin–destination flows (2010)")
                                  .to_html(full_html=False, include_plotlyjs=False, div_id="sankey10"))
    sankey_html = "\n".join(sankey_html_blocks) if sankey_html_blocks else \
        '<p class="muted">Origin–destination flow data not found — add <code>od_flows_*.csv</code> to enable Sankey.</p>'

    age_html = ''
    df_age = read_csv_safe(DATA / "migration_by_age_2024.csv")
    if df_age is not None and not df_age.empty:
        age_html = build_age_bar(df_age).to_html(full_html=False, include_plotlyjs=False, div_id="agebar")
    else:
        age_html = '<p class="muted">Age-group breakdown not found — add <code>migration_by_age_2024.csv</code>.</p>'

    sex_html = ''
    df_sex = read_csv_safe(DATA / "migration_by_sex_2024.csv")
    if df_sex is not None and not df_sex.empty:
        sex_html = build_sex_bar(df_sex).to_html(full_html=False, include_plotlyjs=False, div_id="sexbar")
    else:
        sex_html = '<p class="muted">Sex breakdown not found — add <code>migration_by_sex_2024.csv</code>.</p>'

    # What-If projection
    proj_fig = build_projection(annual24)
    proj_html = proj_fig.to_html(full_html=False, include_plotlyjs=False, div_id="projection")

    # HTML skeleton
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Internal Migration in Kazakhstan — Research Dashboard</title>
<style>
  body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; color:#111827; margin: 24px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1,h2,h3 {{ margin: 0 0 8px 0; }}
  .muted {{ color:#6b7280; }}
  .section {{ margin: 22px 0 28px; }}
  .grid {{ display: grid; gap: 16px; grid-template-columns: repeat(auto-fit,minmax(320px,1fr)); }}
  .card {{ border:1px solid #e5e7eb; border-radius:12px; padding:14px; box-shadow:0 1px 3px rgba(0,0,0,0.04); }}
</style>
</head>
<body>
<div class="container">

  <!-- 1. OPENING HOOK -->
  <div class="section">
    <h1>Internal Migration in Kazakhstan</h1>
    <p class="muted">{hook_line}</p>
    {map_html}
  </div>

  <!-- 2. CONTEXT & CRITICAL INSIGHT -->
  <div class="section">
    <h2>Why do Astana and Almaty attract so many migrants — and what does this mean for the regions they leave?</h2>
    <p class="muted">Add literature review here (drivers: wages, universities, services; sending-region effects; selection; temporary vs permanent, etc.).</p>
  </div>

  <!-- 3. DATA STORYTELLING -->
  <div class="section">
    <h2>Data Storytelling</h2>
    <p>Hover the map to see net migration by oblast. Use the dropdown to explore 2022–2024 (and 2010 if available).</p>
    <div class="grid">
      <div class="card" style="grid-column: 1 / -1;">
        <h3>Origin–Destination Flows</h3>
        {sankey_html}
      </div>
      <div class="card">
        <h3>By Age Group</h3>
        {age_html}
      </div>
      <div class="card">
        <h3>By Sex</h3>
        {sex_html}
      </div>
    </div>
  </div>

  <!-- 4. CRITICAL THEMES -->
  <div class="section">
    <h2>Critical Themes</h2>
    <ul>
      <li><b>Urban Overload</b> — housing shortages and transport stress in Astana/Almaty.</li>
      <li><b>Rural Decline</b> — labor shortages and population aging in sending regions.</li>
      <li><b>Policy Blind Spots</b> — undercounted/temporary flows distort planning.</li>
      <li><b>Inequality</b> — moves follow unequal jobs, education, and services.</li>
    </ul>
  </div>

  <!-- 5. WHAT-IF SCENARIO -->
  <div class="section">
    <h2>What If?</h2>
    <div class="card">
      <label>Region:
        <select id="regionSelect"></select>
      </label>
      <label style="margin-left:12px;">Annual change in net migration:
        <input id="rateInput" type="range" min="-20" max="20" value="0" step="1"/>
        <span id="rateLabel">0%</span>
      </label>
      {proj_html}
      <p id="projNote" class="muted" style="margin-top:6px;"></p>
    </div>
  </div>

  <!-- 6. CONCLUDING INSIGHT -->
  <div class="section">
    <h2>Concluding Insight</h2>
    <p><em>Migration is not just people moving — it is Kazakhstan’s future workforce, housing pressure, and regional inequality unfolding in real time.</em></p>
  </div>

</div>

<script>
  // === What-If: tiny client-side updater using meta embedded in the projection figure ===
  (function() {{
    const fig = window.document.getElementById("projection");
    if (!fig) return;
    // Plotly embeds figure data in a script tag near the div; use Plotly API to read back
    const gd = fig;  // the graph div itself
    const base = (gd.layout.meta && gd.layout.meta.base_net) ? gd.layout.meta.base_net : null;
    if (!base) return;

    // Populate region dropdown
    const regions = Object.keys(base).sort();
    const sel = document.getElementById("regionSelect");
    regions.forEach(r => {{
      const opt = document.createElement("option");
      opt.value = r; opt.textContent = r;
      sel.appendChild(opt);
    }});
    sel.value = regions.includes("Astana") ? "Astana" : regions[0];

    const rateInput = document.getElementById("rateInput");
    const rateLabel = document.getElementById("rateLabel");
    const note = document.getElementById("projNote");

    function recompute() {{
      const region = sel.value;
      const r = parseFloat(rateInput.value) / 100.0;
      rateLabel.textContent = (r*100).toFixed(0) + "%";
      const years = [2024,2025,2026,2027,2028,2029,2030];
      const net0 = base[region] || 0;
      // geometric change of annual net; cumulative sum over years
      let annuals = years.map((_,i)=> net0 * Math.pow(1+r, i));
      let cum = [];
      annuals.reduce((acc,v,idx)=> (cum[idx]=acc+v, acc+v), 0);

      Plotly.restyle(gd, {{"x":[years], "y":[cum], "name":[region]}}, [0]);

      // quick narrative: relative to 2024 net times (years)
      const total2030 = cum[cum.length-1] || 0;
      note.textContent = region + " cumulative net by 2030: " + Math.round(total2030).toLocaleString('en-US') + " people"
        + " (assumes constant composition; slider applies ±% change to annual net).";
    }}

    rateInput.addEventListener("input", recompute);
    sel.addEventListener("change", recompute);
    recompute();
  }})();
</script>

</body>
</html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {OUT}")

if __name__ == "__main__":
    main()

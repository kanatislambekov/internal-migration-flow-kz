"""
Internal Migration in Kazakhstan — Research Dashboard (2022–2024, optional 2010)

Sections:
1) Opening Hook: hero map + one-line kicker.
2) Context & Critical Insight: provocative question + placeholder for lit review.
3) Data Storytelling: interactive map (hover), year toggle (2022–2024, 2010 if available),
   OD Sankey (if data present), small charts (age/sex).
4) Critical Themes: short commentary bullets.
5) What-If: simple projection to 2030 with adjustable annual trend (baseline year selectable).
6) Concluding Insight: compact takeaway.

Data files (CSV, tidy):
- data/internal_migration_2024.csv : region,month,arrivals,departures
- data/internal_migration_2023.csv : region,month,arrivals,departures   (optional)
- data/internal_migration_2022.csv : region,month,arrivals,departures   (optional)
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
MONTHS = [
    "january","february","march","april","may","june",
    "july","august","september","october","november","december"
]

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
    # modest bubble sizing for cities
    return 6 + 0.18 * np.sqrt(values.abs())

def aggregate_months(df: pd.DataFrame, months: List[str]) -> pd.DataFrame:
    # robust to different month labels; if none match, use all rows
    if "month" in df.columns:
        mcol = df["month"].astype(str).str.lower()
        mask = mcol.isin(months)
        use = df.loc[mask].copy() if mask.any() else df.copy()
    else:
        use = df.copy()

    agg = (
        use.groupby("region", as_index=False)[["arrivals","departures"]]
           .sum()
    )
    agg["net"] = agg["arrivals"] - agg["departures"]
    return agg

def total_movers_text(df_annual: pd.DataFrame, year_label: str) -> str:
    movers = int(df_annual["arrivals"].sum())
    return f"Every year tens of thousands move within Kazakhstan. (≈{movers:,} arrivals recorded in {year_label}.)"

# -------------------------
# MAP with Year Toggle (frames)
# -------------------------
def build_map_with_year_toggle(
    datasets: Dict[str, pd.DataFrame], centroids: pd.DataFrame, geo: dict
) -> go.Figure:
    if not datasets:
        raise ValueError("At least one annual dataset is required to render the map.")

    # Sort years numerically where possible so dropdown order is nice
    def _year_key(k: str):
        try:
            return int(k)
        except:
            return -10**9  # non-numeric first

    ordered_items = sorted(datasets.items(), key=lambda kv: _year_key(kv[0]))
    views: Dict[str, pd.DataFrame] = {}

    def prep(df_raw: pd.DataFrame, label: str) -> pd.DataFrame:
        annual = aggregate_months(df_raw, MONTHS).merge(centroids, on="region", how="left")
        if annual[["latitude", "longitude"]].isna().any().any():
            miss = annual.loc[annual["latitude"].isna(), "region"].unique().tolist()
            raise ValueError("Missing coordinates for: " + ", ".join(miss))
        annual = norm_regions_for_geojson(annual, geo)
        annual["symbol_size"] = symbol_size(annual["net"])
        annual["year_label"] = label
        return annual

    for label, df in ordered_items:
        views[label] = prep(df, label)

    max_abs = float(max(1, max(v["net"].abs().max() for v in views.values())))

    def frame_traces(df: pd.DataFrame):
        choro = go.Choropleth(
            geojson=geo,
            locations=df["geo_region"],
            featureidkey=GEOJSON_FEATURE_KEY,
            z=df["net"],
            customdata=df[["region", "arrivals", "departures", "net"]].to_numpy(),
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "Arrivals: %{customdata[1]:,}<br>"
                          "Departures: %{customdata[2]:,}<br>"
                          "Net: %{customdata[3]:,}<extra></extra>",
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
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "Arrivals: %{customdata[1]:,}<br>"
                          "Departures: %{customdata[2]:,}<br>"
                          "Net: %{customdata[3]:,}<extra></extra>",
            name="",
            showlegend=False,
        )
        return [choro, scatter]

    # start on the latest year (the last key after sorting)
    init_year = list(views.keys())[-1]

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
    regions = sorted(set(df_od["origin"]).union(set(df_od["destination"])))
    idx = {r: i for i, r in enumerate(regions)}
    link = dict(
        source=df_od["origin"].map(idx),
        target=df_od["destination"].map(idx),
        value=df_od["count"],
    )
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=regions, pad=12, thickness=12),
        link=link
    ))
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# -------------------------
# Small charts (age/sex)
# -------------------------
def build_age_bar(df_age: pd.DataFrame) -> go.Figure:
    order = ["0-14","15-24","25-34","35-44","45-54","55-64","65+"]
    df = df_age.copy()
    if "age_group" in df and set(order).issuperset(set(df["age_group"].unique())):
        df["age_group"] = pd.Categorical(df["age_group"], order, ordered=True)
    fig = px.bar(
        df, x="age_group", y="value", color="region", barmode="group",
        labels={"value": "Migrants", "age_group": "Age group"},
        title="Migration by age group (selected year)"
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=20), legend_title="")
    return fig

def build_sex_bar(df_sex: pd.DataFrame) -> go.Figure:
    df = df_sex.copy()
    fig = px.bar(
        df, x="sex", y="value", color="region", barmode="group",
        labels={"value": "Migrants", "sex": "Sex"},
        title="Migration by sex (selected year)"
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=20), legend_title="")
    return fig

# -------------------------
# What-If projection (to 2030) — baseline year selectable
# -------------------------
def build_projection(bases: Dict[str, pd.DataFrame], default_region: str = "Astana") -> go.Figure:
    """
    Prepare a single trace; JS will swap region/year and recompute cumulative curve.
    We embed per-year base nets in layout.meta for the front-end updater.
    """
    # Build base net dicts per year
    meta_base = {}
    for y, df in bases.items():  # y like "2022"/"2023"/"2024"
        annual = df[["region", "net"]].set_index("region")["net"].to_dict()
        meta_base[y] = annual

    # Choose display defaults (prefer 2024 if present)
    start_year = "2024" if "2024" in meta_base else sorted(meta_base.keys())[-1]
    region = default_region if default_region in meta_base[start_year] else next(iter(meta_base[start_year].keys()))
    net0 = meta_base[start_year][region]

    # initial x/y (will be replaced by JS anyway)
    years = list(range(int(start_year), 2031))
    cum = [net0 * (i) for i in range(len(years))]

    fig = go.Figure(go.Scatter(x=years, y=cum, mode="lines+markers", name=f"{region} ({start_year})"))
    fig.update_layout(
        title=f"Simple projection of cumulative net migration — {region} ({start_year}) to 2030",
        xaxis_title="Year", yaxis_title=f"Cumulative net migration since {start_year}",
        margin=dict(l=40, r=10, t=50, b=40),
        meta=dict(base_net_by_year=meta_base)  # embed for JS
    )
    return fig

# -------------------------
# Assemble HTML shell
# -------------------------
def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Load core datasets
    df24 = read_csv_safe(DATA / "internal_migration_2024.csv")
    df23 = read_csv_safe(DATA / "internal_migration_2023.csv")
    df22 = read_csv_safe(DATA / "internal_migration_2022.csv")
    df10 = read_csv_safe(DATA / "internal_migration_2010.csv")
    centroids = pd.read_csv(DATA / "region_centroids.csv")
    geo = load_geojson()

    # Build datasets dict for the map (only non-empty)
    datasets: Dict[str, pd.DataFrame] = {}
    if df22 is not None: datasets["2022"] = df22
    if df23 is not None: datasets["2023"] = df23
    if df24 is not None: datasets["2024"] = df24
    if df10 is not None: datasets["2010"] = df10

    if not datasets:
        raise RuntimeError("No internal_migration_*.csv files found.")

    # Hook text uses the latest available numeric year for punchy number
    latest_year = max(int(y) for y in datasets.keys() if y.isdigit())
    annual_latest = aggregate_months(datasets[str(latest_year)], MONTHS)
    hook_line = total_movers_text(annual_latest, str(latest_year))

    # Build figures
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
        sankey_html_blocks.append(
            build_sankey(od24, "Origin–destination flows (2024)")
            .to_html(full_html=False, include_plotlyjs=False, div_id="sankey24")
        )
    od10 = read_csv_safe(DATA / "od_flows_2010.csv")
    if od10 is not None and not od10.empty:
        sankey_html_blocks.append(
            build_sankey(od10, "Origin–destination flows (2010)")
            .to_html(full_html=False, include_plotlyjs=False, div_id="sankey10")
        )
    sankey_html = "\n".join(sankey_html_blocks) if sankey_html_blocks else \
        '<p class="muted">Origin–destination flow data not found — add <code>od_flows_*.csv</code> to enable Sankey.</p>'

    df_age = read_csv_safe(DATA / "migration_by_age_2024.csv")
    age_html = build_age_bar(df_age).to_html(full_html=False, include_plotlyjs=False, div_id="agebar") \
        if (df_age is not None and not df_age.empty) \
        else '<p class="muted">Age-group breakdown not found — add <code>migration_by_age_2024.csv</code>.</p>'

    df_sex = read_csv_safe(DATA / "migration_by_sex_2024.csv")
    sex_html = build_sex_bar(df_sex).to_html(full_html=False, include_plotlyjs=False, div_id="sexbar") \
        if (df_sex is not None and not df_sex.empty) \
        else '<p class="muted">Sex breakdown not found — add <code>migration_by_sex_2024.csv</code>.</p>'

    # What-If projection (supports 2022/2023/2024 baselines)
    bases_for_projection: Dict[str, pd.DataFrame] = {}
    for y in ["2022", "2023", "2024"]:
        if y in datasets:
            bases_for_projection[y] = aggregate_months(datasets[y], MONTHS)
    proj_fig = build_projection(bases_for_projection)
    proj_html = proj_fig.to_html(full_html=False, include_plotlyjs=False, div_id="projection")

    # HTML skeleton (with patched JS)
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
  label {{ font-size:14px; }}
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
    <p class="muted">According to demographic forecasts, by 2050, approximately one-third of Kazakhstan’spopulation is expected to reside in megacities such as Astana and Almaty, while many regions,particularly peripheral areas, are anticipated to experience significant population decline (Kasenov& Nurmagambetov, 2017). Similar findings are supported by reports from the Ministry of NationalEconomy of Kazakhstan (2020) and the World Urbanization Prospects, 2018 Revision, publishedby the United Nations Department of Economic and Social Affairs (2018).
</p>
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
      <div style="margin-bottom:8px;">
        <label>Baseline year:
          <select id="yearSelect"></select>
        </label>
        <label style="margin-left:12px;">Region:
          <select id="regionSelect"></select>
        </label>
        <label style="margin-left:12px;">Annual change in net migration:
          <input id="rateInput" type="range" min="-20" max="20" value="0" step="1"/>
          <span id="rateLabel">0%</span>
        </label>
      </div>
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
  // === What-If: client-side updater using per-year base nets in layout.meta ===
  (function() {{
    const fig = document.getElementById("projection");
    if (!fig) return;
    const gd = fig;
    const meta = gd.layout.meta || {{}};
    const baseByYear = meta.base_net_by_year || {{}};
    const yearsAvail = Object.keys(baseByYear).sort();   // e.g., ["2022","2023","2024"]

    const yearSel  = document.getElementById("yearSelect");
    const regionSel= document.getElementById("regionSelect");
    const rateInput= document.getElementById("rateInput");
    const rateLabel= document.getElementById("rateLabel");
    const note     = document.getElementById("projNote");

    // Populate baseline-year dropdown (prefer 2024 if present)
    yearsAvail.forEach(y => {{
      const opt = document.createElement("option");
      opt.value = y; opt.textContent = y;
      yearSel.appendChild(opt);
    }});
    if (yearsAvail.includes("2024")) yearSel.value = "2024";

    function populateRegions() {{
      const y = yearSel.value;
      const base = baseByYear[y] || {{}};
      const regions = Object.keys(base).sort();
      regionSel.innerHTML = "";
      regions.forEach(r => {{
        const opt = document.createElement("option");
        opt.value = r; opt.textContent = r;
        regionSel.appendChild(opt);
      }});
      if (regions.includes("Astana")) regionSel.value = "Astana";
      else if (regions.length) regionSel.value = regions[0];
    }}

    function recompute() {{
      const y0Str = yearSel.value;
      const y0    = parseInt(y0Str, 10);                 // baseline year (e.g., 2022)
      const base  = baseByYear[y0Str] || {{}};
      const region= regionSel.value;
      const r     = parseFloat(rateInput.value) / 100.0; // annual % change

      // Build dynamic years from baseline to 2030
      const yearsX = Array.from({{length: Math.max(0, 2030 - y0) + 1}}, (_, i) => y0 + i);

      // Annual nets with geometric change, then cumulative since baseline
      const net0 = base[region] || 0;
      const annuals = yearsX.map((_, i) => net0 * Math.pow(1 + r, i));
      const cum = [];
      annuals.reduce((acc, v, i) => (cum[i] = acc + v, acc + v), 0);

      // Update the trace
      Plotly.restyle(gd, {{ x: [yearsX], y: [cum], name: [`${{region}} (${{y0Str}})`] }}, [0]);

      // Update title and y-axis label to reflect baseline
      Plotly.relayout(gd, {{
        title: `Simple projection of cumulative net migration — ${{region}} (${{y0Str}}) to 2030`,
        "yaxis.title.text": `Cumulative net migration since ${{y0Str}}`
      }});

      // Narrative note
      const total2030 = cum[cum.length - 1] || 0;
      rateLabel.textContent = `${{(r * 100).toFixed(0)}}%`;
      note.textContent = `${{region}} cumulative net by 2030: ${{Math.round(total2030).toLocaleString('en-US')}} people `
        + `(baseline ${{y0Str}}; slider applies ±% change to annual net).`;
    }}

    yearSel.addEventListener("change", () => {{ populateRegions(); recompute(); }});
    regionSel.addEventListener("change", recompute);
    rateInput.addEventListener("input", recompute);

    // Init
    populateRegions();
    recompute();
  }})();
</script>

</body>
</html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {OUT}")

if __name__ == "__main__":
    main()

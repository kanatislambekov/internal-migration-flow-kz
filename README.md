# Internal Migration Flow – Kazakhstan

This repository packages a lightweight, text-only version of the 2024
internal migration statistics for Kazakhstan together with a small Python
utility that renders an interactive flow map.  The previous iteration of
this project relied on spreadsheet workbooks.

## Project structure

```
├── data/
│   ├── internal_migration_2024.csv   # tidy arrivals/departures/net counts
│   └── region_centroids.csv          # lat/long coordinates for the map
├── outputs/                          # generated HTML maps (gitignored)
├── scripts/
│   └── plot_internal_migration.py    # building the map
└── requirements.txt                  # runtime dependencies
```

## Getting started

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Generate an interactive HTML map.  By default the script aggregates all
   months into one annual view and writes the output to `outputs/migration_map.html`:

   ```bash
   python scripts/plot_internal_migration.py
   ```


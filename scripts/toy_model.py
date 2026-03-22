import json
import math
import requests
from collections import defaultdict
from pathlib import Path

OUTPUT_JSON = Path("outputs/latest/toy_cells.json")

USGS_URL = (
    "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/"
    "all_month.geojson"
)


# -----------------------------
# Fetch data
# -----------------------------
def fetch_usgs():
    print("Fetching USGS data...")
    r = requests.get(USGS_URL, timeout=30)
    r.raise_for_status()
    return r.json()


# -----------------------------
# Grid function (2° x 2°)
# -----------------------------
def grid_key(lat, lon):
    lat_cell = int(lat // 2 * 2)
    lon_cell = int(lon // 2 * 2)
    return lat_cell, lon_cell


# -----------------------------
# Phase assignment
# -----------------------------
def assign_phase(score):
    if score < 0.5:
        return "stable"
    elif score < 1.0:
        return "low"
    elif score < 2.0:
        return "moderate"
    elif score < 4.0:
        return "high"
    else:
        return "critical"


# -----------------------------
# Build cells
# -----------------------------
def build_cells(data):
    cells = defaultdict(lambda: {"count": 0, "mag_sum": 0.0, "events": []})

    features = data.get("features", [])

    for feature in features:
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [])

        if len(coords) < 2:
            continue

        lon = coords[0]
        lat = coords[1]
        mag = props.get("mag")

        if mag is None:
            continue

        key = grid_key(lat, lon)

        cells[key]["count"] += 1
        cells[key]["mag_sum"] += float(mag)
        cells[key]["events"].append(
            {
                "mag": float(mag),
                "place": props.get("place", "unknown"),
                "time": props.get("time"),
                "lat": lat,
                "lon": lon,
            }
        )

    # -----------------------------
    # Compute baseline (normalization)
    # -----------------------------
    raw_scores = []
    for cell_data in cells.values():
        avg_mag = cell_data["mag_sum"] / cell_data["count"]
        score = cell_data["count"] * avg_mag
        raw_scores.append(score)

    baseline = sum(raw_scores) / len(raw_scores) if raw_scores else 1.0

    # -----------------------------
    # Build results
    # -----------------------------
    results = []

    for (lat_cell, lon_cell), cell_data in cells.items():
        avg_mag = cell_data["mag_sum"] / cell_data["count"]

        toy_score = (
            (cell_data["count"] * avg_mag) / baseline if baseline > 0 else 0.0
        )

        phase = assign_phase(toy_score)

        results.append(
            {
                "lat_cell": lat_cell,
                "lon_cell": lon_cell,
                "count": cell_data["count"],
                "avg_mag": round(avg_mag, 3),
                "toy_score": round(toy_score, 3),
                "phase": phase,
            }
        )

    # ordina per score
    results.sort(key=lambda x: x["toy_score"], reverse=True)

    return results


# -----------------------------
# Main
# -----------------------------
def main():
    data = fetch_usgs()
    cells = build_cells(data)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # salva solo top 100 celle
    OUTPUT_JSON.write_text(json.dumps(cells[:100], indent=2), encoding="utf-8")

    print(f"Saved {min(len(cells), 100)} toy model cells to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

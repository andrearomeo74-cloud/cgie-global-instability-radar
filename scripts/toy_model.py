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


def fetch_usgs():
    print("Fetching USGS data...")
    r = requests.get(USGS_URL, timeout=30)
    r.raise_for_status()
    return r.json()


def grid_key(lat, lon):
    lat_cell = int(lat // 2 * 2)
    lon_cell = int(lon // 2 * 2)
    return lat_cell, lon_cell


def percentile(sorted_values, p):
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)

    if f == c:
        return sorted_values[f]

    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def assign_phase(score, p30, p50, p70, p90):
    if score >= p90:
        return "critical"
    elif score >= p70:
        return "high"
    elif score >= p50:
        return "moderate"
    elif score >= p30:
        return "low"
    else:
        return "stable"


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

    raw_scores = []
    for cell_data in cells.values():
        avg_mag = cell_data["mag_sum"] / cell_data["count"]
        base_score = cell_data["count"] * avg_mag
        raw_scores.append(base_score)

    baseline = sum(raw_scores) / len(raw_scores) if raw_scores else 1.0

    temp_results = []
    for (lat_cell, lon_cell), cell_data in cells.items():
        avg_mag = cell_data["mag_sum"] / cell_data["count"]

        raw_score = (
            (cell_data["count"] * avg_mag) / baseline if baseline > 0 else 0.0
        )
        toy_score = math.log1p(raw_score)

        temp_results.append(
            {
                "lat_cell": lat_cell,
                "lon_cell": lon_cell,
                "count": cell_data["count"],
                "avg_mag": round(avg_mag, 3),
                "toy_score": round(toy_score, 3),
            }
        )

    scores = sorted([r["toy_score"] for r in temp_results])

    p30 = percentile(scores, 30)
    p50 = percentile(scores, 50)
    p70 = percentile(scores, 70)
    p90 = percentile(scores, 90)

    print("Percentiles:", p30, p50, p70, p90)

    results = []
    for row in temp_results:
        phase = assign_phase(row["toy_score"], p30, p50, p70, p90)
        row["phase"] = phase
        results.append(row)

    results.sort(key=lambda x: x["toy_score"], reverse=True)
    return results


def main():
    data = fetch_usgs()
    cells = build_cells(data)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(cells[:100], indent=2), encoding="utf-8")

    print(f"Saved {min(len(cells), 100)} toy model cells to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

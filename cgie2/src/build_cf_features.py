#!/usr/bin/env python3
"""
CF-RETRO-01 — Deterministic seismic feature builder.

This script reads:

    cgie2/config/cf_retro_01.yaml
    outputs/CF_RETRO_01_catalog_frozen.csv
    outputs/CF_RETRO_01_catalog_frozen.csv.sha256

and produces:

    outputs/CF_RETRO_01_features.csv
    outputs/CF_RETRO_01_features.csv.sha256
    outputs/CF_RETRO_01_features_manifest.json

The script:

1. verifies the frozen catalogue SHA-256 hash;
2. verifies the configuration and experiment identity;
3. rejects target-event and post-cutoff observations;
4. constructs an hourly UTC evaluation grid;
5. calculates strictly trailing-window features;
6. generates windows of 1, 3, 7 and 30 days;
7. records hashes, code commit and execution metadata;
8. serializes the feature table deterministically.

Run from the repository root:

    python cgie2/src/build_cf_features.py

Required packages:

    pandas
    numpy
    PyYAML
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()

REPOSITORY_ROOT = SCRIPT_PATH.parents[2]

CONFIG_PATH = (
    REPOSITORY_ROOT
    / "cgie2"
    / "config"
    / "cf_retro_01.yaml"
)


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def fail(message: str, exit_code: int = 1) -> None:
    """Print a fatal error and terminate execution."""
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def sha256_bytes(content: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def get_git_commit(repository_root: Path) -> str:
    """
    Return the current Git commit.

    UNAVAILABLE is returned when Git is absent or the script is not
    running inside a Git repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def parse_utc(value: str) -> pd.Timestamp:
    """Parse a timestamp and normalize it to UTC."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid timestamp in configuration: {value}"
        ) from exc

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp


def utc_string(value: pd.Timestamp) -> str:
    """Serialize a pandas timestamp as an ISO-8601 UTC string."""
    timestamp = value

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp.isoformat().replace("+00:00", "Z")


def resolve_repository_path(
    repository_root: Path,
    configured_path: str,
) -> Path:
    """
    Resolve a repository-relative path and prevent directory traversal.
    """
    resolved = (
        repository_root
        / Path(configured_path)
    ).resolve()

    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError:
        fail(
            "Configured path escapes the repository root: "
            f"{configured_path}"
        )

    return resolved


def load_configuration(path: Path) -> dict[str, Any]:
    """Load and validate the frozen YAML configuration."""
    if not path.exists():
        fail(
            "Configuration file not found:\n"
            f"{path}"
        )

    raw_content = path.read_bytes()

    try:
        configuration = yaml.safe_load(raw_content)
    except yaml.YAMLError as exc:
        fail(f"Invalid YAML configuration: {exc}")

    if not isinstance(configuration, dict):
        fail("Configuration root must be a YAML mapping.")

    required_sections = {
        "experiment",
        "target_event",
        "geography",
        "baseline",
        "test_period",
        "catalogue",
        "feature_engineering",
        "outputs",
        "rules",
    }

    missing_sections = sorted(
        required_sections.difference(configuration)
    )

    if missing_sections:
        fail(
            "Missing configuration sections: "
            + ", ".join(missing_sections)
        )

    experiment = configuration["experiment"]

    if experiment.get("id") != "CF_RETRO_01":
        fail("Unexpected experiment identifier.")

    if experiment.get("status") != "FROZEN":
        fail("Experiment configuration is not marked FROZEN.")

    rules = configuration["rules"]

    mandatory_false_rules = [
        "use_future_data",
        "modify_thresholds_after_run",
        "modify_radius_after_run",
        "modify_baseline_after_run",
        "modify_feature_definitions_after_run",
        "allow_post_event_features",
        "allow_target_event_in_features",
        "allow_silent_protocol_changes",
    ]

    for rule in mandatory_false_rules:
        if rules.get(rule) is not False:
            fail(
                f"Frozen safety rule must be false: {rule}"
            )

    feature_engineering = configuration["feature_engineering"]

    if feature_engineering.get("window_type") != "trailing":
        fail("CF-RETRO-01 requires trailing windows.")

    if (
        feature_engineering.get("future_data_allowed")
        is not False
    ):
        fail("Feature configuration permits future data.")

    if (
        feature_engineering.get("centered_windows_allowed")
        is not False
    ):
        fail("Centred windows must be prohibited.")

    return configuration


# ---------------------------------------------------------------------
# Hash verification
# ---------------------------------------------------------------------

def read_expected_hash(hash_path: Path) -> str:
    """
    Read a standard SHA-256 sidecar file.

    Expected format:

        <digest>  <filename>
    """
    if not hash_path.exists():
        fail(
            "Catalogue hash file not found:\n"
            f"{hash_path}"
        )

    content = hash_path.read_text(
        encoding="utf-8"
    ).strip()

    if not content:
        fail("Catalogue hash file is empty.")

    expected_hash = content.split()[0].strip().lower()

    if len(expected_hash) != 64:
        fail(
            "Invalid SHA-256 digest in catalogue hash file."
        )

    valid_characters = set("0123456789abcdef")

    if not set(expected_hash).issubset(valid_characters):
        fail(
            "Catalogue hash file contains a non-hexadecimal digest."
        )

    return expected_hash


def verify_file_hash(
    data_path: Path,
    hash_path: Path,
) -> str:
    """Verify a file against its SHA-256 sidecar."""
    if not data_path.exists():
        fail(
            "Required data file not found:\n"
            f"{data_path}"
        )

    expected_hash = read_expected_hash(hash_path)
    actual_hash = sha256_file(data_path)

    if actual_hash.lower() != expected_hash:
        fail(
            "Catalogue SHA-256 verification failed.\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}"
        )

    return actual_hash


# ---------------------------------------------------------------------
# Mathematical utilities
# ---------------------------------------------------------------------

def haversine_distance_km(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
    earth_radius_km: float,
) -> float:
    """Return great-circle distance between two points."""
    lat_1 = math.radians(latitude_1)
    lon_1 = math.radians(longitude_1)
    lat_2 = math.radians(latitude_2)
    lon_2 = math.radians(longitude_2)

    delta_lat = lat_2 - lat_1
    delta_lon = lon_2 - lon_1

    a = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat_1)
        * math.cos(lat_2)
        * math.sin(delta_lon / 2.0) ** 2
    )

    a = min(1.0, max(0.0, a))

    return (
        2.0
        * earth_radius_km
        * math.asin(math.sqrt(a))
    )


def spherical_mean_coordinates(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> tuple[float, float]:
    """
    Calculate a spherical mean coordinate.

    This avoids the principal wrap-around problem of directly averaging
    longitude values.
    """
    latitude_radians = np.radians(latitudes)
    longitude_radians = np.radians(longitudes)

    x = np.mean(
        np.cos(latitude_radians)
        * np.cos(longitude_radians)
    )

    y = np.mean(
        np.cos(latitude_radians)
        * np.sin(longitude_radians)
    )

    z = np.mean(
        np.sin(latitude_radians)
    )

    horizontal = math.sqrt(x * x + y * y)

    mean_latitude = math.degrees(
        math.atan2(z, horizontal)
    )

    mean_longitude = math.degrees(
        math.atan2(y, x)
    )

    return mean_latitude, mean_longitude


def median_absolute_deviation(
    values: np.ndarray,
) -> float:
    """
    Calculate unscaled median absolute deviation.

    The frozen configuration specifies raw MAD, not the
    normal-consistency-scaled MAD.
    """
    if len(values) == 0:
        return math.nan

    median = float(np.median(values))

    return float(
        np.median(
            np.abs(values - median)
        )
    )


def seismic_energy_joule(
    magnitudes: np.ndarray,
) -> np.ndarray:
    """
    Convert magnitude to a radiated-energy proxy in joules.

    Frozen transformation:

        log10(E) = 1.5 * M + 4.8
    """
    return np.power(
        10.0,
        1.5 * magnitudes + 4.8,
    )


def temporal_burstiness(
    intervals_hours: np.ndarray,
) -> float:
    """
    Calculate temporal burstiness:

        B = (sigma - mean) / (sigma + mean)

    Population standard deviation is used deterministically.
    """
    if len(intervals_hours) < 2:
        return math.nan

    mean_interval = float(
        np.mean(intervals_hours)
    )

    standard_deviation = float(
        np.std(intervals_hours, ddof=0)
    )

    denominator = (
        standard_deviation
        + mean_interval
    )

    if denominator == 0.0:
        return math.nan

    return (
        standard_deviation
        - mean_interval
    ) / denominator


# ---------------------------------------------------------------------
# Catalogue loading and quality control
# ---------------------------------------------------------------------

def load_catalogue(
    path: Path,
    configuration: dict[str, Any],
) -> pd.DataFrame:
    """Load and strictly validate the frozen earthquake catalogue."""
    try:
        catalogue = pd.read_csv(
            path,
            encoding="utf-8",
        )
    except Exception as exc:
        fail(f"Unable to read frozen catalogue: {exc}")

    if catalogue.empty:
        fail("Frozen catalogue is empty.")

    required_columns = set(
        configuration["required_variables"]
    )

    missing_columns = sorted(
        required_columns.difference(catalogue.columns)
    )

    if missing_columns:
        fail(
            "Frozen catalogue is missing required columns: "
            + ", ".join(missing_columns)
        )

    working = catalogue.copy()

    if working["event_id"].isna().any():
        fail("Catalogue contains missing event identifiers.")

    working["event_id"] = (
        working["event_id"]
        .astype(str)
        .str.strip()
    )

    if working["event_id"].eq("").any():
        fail("Catalogue contains empty event identifiers.")

    if working["event_id"].duplicated().any():
        duplicates = (
            working.loc[
                working["event_id"].duplicated(
                    keep=False
                ),
                "event_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        fail(
            "Catalogue contains duplicate event identifiers: "
            + ", ".join(duplicates[:10])
        )

    working["time_utc"] = pd.to_datetime(
        working["time_utc"],
        utc=True,
        errors="coerce",
    )

    if working["time_utc"].isna().any():
        fail("Catalogue contains invalid timestamps.")

    numeric_columns = [
        "latitude",
        "longitude",
        "depth_km",
        "magnitude",
        "distance_from_reference_km",
    ]

    for column in numeric_columns:
        working[column] = pd.to_numeric(
            working[column],
            errors="coerce",
        )

        if working[column].isna().any():
            fail(
                "Catalogue contains invalid numeric values in "
                f"{column}."
            )

    if not working["time_utc"].is_monotonic_increasing:
        fail(
            "Catalogue timestamps are not monotonically increasing."
        )

    baseline_start = parse_utc(
        configuration["baseline"]["start"]
    )

    cutoff = parse_utc(
        configuration["target_event"]["cutoff_time"]
    )

    target_event = parse_utc(
        configuration["target_event"]["utc_time"]
    )

    if cutoff >= target_event:
        fail(
            "Frozen cutoff is not earlier than the target event."
        )

    if (working["time_utc"] < baseline_start).any():
        fail(
            "Catalogue contains observations before the "
            "frozen acquisition interval."
        )

    if (working["time_utc"] > cutoff).any():
        fail(
            "Catalogue contains observations after the "
            "frozen cutoff."
        )

    if (working["time_utc"] >= target_event).any():
        fail(
            "Catalogue contains the target event or later data."
        )

    radius_km = float(
        configuration["geography"]["radius_km"]
    )

    if (
        working["distance_from_reference_km"]
        > radius_km + 1e-6
    ).any():
        fail(
            "Catalogue contains observations outside the "
            "frozen geographic radius."
        )

    if not np.isfinite(
        working[numeric_columns].to_numpy(
            dtype=float
        )
    ).all():
        fail(
            "Catalogue contains non-finite numeric values."
        )

    working = working.sort_values(
        by=["time_utc", "event_id"],
        kind="stable",
    ).reset_index(drop=True)

    return working


# ---------------------------------------------------------------------
# Evaluation grid
# ---------------------------------------------------------------------

def build_evaluation_grid(
    configuration: dict[str, Any],
) -> pd.DatetimeIndex:
    """
    Construct the frozen hourly UTC evaluation grid.

    The final frozen cutoff is included as an additional endpoint when
    it does not fall exactly on the hourly grid.
    """
    grid_config = (
        configuration["feature_engineering"]
        ["evaluation_grid"]
    )

    frequency = str(
        grid_config["frequency"]
    )

    timezone_name = str(
        grid_config["timezone"]
    )

    if timezone_name.upper() != "UTC":
        fail("CF-RETRO-01 evaluation grid must use UTC.")

    if frequency != "1h":
        fail(
            "CF-RETRO-01 v1.1 requires an hourly grid."
        )

    baseline_start = parse_utc(
        configuration["baseline"]["start"]
    )

    cutoff = parse_utc(
        configuration["target_event"]["cutoff_time"]
    )

    alignment = str(
        grid_config.get("alignment", "floor")
    )

    if alignment != "floor":
        fail(
            "CF-RETRO-01 requires floor-aligned grid endpoints."
        )

    first_endpoint = baseline_start.floor(frequency)
    final_regular_endpoint = cutoff.floor(frequency)

    grid = pd.date_range(
        start=first_endpoint,
        end=final_regular_endpoint,
        freq=frequency,
        tz="UTC",
    )

    include_cutoff = bool(
        grid_config.get(
            "include_frozen_cutoff",
            True,
        )
    )

    if (
        include_cutoff
        and cutoff not in grid
    ):
        grid = grid.append(
            pd.DatetimeIndex([cutoff])
        )

    grid = pd.DatetimeIndex(
        sorted(set(grid))
    )

    if len(grid) == 0:
        fail("Evaluation grid is empty.")

    if grid.max() > cutoff:
        fail(
            "Evaluation grid extends beyond the frozen cutoff."
        )

    if not grid.is_monotonic_increasing:
        fail(
            "Evaluation grid is not monotonically increasing."
        )

    return grid


# ---------------------------------------------------------------------
# Window feature calculation
# ---------------------------------------------------------------------

FEATURE_NAMES = [
    "event_count",
    "maximum_magnitude",
    "median_magnitude",
    "cumulative_energy_joule",
    "log10_cumulative_energy_joule",
    "median_depth_km",
    "depth_mad_km",
    "centroid_latitude",
    "centroid_longitude",
    "spatial_dispersion_km",
    "median_interevent_time_hours",
    "interevent_time_mad_hours",
    "temporal_burstiness",
]


def calculate_window_features(
    window_frame: pd.DataFrame,
    earth_radius_km: float,
    minimum_depth_events: int,
    minimum_spatial_events: int,
    minimum_interevent_events: int,
    minimum_burstiness_intervals: int,
) -> dict[str, float | int]:
    """Calculate the complete frozen feature set for one window."""
    event_count = len(window_frame)

    result: dict[str, float | int] = {
        "event_count": int(event_count),
        "maximum_magnitude": math.nan,
        "median_magnitude": math.nan,
        "cumulative_energy_joule": 0.0,
        "log10_cumulative_energy_joule": math.nan,
        "median_depth_km": math.nan,
        "depth_mad_km": math.nan,
        "centroid_latitude": math.nan,
        "centroid_longitude": math.nan,
        "spatial_dispersion_km": math.nan,
        "median_interevent_time_hours": math.nan,
        "interevent_time_mad_hours": math.nan,
        "temporal_burstiness": math.nan,
    }

    if event_count == 0:
        return result

    magnitudes = window_frame[
        "magnitude"
    ].to_numpy(dtype=float)

    depths = window_frame[
        "depth_km"
    ].to_numpy(dtype=float)

    latitudes = window_frame[
        "latitude"
    ].to_numpy(dtype=float)

    longitudes = window_frame[
        "longitude"
    ].to_numpy(dtype=float)

    energies = seismic_energy_joule(
        magnitudes
    )

    cumulative_energy = float(
        np.sum(energies)
    )

    result["maximum_magnitude"] = float(
        np.max(magnitudes)
    )

    result["median_magnitude"] = float(
        np.median(magnitudes)
    )

    result["cumulative_energy_joule"] = (
        cumulative_energy
    )

    if cumulative_energy > 0.0:
        result[
            "log10_cumulative_energy_joule"
        ] = float(
            math.log10(cumulative_energy)
        )

    result["median_depth_km"] = float(
        np.median(depths)
    )

    if event_count >= minimum_depth_events:
        result["depth_mad_km"] = (
            median_absolute_deviation(depths)
        )

    centroid_latitude, centroid_longitude = (
        spherical_mean_coordinates(
            latitudes,
            longitudes,
        )
    )

    result["centroid_latitude"] = (
        centroid_latitude
    )

    result["centroid_longitude"] = (
        centroid_longitude
    )

    if event_count >= minimum_spatial_events:
        spatial_distances = np.asarray(
            [
                haversine_distance_km(
                    centroid_latitude,
                    centroid_longitude,
                    float(latitude),
                    float(longitude),
                    earth_radius_km,
                )
                for latitude, longitude in zip(
                    latitudes,
                    longitudes,
                )
            ],
            dtype=float,
        )

        result["spatial_dispersion_km"] = float(
            np.median(spatial_distances)
        )

    if event_count >= minimum_interevent_events:
        event_times_ns = (
            window_frame["time_utc"]
            .astype("int64")
            .to_numpy(dtype=np.int64)
        )

        intervals_hours = (
            np.diff(event_times_ns)
            / 3_600_000_000_000.0
        )

        if len(intervals_hours) >= 1:
            result[
                "median_interevent_time_hours"
            ] = float(
                np.median(intervals_hours)
            )

            result[
                "interevent_time_mad_hours"
            ] = median_absolute_deviation(
                intervals_hours
            )

        if (
            len(intervals_hours)
            >= minimum_burstiness_intervals
        ):
            result["temporal_burstiness"] = (
                temporal_burstiness(
                    intervals_hours
                )
            )

    return result


def build_feature_table(
    catalogue: pd.DataFrame,
    grid: pd.DatetimeIndex,
    configuration: dict[str, Any],
) -> pd.DataFrame:
    """
    Calculate features for all endpoints and all frozen windows.

    Each window follows:

        (endpoint - duration, endpoint]

    Therefore, no event after the endpoint can enter the calculation.
    """
    feature_config = (
        configuration["feature_engineering"]
    )

    windows = feature_config["windows"]

    if not isinstance(windows, list) or not windows:
        fail("No feature windows are configured.")

    expected_window_ids = [
        "1d",
        "3d",
        "7d",
        "30d",
    ]

    configured_window_ids = [
        str(window["id"])
        for window in windows
    ]

    if configured_window_ids != expected_window_ids:
        fail(
            "Unexpected window order or identifiers.\n"
            f"Expected: {expected_window_ids}\n"
            f"Found:    {configured_window_ids}"
        )

    earth_radius_km = float(
        configuration["geography"]["earth_radius_km"]
    )

    minimum_depth_events = int(
        feature_config["depth"]
        ["minimum_events_for_dispersion"]
    )

    minimum_spatial_events = int(
        feature_config["spatial"]
        ["minimum_events_for_dispersion"]
    )

    minimum_interevent_events = int(
        feature_config["interevent_time"]
        ["minimum_events"]
    )

    minimum_burstiness_intervals = int(
        feature_config["temporal_burstiness"]
        ["minimum_intervals"]
    )

    event_times = catalogue[
        "time_utc"
    ].to_numpy(
        dtype="datetime64[ns]"
    )

    records: list[dict[str, Any]] = []

    total_rows = len(grid) * len(windows)
    completed_rows = 0
    progress_step = max(1, total_rows // 20)

    for endpoint in grid:
        endpoint_np = endpoint.to_datetime64()

        right_index = int(
            np.searchsorted(
                event_times,
                endpoint_np,
                side="right",
            )
        )

        for window in windows:
            window_id = str(window["id"])
            duration_hours = int(
                window["duration_hours"]
            )

            window_start = (
                endpoint
                - pd.Timedelta(
                    hours=duration_hours
                )
            )

            window_start_np = (
                window_start.to_datetime64()
            )

            left_index = int(
                np.searchsorted(
                    event_times,
                    window_start_np,
                    side="right",
                )
            )

            window_frame = catalogue.iloc[
                left_index:right_index
            ]

            if not window_frame.empty:
                if (
                    window_frame["time_utc"].min()
                    <= window_start
                ):
                    fail(
                        "Window interval safety check failed: "
                        "left boundary is not open."
                    )

                if (
                    window_frame["time_utc"].max()
                    > endpoint
                ):
                    fail(
                        "Future observation entered a trailing window."
                    )

            features = calculate_window_features(
                window_frame=window_frame,
                earth_radius_km=earth_radius_km,
                minimum_depth_events=(
                    minimum_depth_events
                ),
                minimum_spatial_events=(
                    minimum_spatial_events
                ),
                minimum_interevent_events=(
                    minimum_interevent_events
                ),
                minimum_burstiness_intervals=(
                    minimum_burstiness_intervals
                ),
            )

            record: dict[str, Any] = {
                "experiment_id": "CF_RETRO_01",
                "endpoint_utc": utc_string(endpoint),
                "window_id": window_id,
                "window_duration_hours":
                    duration_hours,
                "window_start_exclusive_utc":
                    utc_string(window_start),
                "window_end_inclusive_utc":
                    utc_string(endpoint),
            }

            record.update(features)

            records.append(record)

            completed_rows += 1

            if (
                completed_rows % progress_step == 0
                or completed_rows == total_rows
            ):
                percentage = (
                    100.0
                    * completed_rows
                    / total_rows
                )

                print(
                    "Feature rows completed: "
                    f"{completed_rows}/{total_rows} "
                    f"({percentage:.1f}%)"
                )

    feature_table = pd.DataFrame.from_records(
        records
    )

    expected_columns = [
        "experiment_id",
        "endpoint_utc",
        "window_id",
        "window_duration_hours",
        "window_start_exclusive_utc",
        "window_end_inclusive_utc",
        *FEATURE_NAMES,
    ]

    feature_table = feature_table[
        expected_columns
    ]

    if len(feature_table) != total_rows:
        fail(
            "Unexpected feature-table row count."
        )

    return feature_table


# ---------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------

def validate_feature_table(
    feature_table: pd.DataFrame,
    grid: pd.DatetimeIndex,
    configuration: dict[str, Any],
) -> None:
    """Run deterministic safety checks before writing outputs."""
    if feature_table.empty:
        fail("Generated feature table is empty.")

    expected_rows = (
        len(grid)
        * len(
            configuration["feature_engineering"]
            ["windows"]
        )
    )

    if len(feature_table) != expected_rows:
        fail(
            "Generated feature table has an unexpected row count.\n"
            f"Expected: {expected_rows}\n"
            f"Found:    {len(feature_table)}"
        )

    endpoints = pd.to_datetime(
        feature_table["endpoint_utc"],
        utc=True,
        errors="coerce",
    )

    if endpoints.isna().any():
        fail(
            "Generated feature table contains invalid endpoints."
        )

    cutoff = parse_utc(
        configuration["target_event"]["cutoff_time"]
    )

    target_event = parse_utc(
        configuration["target_event"]["utc_time"]
    )

    if endpoints.max() > cutoff:
        fail(
            "Generated features extend past the frozen cutoff."
        )

    if (endpoints >= target_event).any():
        fail(
            "Generated features include the target event or later time."
        )

    event_counts = pd.to_numeric(
        feature_table["event_count"],
        errors="coerce",
    )

    if event_counts.isna().any():
        fail(
            "Generated feature table contains invalid event counts."
        )

    if (event_counts < 0).any():
        fail(
            "Generated feature table contains negative event counts."
        )

    cumulative_energy = pd.to_numeric(
        feature_table["cumulative_energy_joule"],
        errors="coerce",
    )

    if cumulative_energy.isna().any():
        fail(
            "Cumulative energy must be zero, not missing, "
            "for empty windows."
        )

    if (cumulative_energy < 0.0).any():
        fail(
            "Generated feature table contains negative energy."
        )

    zero_event_rows = event_counts == 0

    if (
        cumulative_energy.loc[zero_event_rows]
        != 0.0
    ).any():
        fail(
            "An empty window contains non-zero cumulative energy."
        )

    nonzero_event_rows = event_counts > 0

    if (
        cumulative_energy.loc[nonzero_event_rows]
        <= 0.0
    ).any():
        fail(
            "A non-empty window contains non-positive energy."
        )

    duplicates = feature_table.duplicated(
        subset=[
            "endpoint_utc",
            "window_id",
        ],
        keep=False,
    )

    if duplicates.any():
        fail(
            "Generated feature table contains duplicate "
            "endpoint-window combinations."
        )

    window_order = {
        "1d": 0,
        "3d": 1,
        "7d": 2,
        "30d": 3,
    }

    order_check = feature_table.copy()

    order_check["_endpoint"] = pd.to_datetime(
        order_check["endpoint_utc"],
        utc=True,
    )

    order_check["_window_order"] = (
        order_check["window_id"]
        .map(window_order)
    )

    sorted_index = (
        order_check
        .sort_values(
            by=[
                "_endpoint",
                "_window_order",
            ],
            kind="stable",
        )
        .index
    )

    if not sorted_index.equals(
        feature_table.index
    ):
        fail(
            "Generated feature table is not deterministically sorted."
        )


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 76)
    print("CF-RETRO-01 — DETERMINISTIC FEATURE ENGINEERING")
    print("=" * 76)

    configuration = load_configuration(
        CONFIG_PATH
    )

    configuration_hash = sha256_bytes(
        CONFIG_PATH.read_bytes()
    )

    catalogue_path = resolve_repository_path(
        REPOSITORY_ROOT,
        configuration["catalogue"]["input_file"],
    )

    catalogue_hash_path = resolve_repository_path(
        REPOSITORY_ROOT,
        configuration["catalogue"]["hash_file"],
    )

    features_path = resolve_repository_path(
        REPOSITORY_ROOT,
        configuration["outputs"]["features"],
    )

    features_hash_path = resolve_repository_path(
        REPOSITORY_ROOT,
        configuration["outputs"]["features_hash"],
    )

    manifest_path = resolve_repository_path(
        REPOSITORY_ROOT,
        configuration["outputs"]["features_manifest"],
    )

    print(f"Configuration: {CONFIG_PATH}")
    print(
        f"Configuration SHA-256: {configuration_hash}"
    )
    print(f"Catalogue: {catalogue_path}")

    catalogue_hash = verify_file_hash(
        data_path=catalogue_path,
        hash_path=catalogue_hash_path,
    )

    print(
        "Catalogue SHA-256 verification passed: "
        f"{catalogue_hash}"
    )

    catalogue = load_catalogue(
        path=catalogue_path,
        configuration=configuration,
    )

    print(f"Catalogue rows: {len(catalogue)}")
    print(
        "Catalogue interval: "
        f"{utc_string(catalogue['time_utc'].min())} "
        "through "
        f"{utc_string(catalogue['time_utc'].max())}"
    )

    grid = build_evaluation_grid(
        configuration
    )

    print(f"Evaluation endpoints: {len(grid)}")
    print(
        "First endpoint: "
        f"{utc_string(grid.min())}"
    )
    print(
        "Final endpoint: "
        f"{utc_string(grid.max())}"
    )

    configured_windows = (
        configuration["feature_engineering"]
        ["windows"]
    )

    print(
        "Frozen windows: "
        + ", ".join(
            str(window["id"])
            for window in configured_windows
        )
    )

    print()

    feature_table = build_feature_table(
        catalogue=catalogue,
        grid=grid,
        configuration=configuration,
    )

    validate_feature_table(
        feature_table=feature_table,
        grid=grid,
        configuration=configuration,
    )

    features_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    float_precision = int(
        configuration["reproducibility"]
        ["float_precision"]
    )

    feature_table.to_csv(
        features_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format=f"%.{float_precision}g",
        na_rep="",
    )

    features_hash = sha256_file(
        features_path
    )

    features_hash_path.write_text(
        f"{features_hash}  {features_path.name}\n",
        encoding="utf-8",
    )

    execution_time = datetime.now(
        timezone.utc
    )

    window_statistics: dict[str, Any] = {}

    for window_id, group in feature_table.groupby(
        "window_id",
        sort=False,
    ):
        event_counts = pd.to_numeric(
            group["event_count"],
            errors="raise",
        )

        nonempty_rows = int(
            (event_counts > 0).sum()
        )

        window_statistics[str(window_id)] = {
            "rows": int(len(group)),
            "nonempty_rows": nonempty_rows,
            "empty_rows": int(
                len(group) - nonempty_rows
            ),
            "maximum_event_count": int(
                event_counts.max()
            ),
            "median_event_count": float(
                event_counts.median()
            ),
        }

    manifest = {
        "experiment_id":
            configuration["experiment"]["id"],
        "experiment_version":
            configuration["experiment"]["version"],
        "protocol_status":
            configuration["experiment"]["status"],
        "protocol_version":
            configuration["protocol"]
            ["protocol_version"],
        "configuration_version":
            configuration["protocol"]
            ["configuration_version"],
        "execution_timestamp_utc":
            execution_time
            .isoformat()
            .replace("+00:00", "Z"),
        "code_commit":
            get_git_commit(REPOSITORY_ROOT),
        "configuration_file": str(
            CONFIG_PATH.relative_to(
                REPOSITORY_ROOT
            )
        ),
        "configuration_sha256":
            configuration_hash,
        "input_catalogue_file": str(
            catalogue_path.relative_to(
                REPOSITORY_ROOT
            )
        ),
        "input_catalogue_sha256":
            catalogue_hash,
        "output_features_file": str(
            features_path.relative_to(
                REPOSITORY_ROOT
            )
        ),
        "output_features_sha256":
            features_hash,
        "catalogue_rows":
            int(len(catalogue)),
        "evaluation_grid": {
            "frequency":
                configuration[
                    "feature_engineering"
                ]["evaluation_grid"]["frequency"],
            "timezone": "UTC",
            "first_endpoint_utc":
                utc_string(grid.min()),
            "final_endpoint_utc":
                utc_string(grid.max()),
            "endpoint_count":
                int(len(grid)),
        },
        "window_definition":
            configuration[
                "feature_engineering"
            ]["interval_definition"],
        "windows":
            configured_windows,
        "feature_names":
            FEATURE_NAMES,
        "feature_row_count":
            int(len(feature_table)),
        "window_statistics":
            window_statistics,
        "data_cutoff_utc":
            configuration["target_event"]
            ["cutoff_time"],
        "target_event_utc":
            configuration["target_event"]
            ["utc_time"],
        "future_data_allowed": False,
        "target_event_included": False,
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_hash = sha256_file(
        manifest_path
    )

    print()
    print("=" * 76)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 76)
    print(f"Feature rows: {len(feature_table)}")
    print(
        "Unique endpoints: "
        f"{feature_table['endpoint_utc'].nunique()}"
    )
    print(
        "Window types: "
        f"{feature_table['window_id'].nunique()}"
    )
    print()
    print(f"Features: {features_path}")
    print(f"Features SHA-256: {features_hash}")
    print(f"Hash file: {features_hash_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Manifest SHA-256: {manifest_hash}")
    print()
    print(
        "Safety status: no endpoint exceeds the frozen cutoff, "
        "and no target-event or future observation was used."
    )


if __name__ == "__main__":
    main()
           

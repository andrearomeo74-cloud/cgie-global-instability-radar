#!/usr/bin/env python3
"""
CF-RETRO-01 — INGV earthquake catalogue downloader.

This script:

1. reads the frozen configuration from:
   cgie2/config/cf_retro_01.yaml

2. downloads earthquake data from the official INGV FDSN event service;

3. uses the complete interval:
   2025-01-01 00:00:00 UTC
   through
   2026-07-31 17:46:42 UTC;

4. applies the frozen 15 km radius around the configured reference point;

5. excludes the target earthquake and every later event;

6. validates, sorts and deduplicates the catalogue;

7. produces:
   outputs/CF_RETRO_01_catalog_frozen.csv
   outputs/CF_RETRO_01_catalog_frozen.sha256
   outputs/CF_RETRO_01_catalog_manifest.json

The script must be executed from the repository root:

    python cgie2/src/download_ingv_catalog.py

Required packages:

    requests
    pandas
    PyYAML
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml


# ---------------------------------------------------------------------
# Frozen service settings
# ---------------------------------------------------------------------

INGV_ENDPOINT = "https://webservices.ingv.it/fdsnws/event/1/query"

REQUEST_FORMAT = "text"

REQUEST_TIMEOUT_SECONDS = 60

MAX_RETRIES = 5

RETRY_BACKOFF_SECONDS = 3

CHUNK_DAYS = 30

USER_AGENT = (
    "CGIE-2.0-CF-RETRO-01/1.0 "
    "(retrospective scientific validation; public INGV data)"
)


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
# Utility functions
# ---------------------------------------------------------------------

def fail(message: str, exit_code: int = 1) -> None:
    """Print a fatal error and terminate execution."""
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def parse_utc(value: str) -> datetime:
    """
    Parse an ISO-8601 timestamp and return a timezone-aware UTC datetime.
    """
    normalized = value.strip()

    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO timestamp: {value}") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def utc_iso(value: datetime) -> str:
    """Return a UTC timestamp accepted by the FDSN service."""
    return value.astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )


def sha256_bytes(content: bytes) -> str:
    """Calculate a SHA-256 digest from bytes."""
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 digest from a file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def get_git_commit(repository_root: Path) -> str:
    """
    Return the current Git commit.

    The value UNAVAILABLE is returned when the script is executed
    outside a Git repository or Git is not installed.
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


def load_configuration(path: Path) -> dict[str, Any]:
    """Load and minimally validate the frozen YAML configuration."""
    if not path.exists():
        fail(
            "Configuration file not found:\n"
            f"{path}\n\n"
            "Create cgie2/config/cf_retro_01.yaml first."
        )

    raw_content = path.read_bytes()

    try:
        configuration = yaml.safe_load(raw_content)
    except yaml.YAMLError as exc:
        fail(f"Invalid YAML configuration: {exc}")

    if not isinstance(configuration, dict):
        fail("The YAML configuration must contain a mapping.")

    required_sections = [
        "experiment",
        "target_event",
        "geography",
        "baseline",
        "test_period",
        "outputs",
        "rules",
    ]

    missing_sections = [
        section
        for section in required_sections
        if section not in configuration
    ]

    if missing_sections:
        fail(
            "Missing configuration sections: "
            + ", ".join(missing_sections)
        )

    if configuration["experiment"].get("id") != "CF_RETRO_01":
        fail("Unexpected experiment ID in configuration.")

    if configuration["experiment"].get("status") != "FROZEN":
        fail("The experiment configuration is not marked FROZEN.")

    if configuration["rules"].get("use_future_data") is not False:
        fail("The configuration does not prohibit future data.")

    if (
        configuration["rules"].get("allow_post_event_features")
        is not False
    ):
        fail("The configuration does not prohibit post-event features.")

    return configuration


def resolve_output_path(
    repository_root: Path,
    configured_path: str,
) -> Path:
    """
    Resolve a configured repository-relative output path safely.
    """
    output_path = (
        repository_root
        / Path(configured_path)
    ).resolve()

    try:
        output_path.relative_to(repository_root.resolve())
    except ValueError:
        fail(
            "Configured output path escapes the repository root: "
            f"{configured_path}"
        )

    return output_path


def generate_time_chunks(
    start: datetime,
    end: datetime,
    chunk_days: int,
) -> list[tuple[datetime, datetime]]:
    """
    Divide an inclusive time interval into non-overlapping chunks.

    One second is added between chunks to avoid duplicate boundary
    timestamps. Catalogue records are deduplicated again after download.
    """
    if start > end:
        raise ValueError("Start time occurs after end time.")

    chunks: list[tuple[datetime, datetime]] = []

    current_start = start

    while current_start <= end:
        current_end = min(
            current_start + timedelta(days=chunk_days),
            end,
        )

        chunks.append((current_start, current_end))

        current_start = current_end + timedelta(seconds=1)

    return chunks


def request_chunk(
    session: requests.Session,
    start: datetime,
    end: datetime,
    latitude: float,
    longitude: float,
    radius_km: float,
) -> str:
    """
    Download one catalogue chunk with retries.
    """
    parameters = {
        "starttime": utc_iso(start),
        "endtime": utc_iso(end),
        "latitude": f"{latitude:.6f}",
        "longitude": f"{longitude:.6f}",
        "maxradiuskm": f"{radius_km:.3f}",
        "format": REQUEST_FORMAT,
        "orderby": "time-asc",
    }

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                INGV_ENDPOINT,
                params=parameters,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            # FDSN services may return HTTP 204 when no events exist.
            if response.status_code == 204:
                return ""

            response.raise_for_status()

            return response.text

        except requests.RequestException as exc:
            last_error = exc

            if attempt == MAX_RETRIES:
                break

            wait_seconds = RETRY_BACKOFF_SECONDS * attempt

            print(
                f"  Request failed on attempt {attempt}/"
                f"{MAX_RETRIES}: {exc}"
            )
            print(
                f"  Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        "INGV request failed after "
        f"{MAX_RETRIES} attempts: {last_error}"
    )


# ---------------------------------------------------------------------
# INGV/FDSN parsing
# ---------------------------------------------------------------------

COLUMN_ALIASES = {
    "eventid": "event_id",
    "event_id": "event_id",
    "time": "time",
    "latitude": "latitude",
    "longitude": "longitude",
    "depth/km": "depth",
    "depth": "depth",
    "magtype": "magnitude_type",
    "magnitude_type": "magnitude_type",
    "magnitude": "magnitude",
    "eventlocationname": "location_name",
    "location_name": "location_name",
    "author": "author",
    "catalog": "catalog",
    "contributor": "contributor",
    "contributorid": "contributor_id",
    "magauthor": "magnitude_author",
}


def normalize_header(value: str) -> str:
    """Normalize an INGV/FDSN column name."""
    cleaned = value.strip().lstrip("#").strip()

    compact = (
        cleaned.lower()
        .replace(" ", "")
        .replace("-", "")
    )

    return COLUMN_ALIASES.get(
        compact,
        cleaned.lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_"),
    )


def parse_fdsn_text(payload: str) -> pd.DataFrame:
    """
    Parse the pipe-delimited FDSN text response.

    Expected standard fields include EventID, Time, Latitude,
    Longitude, Depth/Km, MagType and Magnitude.
    """
    stripped = payload.strip()

    if not stripped:
        return pd.DataFrame()

    lines = [
        line.strip()
        for line in stripped.splitlines()
        if line.strip()
    ]

    if not lines:
        return pd.DataFrame()

    header_index: int | None = None

    for index, line in enumerate(lines):
        candidate = line.lstrip("#").strip().lower()

        if (
            "eventid" in candidate
            and "time" in candidate
            and "latitude" in candidate
            and "|" in candidate
        ):
            header_index = index
            break

    if header_index is None:
        preview = "\n".join(lines[:5])
        raise ValueError(
            "Unable to find the FDSN header in the INGV response.\n"
            f"Response preview:\n{preview}"
        )

    raw_header = lines[header_index].lstrip("#").strip()

    columns = [
        normalize_header(column)
        for column in raw_header.split("|")
    ]

    records: list[list[str]] = []

    for line in lines[header_index + 1:]:
        if line.startswith("#"):
            continue

        values = [
            value.strip()
            for value in line.split("|")
        ]

        if len(values) != len(columns):
            raise ValueError(
                "Unexpected number of fields in catalogue row.\n"
                f"Expected: {len(columns)}\n"
                f"Received: {len(values)}\n"
                f"Row: {line}"
            )

        records.append(values)

    return pd.DataFrame(records, columns=columns)


# ---------------------------------------------------------------------
# Catalogue validation and normalization
# ---------------------------------------------------------------------

def haversine_distance_km(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    """Calculate great-circle distance between two coordinates."""
    earth_radius_km = 6371.0088

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

    return (
        2.0
        * earth_radius_km
        * math.asin(math.sqrt(a))
    )


def normalize_catalogue(
    frame: pd.DataFrame,
    cutoff: datetime,
    centre_latitude: float,
    centre_longitude: float,
    radius_km: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Normalize, validate and freeze the downloaded catalogue.
    """
    if frame.empty:
        fail(
            "The downloaded catalogue is empty. "
            "Check the service response and configured interval."
        )

    required_columns = {
        "event_id",
        "time",
        "latitude",
        "longitude",
        "depth",
        "magnitude",
        "magnitude_type",
    }

    missing_columns = sorted(
        required_columns.difference(frame.columns)
    )

    if missing_columns:
        fail(
            "Downloaded catalogue is missing required columns: "
            + ", ".join(missing_columns)
        )

    working = frame.copy()

    initial_rows = len(working)

    working["event_id"] = (
        working["event_id"]
        .astype(str)
        .str.strip()
    )

    working["time_utc"] = pd.to_datetime(
        working["time"],
        utc=True,
        errors="coerce",
    )

    numeric_columns = [
        "latitude",
        "longitude",
        "depth",
        "magnitude",
    ]

    for column in numeric_columns:
        working[column] = pd.to_numeric(
            working[column],
            errors="coerce",
        )

    invalid_required = (
        working["event_id"].eq("")
        | working["time_utc"].isna()
        | working["latitude"].isna()
        | working["longitude"].isna()
        | working["depth"].isna()
        | working["magnitude"].isna()
    )

    invalid_rows_removed = int(invalid_required.sum())

    working = working.loc[~invalid_required].copy()

    cutoff_timestamp = pd.Timestamp(cutoff)

    post_cutoff = working["time_utc"] > cutoff_timestamp

    post_cutoff_rows_removed = int(post_cutoff.sum())

    working = working.loc[~post_cutoff].copy()

    working["distance_from_reference_km"] = [
        haversine_distance_km(
            centre_latitude,
            centre_longitude,
            float(latitude),
            float(longitude),
        )
        for latitude, longitude in zip(
            working["latitude"],
            working["longitude"],
        )
    ]

    outside_radius = (
        working["distance_from_reference_km"]
        > radius_km + 1e-6
    )

    outside_radius_rows_removed = int(outside_radius.sum())

    working = working.loc[~outside_radius].copy()

    duplicate_mask = working.duplicated(
        subset=["event_id"],
        keep="first",
    )

    duplicate_rows_removed = int(duplicate_mask.sum())

    working = working.loc[~duplicate_mask].copy()

    working["time_utc"] = (
        working["time_utc"]
        .dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        .str.replace(
            r"\.000000Z$",
            "Z",
            regex=True,
        )
        .str.replace(
            r"(\.\d{3})\d+Z$",
            r"\1Z",
            regex=True,
        )
    )

    working["source"] = "INGV_FDSN_EVENT_SERVICE"

    optional_columns = [
        "author",
        "catalog",
        "contributor",
        "contributor_id",
        "magnitude_author",
        "location_name",
    ]

    for column in optional_columns:
        if column not in working.columns:
            working[column] = ""

    final_columns = [
        "event_id",
        "time_utc",
        "latitude",
        "longitude",
        "depth",
        "magnitude",
        "magnitude_type",
        "distance_from_reference_km",
        "author",
        "catalog",
        "contributor",
        "contributor_id",
        "magnitude_author",
        "location_name",
        "source",
    ]

    working = working[final_columns]

    working = working.rename(
        columns={
            "depth": "depth_km",
        }
    )

    working = working.sort_values(
        by=["time_utc", "event_id"],
        kind="stable",
    ).reset_index(drop=True)

    statistics = {
        "downloaded_rows": initial_rows,
        "invalid_rows_removed": invalid_rows_removed,
        "post_cutoff_rows_removed": post_cutoff_rows_removed,
        "outside_radius_rows_removed":
            outside_radius_rows_removed,
        "duplicate_rows_removed": duplicate_rows_removed,
        "final_rows": len(working),
    }

    return working, statistics


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("CF-RETRO-01 — FROZEN INGV CATALOGUE ACQUISITION")
    print("=" * 72)

    configuration = load_configuration(CONFIG_PATH)

    configuration_bytes = CONFIG_PATH.read_bytes()
    configuration_hash = sha256_bytes(configuration_bytes)

    baseline_start = parse_utc(
        configuration["baseline"]["start"]
        + "T00:00:00Z"
    )

    test_end = parse_utc(
        configuration["test_period"]["end"]
    )

    cutoff = parse_utc(
        configuration["target_event"]["cutoff_time"]
    )

    target_event_time = parse_utc(
        configuration["target_event"]["utc_time"]
    )

    if test_end != cutoff:
        fail(
            "The test-period end does not match the frozen cutoff.\n"
            f"Test end: {test_end.isoformat()}\n"
            f"Cutoff:   {cutoff.isoformat()}"
        )

    if cutoff >= target_event_time:
        fail(
            "The frozen cutoff must occur before the target event."
        )

    geography = configuration["geography"]

    centre_latitude = float(geography["latitude"])
    centre_longitude = float(geography["longitude"])
    radius_km = float(geography["radius_km"])

    if radius_km != 15.0:
        fail(
            "CF-RETRO-01 v1.0 requires the frozen 15 km radius."
        )

    output_path = resolve_output_path(
        REPOSITORY_ROOT,
        configuration["outputs"]["catalog"],
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sha256_path = output_path.with_suffix(
        output_path.suffix + ".sha256"
    )

    manifest_path = (
        output_path.parent
        / "CF_RETRO_01_catalog_manifest.json"
    )

    chunks = generate_time_chunks(
        baseline_start,
        cutoff,
        CHUNK_DAYS,
    )

    print(f"Configuration: {CONFIG_PATH}")
    print(f"Configuration SHA-256: {configuration_hash}")
    print(f"Service: {INGV_ENDPOINT}")
    print(
        "Interval: "
        f"{utc_iso(baseline_start)} through {utc_iso(cutoff)} UTC"
    )
    print(
        "Geographic domain: "
        f"{centre_latitude}, {centre_longitude}, "
        f"radius {radius_km} km"
    )
    print(f"Download chunks: {len(chunks)}")
    print()

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/plain",
        }
    )

    downloaded_frames: list[pd.DataFrame] = []

    query_log: list[dict[str, Any]] = []

    for index, (chunk_start, chunk_end) in enumerate(
        chunks,
        start=1,
    ):
        print(
            f"[{index:02d}/{len(chunks):02d}] "
            f"{utc_iso(chunk_start)} -> "
            f"{utc_iso(chunk_end)}"
        )

        payload = request_chunk(
            session=session,
            start=chunk_start,
            end=chunk_end,
            latitude=centre_latitude,
            longitude=centre_longitude,
            radius_km=radius_km,
        )

        frame = parse_fdsn_text(payload)

        print(f"  Records received: {len(frame)}")

        query_log.append(
            {
                "chunk_number": index,
                "start_utc": utc_iso(chunk_start) + "Z",
                "end_utc": utc_iso(chunk_end) + "Z",
                "records_received": len(frame),
                "response_sha256": sha256_bytes(
                    payload.encode("utf-8")
                ),
            }
        )

        if not frame.empty:
            downloaded_frames.append(frame)

    session.close()

    if not downloaded_frames:
        fail(
            "No earthquake records were returned for any chunk."
        )

    combined = pd.concat(
        downloaded_frames,
        ignore_index=True,
        sort=False,
    )

    catalogue, statistics = normalize_catalogue(
        frame=combined,
        cutoff=cutoff,
        centre_latitude=centre_latitude,
        centre_longitude=centre_longitude,
        radius_km=radius_km,
    )

    if catalogue.empty:
        fail(
            "No valid events remained after frozen filtering."
        )

    # Deterministic CSV serialization.
    catalogue.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.6f",
    )

    catalogue_hash = sha256_file(output_path)

    sha256_path.write_text(
        f"{catalogue_hash}  {output_path.name}\n",
        encoding="utf-8",
    )

    execution_time = datetime.now(timezone.utc)

    earliest_event = str(catalogue["time_utc"].min())
    latest_event = str(catalogue["time_utc"].max())

    if parse_utc(latest_event) > cutoff:
        fail(
            "Safety check failed: the saved catalogue contains "
            "post-cutoff data."
        )

    manifest = {
        "experiment_id": configuration["experiment"]["id"],
        "experiment_version":
            configuration["experiment"]["version"],
        "protocol_status":
            configuration["experiment"]["status"],
        "execution_timestamp_utc":
            execution_time.isoformat().replace("+00:00", "Z"),
        "service_endpoint": INGV_ENDPOINT,
        "request_format": REQUEST_FORMAT,
        "data_start_utc":
            utc_iso(baseline_start) + "Z",
        "data_cutoff_utc":
            utc_iso(cutoff) + "Z",
        "target_event_utc":
            utc_iso(target_event_time) + "Z",
        "geographic_domain": {
            "latitude": centre_latitude,
            "longitude": centre_longitude,
            "radius_km": radius_km,
        },
        "chunk_days": CHUNK_DAYS,
        "chunk_count": len(chunks),
        "statistics": statistics,
        "earliest_saved_event_utc": earliest_event,
        "latest_saved_event_utc": latest_event,
        "output_file": str(
            output_path.relative_to(REPOSITORY_ROOT)
        ),
        "output_sha256": catalogue_hash,
        "configuration_file": str(
            CONFIG_PATH.relative_to(REPOSITORY_ROOT)
        ),
        "configuration_sha256": configuration_hash,
        "code_commit": get_git_commit(REPOSITORY_ROOT),
        "queries": query_log,
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

    manifest_hash = sha256_file(manifest_path)

    print()
    print("=" * 72)
    print("CATALOGUE ACQUISITION COMPLETED")
    print("=" * 72)
    print(f"Downloaded rows:      {statistics['downloaded_rows']}")
    print(
        "Invalid rows removed: "
        f"{statistics['invalid_rows_removed']}"
    )
    print(
        "Post-cutoff removed:  "
        f"{statistics['post_cutoff_rows_removed']}"
    )
    print(
        "Outside radius:       "
        f"{statistics['outside_radius_rows_removed']}"
    )
    print(
        "Duplicates removed:   "
        f"{statistics['duplicate_rows_removed']}"
    )
    print(f"Frozen catalogue rows:{statistics['final_rows']:>8}")
    print(f"Earliest event:       {earliest_event}")
    print(f"Latest event:         {latest_event}")
    print()
    print(f"Catalogue: {output_path}")
    print(f"SHA-256:   {catalogue_hash}")
    print(f"Hash file: {sha256_path}")
    print(f"Manifest:  {manifest_path}")
    print(f"Manifest SHA-256: {manifest_hash}")
    print()
    print(
        "The target earthquake and all later observations "
        "are excluded from this frozen catalogue."
    )


if __name__ == "__main__":
    main()

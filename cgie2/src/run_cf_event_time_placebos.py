#!/usr/bin/env python3
"""
CF-NEG-02 — Event-time placebo control.

This experiment keeps the frozen CF-RETRO-01 alert episodes fixed and
circularly shifts the complete significant-event-cluster sequence.

The objective is to determine whether the observed temporal association
between alert episodes and significant seismic clusters is stronger than
expected under placebo event timings.

Run from the repository root:

    python cgie2/src/run_cf_event_time_placebos.py
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


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[2]

CONFIG_PATH = (
    REPOSITORY_ROOT
    / "cgie2"
    / "config"
    / "cf_neg_02.yaml"
)


def fail(message: str) -> None:
    """Terminate execution with a clear error message."""
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_utc(value: Any) -> pd.Timestamp:
    """Parse a timestamp and normalize it to UTC."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid timestamp: {value}"
        ) from exc

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")

    return timestamp.tz_convert("UTC")


def utc_string(value: Any) -> str:
    """Serialize a timestamp as ISO-8601 UTC."""
    return (
        parse_utc(value)
        .isoformat()
        .replace("+00:00", "Z")
    )


def resolve_path(relative_path: str) -> Path:
    """Resolve a repository-relative path safely."""
    resolved = (
        REPOSITORY_ROOT
        / Path(relative_path)
    ).resolve()

    try:
        resolved.relative_to(
            REPOSITORY_ROOT.resolve()
        )
    except ValueError:
        fail(
            "Configured path escapes repository root: "
            f"{relative_path}"
        )

    return resolved


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    if not path.exists():
        fail(f"Configuration file not found: {path}")

    try:
        payload = yaml.safe_load(
            path.read_text(encoding="utf-8")
        )
    except yaml.YAMLError as exc:
        fail(f"Invalid YAML in {path}: {exc}")

    if not isinstance(payload, dict):
        fail(
            "Configuration root must be a mapping."
        )

    return payload


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 file hash."""
    if not path.exists():
        fail(f"File not found for hashing: {path}")

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_expected_hash(path: Path) -> str:
    """Read a SHA-256 sidecar file."""
    if not path.exists():
        fail(f"Hash file not found: {path}")

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        fail(f"Hash file is empty: {path}")

    digest = text.split()[0].lower()

    if (
        len(digest) != 64
        or not set(digest).issubset(
            set("0123456789abcdef")
        )
    ):
        fail(f"Invalid SHA-256 digest in {path}")

    return digest


def verify_hash(
    data_path: Path,
    hash_path: Path,
) -> str:
    """Verify a file against its SHA-256 sidecar."""
    expected = read_expected_hash(hash_path)
    actual = sha256_file(data_path)

    if actual != expected:
        fail(
            "SHA-256 verification failed.\n"
            f"File: {data_path}\n"
            f"Expected: {expected}\n"
            f"Actual: {actual}"
        )

    return actual


def get_git_commit() -> str:
    """Return the current Git commit."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return "UNAVAILABLE"


def write_json(
    path: Path,
    payload: Any,
) -> None:
    """Write deterministic JSON."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    frame: pd.DataFrame,
    float_precision: int,
) -> None:
    """Write deterministic CSV."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = frame.copy()

    for column in output.columns:
        if isinstance(
            output[column].dtype,
            pd.DatetimeTZDtype,
        ):
            output[column] = output[column].map(
                utc_string
            )

    output.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format=f"%.{float_precision}g",
        na_rep="",
    )


def validate_configuration(
    config: dict[str, Any],
) -> None:
    """Validate the frozen CF-NEG-02 configuration."""
    required_sections = {
        "experiment",
        "interpretation_boundary",
        "inputs",
        "analysis_period",
        "significant_event_definition",
        "alert_episode_definition",
        "association_rule",
        "primary_statistic",
        "placebo_control",
        "empirical_test",
        "effect_size",
        "evidence_classes",
        "robustness_checks",
        "required_outputs",
        "reproducibility",
        "safety_rules",
        "prohibited_practices",
    }

    missing = sorted(
        required_sections.difference(config)
    )

    if missing:
        fail(
            "Missing configuration sections: "
            + ", ".join(missing)
        )

    experiment = config["experiment"]

    if experiment.get("id") != "CF_NEG_02":
        fail("Unexpected experiment identifier.")

    if experiment.get("status") != "FROZEN":
        fail("CF-NEG-02 configuration is not FROZEN.")

    boundary = config[
        "interpretation_boundary"
    ]

    if boundary.get(
        "modifies_cf_retro_01"
    ) is not False:
        fail(
            "CF-NEG-02 must not modify CF-RETRO-01."
        )

    if boundary.get(
        "modifies_cf_neg_01"
    ) is not False:
        fail(
            "CF-NEG-02 must not modify CF-NEG-01."
        )

    if boundary.get(
        "permits_threshold_tuning"
    ) is not False:
        fail(
            "Threshold tuning must remain prohibited."
        )

    repetitions = int(
        config["placebo_control"]["repetitions"]
    )

    if repetitions != 1000:
        fail(
            "Frozen placebo repetitions must equal 1000."
        )

    if (
        config["placebo_control"]["method"]
        != "circular_event_cluster_shift"
    ):
        fail(
            "Unsupported placebo-control method."
        )

    if float(
        config[
            "significant_event_definition"
        ]["primary_minimum_magnitude"]
    ) != 3.0:
        fail(
            "Primary magnitude threshold must remain 3.0."
        )

    if float(
        config["association_rule"][
            "primary_maximum_lead_hours"
        ]
    ) != 168.0:
        fail(
            "Primary association window must remain 168 hours."
        )

    if int(
        config["primary_statistic"][
            "expected_observed_value"
        ]
    ) != 1:
        fail(
            "Expected observed associated-episode count "
            "must remain 1."
        )

    analysis_start = parse_utc(
        config["analysis_period"]["start_utc"]
    )

    analysis_end = parse_utc(
        config["analysis_period"]["end_utc"]
    )

    if analysis_start >= analysis_end:
        fail("Invalid analysis interval.")


def load_catalogue(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    """Load and verify the frozen earthquake catalogue."""
    catalogue_config = config[
        "inputs"
    ]["frozen_catalogue"]

    catalogue_path = resolve_path(
        catalogue_config["file"]
    )

    hash_path = resolve_path(
        catalogue_config["hash_file"]
    )

    verified_hash = verify_hash(
        catalogue_path,
        hash_path,
    )

    catalogue = pd.read_csv(
        catalogue_path,
        encoding="utf-8",
    )

    required_columns = {
        "event_id",
        "time_utc",
        "magnitude",
        "depth_km",
    }

    missing = sorted(
        required_columns.difference(
            catalogue.columns
        )
    )

    if missing:
        fail(
            "Catalogue missing columns: "
            + ", ".join(missing)
        )

    catalogue["time_utc"] = pd.to_datetime(
        catalogue["time_utc"],
        utc=True,
        errors="coerce",
        format="mixed",
    )

    catalogue["magnitude"] = pd.to_numeric(
        catalogue["magnitude"],
        errors="coerce",
    )

    catalogue["depth_km"] = pd.to_numeric(
        catalogue["depth_km"],
        errors="coerce",
    )

    if catalogue["time_utc"].isna().any():
        fail(
            "Catalogue contains invalid timestamps."
        )

    if catalogue["magnitude"].isna().any():
        fail(
            "Catalogue contains invalid magnitudes."
        )

    analysis_start = parse_utc(
        config["analysis_period"]["start_utc"]
    )

    analysis_end = parse_utc(
        config["analysis_period"]["end_utc"]
    )

    catalogue = catalogue.loc[
        (
            catalogue["time_utc"]
            >= analysis_start
        )
        & (
            catalogue["time_utc"]
            <= analysis_end
        )
    ].copy()

    if catalogue.empty:
        fail(
            "No catalogue events remain inside "
            "the frozen analysis period."
        )

    return (
        catalogue.sort_values(
            by=["time_utc", "event_id"],
            kind="stable",
        ).reset_index(drop=True),
        verified_hash,
    )


def load_alert_episodes(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    """Load frozen CF-RETRO-01 alert episodes."""
    path = resolve_path(
        config["inputs"]["alert_episodes"]["file"]
    )

    if not path.exists():
        fail(f"Alert episodes not found: {path}")

    episodes_hash = sha256_file(path)

    episodes = pd.read_csv(
        path,
        encoding="utf-8",
    )

    required_columns = {
        "episode_id",
        "start_utc",
        "end_utc",
        "duration_hours",
        "maximum_level",
    }

    missing = sorted(
        required_columns.difference(
            episodes.columns
        )
    )

    if missing:
        fail(
            "Alert episodes missing columns: "
            + ", ".join(missing)
        )

    episodes["start_utc"] = pd.to_datetime(
        episodes["start_utc"],
        utc=True,
        errors="coerce",
        format="mixed",
    )

    episodes["end_utc"] = pd.to_datetime(
        episodes["end_utc"],
        utc=True,
        errors="coerce",
        format="mixed",
    )

    episodes["duration_hours"] = pd.to_numeric(
        episodes["duration_hours"],
        errors="coerce",
    )

    if (
        episodes["start_utc"].isna().any()
        or episodes["end_utc"].isna().any()
    ):
        fail(
            "Alert episodes contain invalid timestamps."
        )

    if (
        episodes["end_utc"]
        < episodes["start_utc"]
    ).any():
        fail(
            "An alert episode ends before it starts."
        )

    analysis_start = parse_utc(
        config["analysis_period"]["start_utc"]
    )

    analysis_end = parse_utc(
        config["analysis_period"]["end_utc"]
    )

    overlap_mask = (
        (episodes["end_utc"] >= analysis_start)
        & (episodes["start_utc"] <= analysis_end)
    )

    episodes = episodes.loc[
        overlap_mask
    ].copy()

    if episodes.empty:
        fail(
            "No alert episodes overlap "
            "the frozen analysis period."
        )

    if (
        episodes["end_utc"] > analysis_end
    ).any():
        fail(
            "An alert episode exceeds "
            "the frozen analysis cutoff."
        )

    return (
        episodes.sort_values(
            by=["start_utc", "episode_id"],
            kind="stable",
        ).reset_index(drop=True),
        episodes_hash,
    )


def load_event_clusters(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    """Load frozen CF-NEG-01 event clusters."""
    path = resolve_path(
        config["inputs"][
            "cf_neg_01_event_clusters"
        ]["file"]
    )

    if not path.exists():
        fail(f"Event clusters not found: {path}")

    clusters_hash = sha256_file(path)

    clusters = pd.read_csv(
        path,
        encoding="utf-8",
    )

    required_columns = {
        "cluster_id",
        "representative_event_id",
        "representative_event_time_utc",
        "representative_magnitude",
        "representative_depth_km",
        "event_count",
    }

    missing = sorted(
        required_columns.difference(
            clusters.columns
        )
    )

    if missing:
        fail(
            "Event clusters missing columns: "
            + ", ".join(missing)
        )

    clusters[
        "representative_event_time_utc"
    ] = pd.to_datetime(
        clusters[
            "representative_event_time_utc"
        ],
        utc=True,
        errors="coerce",
        format="mixed",
    )

    clusters[
        "representative_magnitude"
    ] = pd.to_numeric(
        clusters[
            "representative_magnitude"
        ],
        errors="coerce",
    )

    if clusters[
        "representative_event_time_utc"
    ].isna().any():
        fail(
            "Event clusters contain invalid timestamps."
        )

    expected_threshold = float(
        config[
            "significant_event_definition"
        ]["primary_minimum_magnitude"]
    )

    if (
        clusters["representative_magnitude"]
        < expected_threshold
    ).any():
        fail(
            "CF-NEG-01 event-cluster input contains "
            "a representative magnitude below 3.0."
        )

    return (
        clusters.sort_values(
            by=[
                "representative_event_time_utc",
                "cluster_id",
            ],
            kind="stable",
        ).reset_index(drop=True),
        clusters_hash,
    )


def load_cf_neg_01_summary(
    config: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Load the frozen CF-NEG-01 summary."""
    path = resolve_path(
        config["inputs"][
            "cf_neg_01_summary"
        ]["file"]
    )

    if not path.exists():
        fail(f"CF-NEG-01 summary not found: {path}")

    try:
        summary = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        fail(
            f"Invalid CF-NEG-01 summary JSON: {exc}"
        )

    expected_observed = int(
        config["primary_statistic"][
            "expected_observed_value"
        ]
    )

    observed = int(
        summary[
            "primary_observed_metrics"
        ]["associated_episode_count"]
    )

    if observed != expected_observed:
        fail(
            "Observed CF-NEG-01 association count "
            f"is {observed}, expected {expected_observed}."
        )

    return summary, sha256_file(path)


def associate_episodes(
    episodes: pd.DataFrame,
    clusters: pd.DataFrame,
    maximum_lead_hours: float,
) -> pd.DataFrame:
    """Associate fixed alert episodes with event clusters."""
    cluster_times = pd.to_datetime(
        clusters[
            "representative_event_time_utc"
        ],
        utc=True,
        errors="raise",
        format="mixed",
    )

    records: list[dict[str, Any]] = []

    for episode in episodes.itertuples(
        index=False
    ):
        episode_start = parse_utc(
            episode.start_utc
        )

        episode_end = parse_utc(
            episode.end_utc
        )

        association_end = (
            episode_start
            + pd.Timedelta(
                hours=maximum_lead_hours
            )
        )

        mask = (
            (cluster_times >= episode_start)
            & (
                cluster_times
                <= association_end
            )
        )

        candidates = clusters.loc[
            mask
        ].copy()

        if candidates.empty:
            associated = False
            classification = "unassociated"
            first_cluster_id = None
            first_event_id = ""
            first_event_time = None
            first_event_magnitude = math.nan
            lead_time_hours = math.nan
            cluster_count = 0

        else:
            candidates = candidates.sort_values(
                by=[
                    "representative_event_time_utc",
                    "cluster_id",
                ],
                kind="stable",
            )

            first = candidates.iloc[0]

            first_event_time = parse_utc(
                first[
                    "representative_event_time_utc"
                ]
            )

            associated = True

            if first_event_time <= episode_end:
                classification = "overlapping"
            else:
                classification = (
                    "post_episode_associated"
                )

            first_cluster_id = int(
                first["cluster_id"]
            )

            first_event_id = str(
                first["representative_event_id"]
            )

            first_event_magnitude = float(
                first["representative_magnitude"]
            )

            lead_time_hours = float(
                (
                    first_event_time
                    - episode_start
                ).total_seconds()
                / 3600.0
            )

            cluster_count = int(
                len(candidates)
            )

        records.append(
            {
                "episode_id":
                    int(episode.episode_id),
                "episode_start_utc":
                    utc_string(episode_start),
                "episode_end_utc":
                    utc_string(episode_end),
                "episode_duration_hours":
                    float(
                        episode.duration_hours
                    ),
                "maximum_level":
                    str(episode.maximum_level),
                "associated":
                    associated,
                "classification":
                    classification,
                "associated_cluster_count":
                    cluster_count,
                "first_event_cluster_id":
                    first_cluster_id,
                "first_event_id":
                    first_event_id,
                "first_event_time_utc":
                    (
                        utc_string(first_event_time)
                        if first_event_time
                        is not None
                        else ""
                    ),
                "first_event_magnitude":
                    first_event_magnitude,
                "lead_time_hours":
                    lead_time_hours,
            }
        )

    return pd.DataFrame.from_records(
        records
    )


def summarize_associations(
    associations: pd.DataFrame,
    clusters: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize one observed or placebo association result."""
    episode_count = int(
        len(associations)
    )

    associated_count = int(
        associations["associated"].sum()
    )

    overlapping_count = int(
        (
            associations["classification"]
            == "overlapping"
        ).sum()
    )

    post_episode_count = int(
        (
            associations["classification"]
            == "post_episode_associated"
        ).sum()
    )

    unassociated_count = (
        episode_count - associated_count
    )

    lead_values = pd.to_numeric(
        associations.loc[
            associations["associated"],
            "lead_time_hours",
        ],
        errors="coerce",
    ).dropna()

    associated_fraction = (
        float(
            associated_count
            / episode_count
        )
        if episode_count > 0
        else 0.0
    )

    false_discovery_fraction = (
        float(
            unassociated_count
            / episode_count
        )
        if episode_count > 0
        else 0.0
    )

    associated_cluster_ids = set(
        pd.to_numeric(
            associations.loc[
                associations["associated"],
                "first_event_cluster_id",
            ],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )

    cluster_count = int(len(clusters))

    event_detection_fraction = (
        float(
            len(associated_cluster_ids)
            / cluster_count
        )
        if cluster_count > 0
        else 0.0
    )

    return {
        "alert_episode_count":
            episode_count,
        "event_cluster_count":
            cluster_count,
        "associated_episode_count":
            associated_count,
        "overlapping_episode_count":
            overlapping_count,
        "post_episode_associated_count":
            post_episode_count,
        "unassociated_episode_count":
            unassociated_count,
        "associated_episode_fraction":
            associated_fraction,
        "false_discovery_fraction":
            false_discovery_fraction,
        "event_clusters_preceded_by_alert":
            int(len(associated_cluster_ids)),
        "event_detection_fraction":
            event_detection_fraction,
        "median_lead_time_hours":
            (
                float(lead_values.median())
                if not lead_values.empty
                else None
            ),
    }


def circular_shift_clusters(
    clusters: pd.DataFrame,
    shift_hours: int,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> pd.DataFrame:
    """Circularly shift all cluster times by one common offset."""
    period_hours = int(
        (
            period_end - period_start
        ).total_seconds()
        // 3600
    ) + 1

    shifted = clusters.copy()

    original_times = pd.to_datetime(
        shifted[
            "representative_event_time_utc"
        ],
        utc=True,
        errors="raise",
        format="mixed",
    )

    offsets = (
        (
            original_times
            - period_start
        )
        .dt.total_seconds()
        .div(3600.0)
    )

    shifted_offsets = (
        offsets + shift_hours
    ) % period_hours

    shifted_times = (
        period_start
        + pd.to_timedelta(
            shifted_offsets,
            unit="h",
        )
    )

    shifted[
        "representative_event_time_utc"
    ] = shifted_times

    if "cluster_start_utc" in shifted.columns:
        original_start = pd.to_datetime(
            shifted["cluster_start_utc"],
            utc=True,
            errors="coerce",
            format="mixed",
        )

        relative_start = (
            original_start - original_times
        )

        shifted["cluster_start_utc"] = (
            shifted_times + relative_start
        )

    if "cluster_end_utc" in shifted.columns:
        original_end = pd.to_datetime(
            shifted["cluster_end_utc"],
            utc=True,
            errors="coerce",
            format="mixed",
        )

        relative_end = (
            original_end - original_times
        )

        shifted["cluster_end_utc"] = (
            shifted_times + relative_end
        )

    return shifted.sort_values(
        by=[
            "representative_event_time_utc",
            "cluster_id",
        ],
        kind="stable",
    ).reset_index(drop=True)


def run_placebos(
    episodes: pd.DataFrame,
    clusters: pd.DataFrame,
    config: dict[str, Any],
    observed_statistic: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run 1000 circular event-time placebo shifts."""
    placebo_config = config[
        "placebo_control"
    ]

    repetitions = int(
        placebo_config["repetitions"]
    )

    seed = int(
        placebo_config["random_seed"]
    )

    minimum_shift = int(
        placebo_config["shift"][
            "minimum_absolute_shift_hours"
        ]
    )

    period_start = parse_utc(
        config["analysis_period"]["start_utc"]
    )

    period_end = parse_utc(
        config["analysis_period"]["end_utc"]
    )

    period_hours = int(
        (
            period_end - period_start
        ).total_seconds()
        // 3600
    ) + 1

    allowed_shifts = np.array(
        [
            shift
            for shift in range(
                1,
                period_hours,
            )
            if (
                shift >= minimum_shift
                and (
                    period_hours - shift
                )
                >= minimum_shift
            )
        ],
        dtype=int,
    )

    if len(allowed_shifts) == 0:
        fail(
            "No admissible circular shifts remain."
        )

    rng = np.random.default_rng(seed)

    sampled_shifts = rng.choice(
        allowed_shifts,
        size=repetitions,
        replace=True,
    )

    maximum_lead_hours = float(
        config["association_rule"][
            "primary_maximum_lead_hours"
        ]
    )

    records: list[dict[str, Any]] = []

    for repetition, shift_hours in enumerate(
        sampled_shifts,
        start=1,
    ):
        shifted_clusters = (
            circular_shift_clusters(
                clusters=clusters,
                shift_hours=int(
                    shift_hours
                ),
                period_start=period_start,
                period_end=period_end,
            )
        )

        associations = associate_episodes(
            episodes=episodes,
            clusters=shifted_clusters,
            maximum_lead_hours=(
                maximum_lead_hours
            ),
        )

        metrics = summarize_associations(
            associations=associations,
            clusters=shifted_clusters,
        )

        records.append(
            {
                "repetition":
                    repetition,
                "shift_hours":
                    int(shift_hours),
                "associated_episode_count":
                    metrics[
                        "associated_episode_count"
                    ],
                "overlapping_episode_count":
                    metrics[
                        "overlapping_episode_count"
                    ],
                "post_episode_associated_count":
                    metrics[
                        "post_episode_associated_count"
                    ],
                "associated_episode_fraction":
                    metrics[
                        "associated_episode_fraction"
                    ],
                "false_discovery_fraction":
                    metrics[
                        "false_discovery_fraction"
                    ],
                "event_detection_fraction":
                    metrics[
                        "event_detection_fraction"
                    ],
                "median_lead_time_hours":
                    metrics[
                        "median_lead_time_hours"
                    ],
            }
        )

        if (
            repetition % 100 == 0
            or repetition == repetitions
        ):
            print(
                "Event-time placebos completed: "
                f"{repetition}/{repetitions}"
            )

    placebo_frame = pd.DataFrame.from_records(
        records
    )

    placebo_values = pd.to_numeric(
        placebo_frame[
            "associated_episode_count"
        ],
        errors="raise",
    )

    greater_or_equal_count = int(
        (
            placebo_values
            >= observed_statistic
        ).sum()
    )

    empirical_p_value = float(
        (
            1 + greater_or_equal_count
        )
        / (
            1 + repetitions
        )
    )

    placebo_mean = float(
        placebo_values.mean()
    )

    placebo_median = float(
        placebo_values.median()
    )

    placebo_standard_deviation = float(
        placebo_values.std(
            ddof=0
        )
    )

    standardized_difference = (
        float(
            (
                observed_statistic
                - placebo_mean
            )
            / placebo_standard_deviation
        )
        if placebo_standard_deviation > 0.0
        else None
    )

    observed_percentile = float(
        100.0
        * (
            placebo_values
            <= observed_statistic
        ).mean()
    )

    summary = {
        "repetitions":
            repetitions,
        "random_seed":
            seed,
        "minimum_absolute_shift_hours":
            minimum_shift,
        "observed_associated_episode_count":
            int(observed_statistic),
        "placebo_associated_count_minimum":
            int(placebo_values.min()),
        "placebo_associated_count_median":
            placebo_median,
        "placebo_associated_count_mean":
            placebo_mean,
        "placebo_associated_count_maximum":
            int(placebo_values.max()),
        "placebo_associated_count_standard_deviation":
            placebo_standard_deviation,
        "placebo_greater_than_or_equal_count":
            greater_or_equal_count,
        "empirical_p_value":
            empirical_p_value,
        "observed_minus_placebo_mean":
            float(
                observed_statistic
                - placebo_mean
            ),
        "observed_minus_placebo_median":
            float(
                observed_statistic
                - placebo_median
            ),
        "standardized_difference":
            standardized_difference,
        "observed_percentile_in_placebo_distribution":
            observed_percentile,
    }

    return placebo_frame, summary


def build_robustness_table(
    episodes: pd.DataFrame,
    clusters: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build frozen robustness checks."""
    records: list[dict[str, Any]] = []

    primary_window = float(
        config["association_rule"][
            "primary_maximum_lead_hours"
        ]
    )

    observed = associate_episodes(
        episodes=episodes,
        clusters=clusters,
        maximum_lead_hours=primary_window,
    )

    observed_summary = summarize_associations(
        associations=observed,
        clusters=clusters,
    )

    records.append(
        {
            "check_type":
                "primary",
            "excluded_id":
                "",
            "lead_window_hours":
                primary_window,
            **observed_summary,
        }
    )

    non_overlap_associations = observed.loc[
        observed["classification"]
        != "overlapping"
    ].copy()

    records.append(
        {
            "check_type":
                "exclude_overlapping",
            "excluded_id":
                "",
            "lead_window_hours":
                primary_window,
            "alert_episode_count":
                int(len(observed)),
            "event_cluster_count":
                int(len(clusters)),
            "associated_episode_count":
                int(
                    (
                        non_overlap_associations[
                            "classification"
                        ]
                        == "post_episode_associated"
                    ).sum()
                ),
            "overlapping_episode_count":
                0,
            "post_episode_associated_count":
                int(
                    (
                        non_overlap_associations[
                            "classification"
                        ]
                        == "post_episode_associated"
                    ).sum()
                ),
            "unassociated_episode_count":
                int(
                    len(observed)
                    - (
                        non_overlap_associations[
                            "classification"
                        ]
                        == "post_episode_associated"
                    ).sum()
                ),
            "associated_episode_fraction":
                float(
                    (
                        non_overlap_associations[
                            "classification"
                        ]
                        == "post_episode_associated"
                    ).sum()
                    / len(observed)
                ),
            "false_discovery_fraction":
                float(
                    1.0
                    - (
                        (
                            non_overlap_associations[
                                "classification"
                            ]
                            == "post_episode_associated"
                        ).sum()
                        / len(observed)
                    )
                ),
            "event_clusters_preceded_by_alert":
                0,
            "event_detection_fraction":
                0.0,
            "median_lead_time_hours":
                None,
        }
    )

    for episode_id in episodes[
        "episode_id"
    ].tolist():
        reduced_episodes = episodes.loc[
            episodes["episode_id"]
            != episode_id
        ].copy()

        associations = associate_episodes(
            episodes=reduced_episodes,
            clusters=clusters,
            maximum_lead_hours=primary_window,
        )

        metrics = summarize_associations(
            associations=associations,
            clusters=clusters,
        )

        records.append(
            {
                "check_type":
                    "leave_one_episode_out",
                "excluded_id":
                    str(episode_id),
                "lead_window_hours":
                    primary_window,
                **metrics,
            }
        )

    for cluster_id in clusters[
        "cluster_id"
    ].tolist():
        reduced_clusters = clusters.loc[
            clusters["cluster_id"]
            != cluster_id
        ].copy()

        associations = associate_episodes(
            episodes=episodes,
            clusters=reduced_clusters,
            maximum_lead_hours=primary_window,
        )

        metrics = summarize_associations(
            associations=associations,
            clusters=reduced_clusters,
        )

        records.append(
            {
                "check_type":
                    "leave_one_event_cluster_out",
                "excluded_id":
                    str(cluster_id),
                "lead_window_hours":
                    primary_window,
                **metrics,
            }
        )

    secondary_windows = config[
        "robustness_checks"
    ]["secondary_lead_windows"]["hours"]

    for window_hours in secondary_windows:
        associations = associate_episodes(
            episodes=episodes,
            clusters=clusters,
            maximum_lead_hours=float(
                window_hours
            ),
        )

        metrics = summarize_associations(
            associations=associations,
            clusters=clusters,
        )

        records.append(
            {
                "check_type":
                    "secondary_lead_window",
                "excluded_id":
                    "",
                "lead_window_hours":
                    float(window_hours),
                **metrics,
            }
        )

    return pd.DataFrame.from_records(
        records
    )


def classify_evidence(
    observed: dict[str, Any],
    placebo_summary: dict[str, Any],
) -> str:
    """Apply the frozen evidence classification."""
    cluster_count = int(
        observed["event_cluster_count"]
    )

    observed_count = int(
        observed[
            "associated_episode_count"
        ]
    )

    placebo_median = float(
        placebo_summary[
            "placebo_associated_count_median"
        ]
    )

    empirical_p = float(
        placebo_summary["empirical_p_value"]
    )

    if cluster_count < 5:
        return "inconclusive_low_power"

    if (
        empirical_p < 0.05
        and observed_count
        > placebo_median
    ):
        return "temporally_specific"

    if (
        0.05 <= empirical_p < 0.10
        and observed_count
        > placebo_median
    ):
        return "preliminary"

    if observed_count <= placebo_median:
        return "non_discriminating"

    return "preliminary"


def write_report(
    path: Path,
    summary: dict[str, Any],
) -> None:
    """Write the human-readable CF-NEG-02 report."""
    observed = summary[
        "observed_metrics"
    ]

    placebo = summary[
        "placebo_control"
    ]

    text = f"""# CF-NEG-02 — Event-Time Placebo Control

## Scientific status

Experiment: CF-NEG-02

Configuration status: FROZEN

Analysis type: retrospective event-time placebo control

Earthquake-prediction claim: no

## Observed result

Alert episodes:

{observed["alert_episode_count"]}

Significant event clusters:

{observed["event_cluster_count"]}

Associated alert episodes:

{observed["associated_episode_count"]}

Overlapping alert episodes:

{observed["overlapping_episode_count"]}

Post-episode associated alerts:

{observed["post_episode_associated_count"]}

False-discovery fraction:

{observed["false_discovery_fraction"]:.6f}

## Event-time placebo distribution

Placebo repetitions:

{placebo["repetitions"]}

Placebo median associated count:

{placebo["placebo_associated_count_median"]:.6f}

Placebo mean associated count:

{placebo["placebo_associated_count_mean"]:.6f}

Placebo maximum associated count:

{placebo["placebo_associated_count_maximum"]}

Empirical p-value:

{placebo["empirical_p_value"]:.6f}

Observed percentile:

{placebo["observed_percentile_in_placebo_distribution"]:.3f}

## Evidence classification

{summary["evidence_class"]}

## Overlap-sensitive interpretation

The observed CF-NEG-01 association is overlapping: the significant
event occurred while the alert episode was already active.

Therefore this result can represent temporal coupling or concurrent
detection of an unstable phase, but it must not be described as early
warning or prediction.

## Interpretation boundary

CF-NEG-02 tests whether the timing of frozen alert episodes is more
strongly coupled to the frozen seismic-cluster sequence than expected
after circular event-time shifts.

It does not validate deterministic earthquake prediction.
"""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        text,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 78)
    print("CF-NEG-02 — EVENT-TIME PLACEBO CONTROL")
    print("=" * 78)

    config = load_yaml(CONFIG_PATH)
    validate_configuration(config)

    configuration_hash = sha256_file(
        CONFIG_PATH
    )

    catalogue, catalogue_hash = (
        load_catalogue(config)
    )

    episodes, episodes_hash = (
        load_alert_episodes(config)
    )

    clusters, clusters_hash = (
        load_event_clusters(config)
    )

    cf_neg_01_summary, summary_hash = (
        load_cf_neg_01_summary(config)
    )

    associations_path = resolve_path(
        config["inputs"][
            "cf_neg_01_episode_associations"
        ]["file"]
    )

    if not associations_path.exists():
        fail(
            "CF-NEG-01 association table not found: "
            f"{associations_path}"
        )

    associations_input_hash = (
        sha256_file(associations_path)
    )

    primary_window = float(
        config["association_rule"][
            "primary_maximum_lead_hours"
        ]
    )

    observed_associations = associate_episodes(
        episodes=episodes,
        clusters=clusters,
        maximum_lead_hours=primary_window,
    )

    observed_metrics = summarize_associations(
        associations=observed_associations,
        clusters=clusters,
    )

    expected_observed = int(
        config["primary_statistic"][
            "expected_observed_value"
        ]
    )

    actual_observed = int(
        observed_metrics[
            "associated_episode_count"
        ]
    )

    if actual_observed != expected_observed:
        fail(
            "Reconstructed observed association count "
            f"is {actual_observed}, expected {expected_observed}."
        )

    placebo_runs, placebo_summary = (
        run_placebos(
            episodes=episodes,
            clusters=clusters,
            config=config,
            observed_statistic=(
                actual_observed
            ),
        )
    )

    robustness = build_robustness_table(
        episodes=episodes,
        clusters=clusters,
        config=config,
    )

    evidence_class = classify_evidence(
        observed=observed_metrics,
        placebo_summary=placebo_summary,
    )

    output_config = config[
        "required_outputs"
    ]

    placebo_path = resolve_path(
        output_config["placebo_runs"]["file"]
    )

    observed_path = resolve_path(
        output_config[
            "observed_associations"
        ]["file"]
    )

    robustness_path = resolve_path(
        output_config["robustness"]["file"]
    )

    summary_path = resolve_path(
        output_config["summary"]["file"]
    )

    report_path = resolve_path(
        output_config["report"]["file"]
    )

    manifest_path = resolve_path(
        output_config["manifest"]["file"]
    )

    float_precision = int(
        config["reproducibility"][
            "float_precision"
        ]
    )

    write_csv(
        placebo_path,
        placebo_runs,
        float_precision,
    )

    write_csv(
        observed_path,
        observed_associations,
        float_precision,
    )

    write_csv(
        robustness_path,
        robustness,
        float_precision,
    )

    summary = {
        "experiment_id":
            "CF_NEG_02",
        "configuration_version":
            config["experiment"]["version"],
        "analysis_start_utc":
            config["analysis_period"][
                "start_utc"
            ],
        "analysis_end_utc":
            config["analysis_period"][
                "end_utc"
            ],
        "catalogue_rows":
            int(len(catalogue)),
        "alert_episode_count":
            int(len(episodes)),
        "event_cluster_count":
            int(len(clusters)),
        "primary_magnitude_threshold":
            float(
                config[
                    "significant_event_definition"
                ][
                    "primary_minimum_magnitude"
                ]
            ),
        "primary_association_window_hours":
            primary_window,
        "observed_metrics":
            observed_metrics,
        "placebo_control":
            placebo_summary,
        "evidence_class":
            evidence_class,
        "overlap_only_observed_signal":
            (
                observed_metrics[
                    "associated_episode_count"
                ]
                > 0
                and observed_metrics[
                    "post_episode_associated_count"
                ]
                == 0
            ),
        "earthquake_prediction_claim":
            False,
        "interpretation":
            (
                "Retrospective event-time placebo "
                "control only."
            ),
    }

    write_json(summary_path, summary)
    write_report(report_path, summary)

    manifest = {
        "experiment_id":
            "CF_NEG_02",
        "stage":
            "event_time_placebo_control",
        "configuration_status":
            config["experiment"]["status"],
        "execution_timestamp_utc":
            datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        "source_commit":
            get_git_commit(),
        "configuration_file":
            str(
                CONFIG_PATH.relative_to(
                    REPOSITORY_ROOT
                )
            ),
        "configuration_sha256":
            configuration_hash,
        "script_sha256":
            sha256_file(SCRIPT_PATH),
        "input_hashes": {
            "catalogue_sha256":
                catalogue_hash,
            "alert_episodes_sha256":
                episodes_hash,
            "event_clusters_sha256":
                clusters_hash,
            "cf_neg_01_summary_sha256":
                summary_hash,
            "cf_neg_01_associations_sha256":
                associations_input_hash,
        },
        "output_hashes": {
            "placebo_runs_sha256":
                sha256_file(placebo_path),
            "observed_associations_sha256":
                sha256_file(observed_path),
            "robustness_sha256":
                sha256_file(robustness_path),
            "summary_sha256":
                sha256_file(summary_path),
            "report_sha256":
                sha256_file(report_path),
        },
        "placebo_repetitions":
            int(
                config["placebo_control"][
                    "repetitions"
                ]
            ),
        "random_seed":
            int(
                config["placebo_control"][
                    "random_seed"
                ]
            ),
        "target_event_included":
            False,
        "post_cutoff_data_used":
            False,
        "cf_retro_01_modified":
            False,
        "cf_neg_01_modified":
            False,
        "summary":
            summary,
    }

    write_json(manifest_path, manifest)

    print()
    print("=" * 78)
    print("CF-NEG-02 COMPLETED")
    print("=" * 78)
    print(
        "Observed associated episodes: "
        f"{actual_observed}"
    )
    print(
        "Observed overlapping episodes: "
        f"{observed_metrics['overlapping_episode_count']}"
    )
    print(
        "Observed post-episode associations: "
        f"{observed_metrics['post_episode_associated_count']}"
    )
    print(
        "Placebo median: "
        f"{placebo_summary['placebo_associated_count_median']:.3f}"
    )
    print(
        "Placebo mean: "
        f"{placebo_summary['placebo_associated_count_mean']:.3f}"
    )
    print(
        "Empirical p-value: "
        f"{placebo_summary['empirical_p_value']:.6f}"
    )
    print(
        "Evidence class: "
        f"{evidence_class}"
    )
    print()
    print(f"Placebo runs: {placebo_path}")
    print(f"Observed associations: {observed_path}")
    print(f"Robustness: {robustness_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
  

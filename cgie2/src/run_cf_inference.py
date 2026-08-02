#!/usr/bin/env python3
"""
CF-RETRO-01 — Frozen retrospective inference engine.

Run from the repository root:

    python cgie2/src/run_cf_inference.py

Inputs:

    cgie2/config/cf_retro_01.yaml
    cgie2/config/cf_retro_01_inference.yaml
    outputs/CF_RETRO_01_features.csv
    outputs/CF_RETRO_01_features.csv.sha256

Outputs:

    outputs/CF_RETRO_01_normalized_features.csv
    outputs/CF_RETRO_01_baseline_parameters.json
    outputs/CF_RETRO_01_metrics.csv
    outputs/CF_RETRO_01_metrics.csv.sha256
    outputs/CF_RETRO_01_thresholds.json
    outputs/CF_RETRO_01_window_alerts.csv
    outputs/CF_RETRO_01_global_alerts.csv
    outputs/CF_RETRO_01_alert_episodes.csv
    outputs/CF_RETRO_01_summary.json
    outputs/CF_RETRO_01_report.md
    outputs/CF_RETRO_01_inference_manifest.json
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

FEATURE_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "cgie2"
    / "config"
    / "cf_retro_01.yaml"
)

INFERENCE_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "cgie2"
    / "config"
    / "cf_retro_01_inference.yaml"
)

LEVEL_PRIORITY = {
    "normal": 0,
    "attention": 1,
    "anomaly": 2,
    "structural": 3,
}


def fail(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"Configuration file not found: {path}")

    try:
        data = yaml.safe_load(
            path.read_text(encoding="utf-8")
        )
    except yaml.YAMLError as exc:
        fail(f"Invalid YAML in {path}: {exc}")

    if not isinstance(data, dict):
        fail(f"YAML root must be a mapping: {path}")

    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_expected_hash(path: Path) -> str:
    if not path.exists():
        fail(f"Hash file not found: {path}")

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        fail(f"Hash file is empty: {path}")

    digest = text.split()[0].lower()

    if len(digest) != 64:
        fail(f"Invalid SHA-256 digest: {path}")

    if not set(digest).issubset(
        set("0123456789abcdef")
    ):
        fail(f"Invalid SHA-256 characters: {path}")

    return digest


def verify_hash(
    data_path: Path,
    hash_path: Path,
) -> str:
    if not data_path.exists():
        fail(f"Input file not found: {data_path}")

    expected = read_expected_hash(hash_path)
    actual = sha256_file(data_path)

    if actual != expected:
        fail(
            "SHA-256 verification failed.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    return actual


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def resolve_path(relative_path: str) -> Path:
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


def parse_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")

    return timestamp.tz_convert("UTC")


def utc_string(value: str | pd.Timestamp) -> str:
    return (
        parse_utc(value)
        .isoformat()
        .replace("+00:00", "Z")
    )


def write_json(
    path: Path,
    payload: Any,
) -> None:
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
    frame: pd.DataFrame,
    path: Path,
    float_precision: int,
) -> None:
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
    feature_config: dict[str, Any],
    inference_config: dict[str, Any],
) -> None:
    if (
        feature_config["experiment"]["id"]
        != "CF_RETRO_01"
    ):
        fail("Unexpected feature experiment ID.")

    experiment = inference_config["experiment"]

    if experiment["id"] != "CF_RETRO_01":
        fail("Unexpected inference experiment ID.")

    if experiment["stage"] != "inference":
        fail("Unexpected inference stage.")

    if experiment["status"] != "FROZEN":
        fail("Inference configuration is not FROZEN.")

    windows = inference_config["windows"]

    if windows["primary_window"] != "7d":
        fail("Frozen primary window must be 7d.")

    expected_order = [
        "1d",
        "3d",
        "7d",
        "30d",
    ]

    if windows["complete_window_order"] != expected_order:
        fail("Unexpected frozen window order.")

    if (
        int(
            windows[
                "multi_window_confirmation_required"
            ]
        )
        != 2
    ):
        fail(
            "Multi-window confirmation must equal 2."
        )

    safety_rules = inference_config[
        "safety_rules"
    ]

    for rule, value in safety_rules.items():
        if value is not False:
            fail(
                "Frozen safety rule must be false: "
                f"{rule}"
            )


def load_features(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    features_path = resolve_path(
        config["input"]["features_file"]
    )

    hash_path = resolve_path(
        config["input"]["features_hash_file"]
    )

    input_hash = verify_hash(
        features_path,
        hash_path,
    )

    features = pd.read_csv(
        features_path,
        encoding="utf-8",
    )

    if features.empty:
        fail("Frozen feature table is empty.")

    selected_features = config[
        "selected_features"
    ]

    required_columns = {
        "experiment_id",
        "endpoint_utc",
        "window_id",
        "window_duration_hours",
        *selected_features,
    }

    missing = sorted(
        required_columns.difference(
            features.columns
        )
    )

    if missing:
        fail(
            "Missing feature columns: "
            + ", ".join(missing)
        )

    features["endpoint_utc"] = pd.to_datetime(
        features["endpoint_utc"],
        utc=True,
        errors="coerce",
        format="mixed",
    )

    if features["endpoint_utc"].isna().any():
        fail("Feature table contains invalid timestamps.")

    cutoff = parse_utc(
        config["target_event"][
            "frozen_cutoff_utc"
        ]
    )

    target_event = parse_utc(
        config["target_event"]["utc_time"]
    )

    if features["endpoint_utc"].max() > cutoff:
        fail("Feature table exceeds frozen cutoff.")

    if (
        features["endpoint_utc"]
        >= target_event
    ).any():
        fail(
            "Target-event or post-event features detected."
        )

    expected_windows = set(
        config["windows"][
            "complete_window_order"
        ]
    )

    observed_windows = set(
        features["window_id"]
        .astype(str)
        .unique()
    )

    if observed_windows != expected_windows:
        fail(
            "Unexpected temporal windows: "
            f"{observed_windows}"
        )

    if features.duplicated(
        subset=[
            "endpoint_utc",
            "window_id",
        ],
        keep=False,
    ).any():
        fail(
            "Duplicate endpoint-window combinations."
        )

    for feature in applicable_features_for_window(
    config,
    str(window_id),
):
        features[feature] = pd.to_numeric(
            features[feature],
            errors="coerce",
        )

    window_order = {
        "1d": 0,
        "3d": 1,
        "7d": 2,
        "30d": 3,
    }

    features["_window_order"] = (
        features["window_id"].map(
            window_order
        )
    )

    features = features.sort_values(
        by=[
            "endpoint_utc",
            "_window_order",
        ],
        kind="stable",
    ).drop(
        columns=["_window_order"]
    ).reset_index(drop=True)

    return features, input_hash


def estimate_robust_parameters(
    values: pd.Series,
    mad_factor: float,
    iqr_factor: float,
) -> dict[str, Any]:
    clean = values.dropna().astype(float)

    if clean.empty:
        fail(
            "Cannot estimate parameters from "
            "an empty baseline series."
        )

    median = float(clean.median())

    mad = float(
        np.median(
            np.abs(
                clean.to_numpy(dtype=float)
                - median
            )
        )
    )

    scale = mad * mad_factor
    scale_source = "normalized_mad"

    if (
        not math.isfinite(scale)
        or scale <= 0.0
    ):
        q25 = float(clean.quantile(0.25))
        q75 = float(clean.quantile(0.75))

        scale = (
            q75 - q25
        ) / iqr_factor

        scale_source = "baseline_iqr"

    if (
        not math.isfinite(scale)
        or scale <= 0.0
    ):
        scale = 1.0
        scale_source = "constant_1"

    return {
        "median": median,
        "scale": float(scale),
        "scale_source": scale_source,
        "baseline_nonmissing_rows": int(
            len(clean)
        ),
    }


def directional_deviation(
    z_score: pd.Series,
    direction: str,
) -> pd.Series:
    if direction == "upper":
        return z_score.clip(lower=0.0)

    if direction == "lower":
        return (-z_score).clip(lower=0.0)

    if direction == "two_sided":
        return z_score.abs()

    fail(f"Unknown feature direction: {direction}")

    return z_score.abs()

    def applicable_features_for_window(
    config: dict[str, Any],
    window_id: str,
) -> list[str]: selected_features = list(config["selected_features"])

    exclusions = config.get(
        "feature_window_exclusions",
        {},
    )

    applicable_features: list[str] = []

    for feature in selected_features:
        excluded_windows = exclusions.get(
            feature,
            [],
        )

        normalized_excluded_windows = {
            str(value) for value in excluded_windows
        }

        if str(window_id) in normalized_excluded_windows:
            continue

        applicable_features.append(feature)

    if not applicable_features:
        fail(
            f"No applicable features remain for window {window_id}."
        )

    return applicable_features

def normalize_features(
    features: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = features.copy()

    selected_features = config[
        "selected_features"
    ]

    directions = config[
        "feature_directions"
    ]

    baseline_start = parse_utc(
        config["periods"][
            "baseline_start_utc"
        ]
    )

    baseline_end = parse_utc(
        config["periods"][
            "baseline_end_utc"
        ]
    )

    normalization = config["normalization"]
    missing_config = config["missing_values"]

    mad_factor = float(
        normalization[
            "normalized_mad_factor"
        ]
    )

    iqr_factor = float(
        normalization[
            "iqr_scale_factor"
        ]
    )

    clip_minimum = float(
        normalization["clipping"]["minimum"]
    )

    clip_maximum = float(
        normalization["clipping"]["maximum"]
    )

    minimum_coverage = float(
        missing_config[
            "minimum_baseline_coverage_fraction"
        ]
    )

    coverage_policy = missing_config.get(
        "coverage_policy",
        {},
    )

    event_dependent_features = set(
        coverage_policy.get(
            "event_dependent_features",
            [
                "maximum_magnitude",
                "median_magnitude",
                "log10_cumulative_energy_joule",
                "median_depth_km",
                "depth_mad_km",
                "spatial_dispersion_km",
                "median_interevent_time_hours",
                "interevent_time_mad_hours",
                "temporal_burstiness",
            ],
        )
    )

    default_coverage_basis = str(
        coverage_policy.get(
            "default_basis",
            "all_baseline_endpoints",
        )
    )

    event_dependent_coverage_basis = str(
        coverage_policy.get(
            "event_dependent_basis",
            "nonempty_baseline_windows",
        )
    )

    if default_coverage_basis != "all_baseline_endpoints":
        fail(
            "Unsupported default baseline coverage basis: "
            f"{default_coverage_basis}"
        )

    if (
        event_dependent_coverage_basis
        != "nonempty_baseline_windows"
    ):
        fail(
            "Unsupported event-dependent coverage basis: "
            f"{event_dependent_coverage_basis}"
        )
    
    coverage_policy = missing_config.get(
        "coverage_policy",
        {},
    )

    default_coverage_basis = coverage_policy.get(
        "default_basis",
        "all_baseline_endpoints",
    )

    event_dependent_coverage_basis = (
        coverage_policy.get(
            "event_dependent_basis",
            "nonempty_baseline_windows",
        )
    )

    event_dependent_features = set(
        coverage_policy.get(
            "event_dependent_features",
            [],
        )
    )

    report_unconditional_coverage = bool(
        coverage_policy.get(
            "report_unconditional_coverage",
            True,
        )
    )

    report_conditional_coverage = bool(
        coverage_policy.get(
            "report_conditional_coverage",
            True,
        )
    )

    if default_coverage_basis != "all_baseline_endpoints":
        fail(
            "Unsupported default coverage basis: "
            f"{default_coverage_basis}"
        )

    if (
        event_dependent_coverage_basis
        != "nonempty_baseline_windows"
    ):
        fail(
            "Unsupported event-dependent coverage basis: "
            f"{event_dependent_coverage_basis}"
        )

    if "event_count" not in output.columns:
        fail(
            "event_count is required for conditional "
            "baseline coverage."
        )

    parameters: dict[str, Any] = {}

    for window_id in config[
        "windows"
    ]["complete_window_order"]:

        window_mask = (
            output["window_id"] == window_id
        )

        baseline_mask = (
            window_mask
            & (
                output["endpoint_utc"]
                >= baseline_start
            )
            & (
                output["endpoint_utc"]
                <= baseline_end
            )
        )

        if not baseline_mask.any():
            fail(
                f"No baseline rows for window {window_id}."
            )

        parameters[window_id] = {}

        for feature in selected_features:
            all_baseline_values = output.loc[
                baseline_mask,
                feature,
            ]

            unconditional_coverage = float(
                all_baseline_values.notna().mean()
            )

            is_event_dependent = (
                feature in event_dependent_features
            )

            if is_event_dependent:
                relevant_baseline_mask = (
                    baseline_mask
                    & (
                        pd.to_numeric(
                            output["event_count"],
                            errors="coerce",
                        ).fillna(0.0)
                        > 0.0
                    )
                )

                baseline_values = output.loc[
                    relevant_baseline_mask,
                    feature,
                ]

                if baseline_values.empty:
                    fail(
                        "No nonempty baseline windows for "
                        f"{window_id}/{feature}."
                    )

                conditional_coverage = float(
                    baseline_values.notna().mean()
                )

                coverage = conditional_coverage
                coverage_basis = (
                    "nonempty_baseline_windows"
                )

            else:
                baseline_values = (
                    all_baseline_values
                )

                conditional_coverage = None
                coverage = unconditional_coverage
                coverage_basis = (
                    "all_baseline_endpoints"
                )

            if coverage < minimum_coverage:
                print()
                print("=" * 70)
                print("BASELINE COVERAGE DIAGNOSTIC")
                print("=" * 70)
                print(f"Window: {window_id}")
                print(f"Feature: {feature}")
                print(f"Coverage basis: {coverage_basis}")
                print(f"Coverage                 : {coverage:.4f}")
                print(f"Required                 : {minimum_coverage:.4f}")
                print(f"Unconditional coverage   : {unconditional_coverage:.4f}")

            
                fail(
                "Baseline coverage below frozen minimum "
                f"for {window_id}/{feature}: {coverage:.4f}"
                )

            estimates = estimate_robust_parameters(
                baseline_values,
                mad_factor,
                iqr_factor,
            )

            median = float(
                estimates["median"]
            )

            scale = float(
                estimates["scale"]
            )

            raw_values = output.loc[
                window_mask,
                feature,
            ]

            missing_column = (
                f"{feature}__missing"
            )

            z_column = f"{feature}__z"

            deviation_column = (
                f"{feature}"
                "__directional_deviation"
            )

            continuity_column = (
                f"{feature}__continuity"
            )

            output.loc[
                window_mask,
                missing_column,
            ] = raw_values.isna().astype(int)

            imputed = raw_values.fillna(median)

            z_score = (
                (imputed - median) / scale
            ).clip(
                lower=clip_minimum,
                upper=clip_maximum,
            )

            deviation = directional_deviation(
                z_score,
                directions[feature],
            )

            continuity = 1.0 / (
                1.0 + deviation
            )

            output.loc[
                window_mask,
                feature,
            ] = imputed

            output.loc[
                window_mask,
                z_column,
            ] = z_score

            output.loc[
                window_mask,
                deviation_column,
            ] = deviation

            output.loc[
                window_mask,
                continuity_column,
            ] = continuity

            parameter_record = {
                **estimates,
                "baseline_coverage_fraction":
                    coverage,
                "coverage_basis":
                    coverage_basis,
                "event_dependent_feature":
                    is_event_dependent,
                "direction":
                    directions[feature],
                "imputation_value":
                    median,
            }

            if report_unconditional_coverage:
                parameter_record[
                    "unconditional_baseline_coverage_fraction"
                ] = unconditional_coverage

            if (
                report_conditional_coverage
                and conditional_coverage is not None
            ):
                parameter_record[
                    "conditional_nonempty_baseline_coverage_fraction"
                ] = conditional_coverage

            parameters[window_id][feature] = (
                parameter_record
            )

    return output, parameters


def calculate_gamma_and_sci(
    normalized: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    output = normalized.copy()

    continuity_columns = [
        f"{feature}__continuity"
        for feature in config[
            "selected_features"
        ]
    ]

    matrix = output[
        continuity_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(matrix).all():
        fail(
            "Non-finite continuity values detected."
        )

    output["Gamma_CF"] = np.min(
        matrix,
        axis=1,
    )

    epsilon = float(
        config["metrics"]["SCI"]["epsilon"]
    )

    output["SCI"] = np.exp(
        np.mean(
            np.log(
                np.maximum(
                    matrix,
                    epsilon,
                )
            ),
            axis=1,
        )
    )

    return output


def correlation_vector(
    frame: pd.DataFrame,
    columns: list[str],
    method: str,
) -> np.ndarray:
    matrix = frame[columns].corr(
        method=method
    ).fillna(0.0)

    values: list[float] = []

    for row_index in range(len(columns)):
        for column_index in range(
            row_index + 1,
            len(columns),
        ):
            values.append(
                float(
                    matrix.iloc[
                        row_index,
                        column_index,
                    ]
                )
            )

    return np.asarray(
        values,
        dtype=float,
    )


def calculate_crm(
    metrics: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = metrics.copy()
    output["CRM"] = np.nan

    crm_config = config["metrics"]["CRM"]

    selected_features = crm_config[
        "selected_features"
    ]

    history_hours = int(
        crm_config["rolling_history_hours"]
    )

    minimum_rows = int(
        crm_config["minimum_history_rows"]
    )

    method = str(
        crm_config["correlation_method"]
    )

    baseline_start = parse_utc(
        config["periods"][
            "baseline_start_utc"
        ]
    )

    baseline_end = parse_utc(
        config["periods"][
            "baseline_end_utc"
        ]
    )

    baseline_payload: dict[str, Any] = {}

    for window_id in config[
        "windows"
    ]["complete_window_order"]:

        group = output.loc[
            output["window_id"] == window_id
        ].sort_values(
            "endpoint_utc",
            kind="stable",
        ).copy()

        baseline = group.loc[
            (
                group["endpoint_utc"]
                >= baseline_start
            )
            & (
                group["endpoint_utc"]
                <= baseline_end
            )
        ]

        if len(baseline) < minimum_rows:
            fail(
                "Insufficient baseline rows for CRM: "
                f"{window_id}"
            )

        baseline_vector = correlation_vector(
            baseline,
            selected_features,
            method,
        )

        relation_count = len(
            baseline_vector
        )

        if relation_count == 0:
            fail(
                "CRM requires at least two features."
            )

        baseline_payload[window_id] = {
            "selected_features":
                selected_features,
            "correlation_method":
                method,
            "relation_count":
                relation_count,
            "upper_triangle_vector":
                baseline_vector.tolist(),
        }

        crm_values: list[float] = []

        for endpoint in group[
            "endpoint_utc"
        ]:
            history_start = (
                endpoint
                - pd.Timedelta(
                    hours=history_hours
                )
            )

            history = group.loc[
                (
                    group["endpoint_utc"]
                    > history_start
                )
                & (
                    group["endpoint_utc"]
                    <= endpoint
                )
            ]

            if len(history) < minimum_rows:
                crm_values.append(math.nan)
                continue

            current_vector = correlation_vector(
                history,
                selected_features,
                method,
            )

            distance = float(
                np.linalg.norm(
                    current_vector
                    - baseline_vector
                )
                / math.sqrt(
                    relation_count
                )
            )

            crm_values.append(distance)

        output.loc[
            group.index,
            "CRM",
        ] = crm_values

    return output, baseline_payload


def estimate_thresholds(
    metrics: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    baseline_start = parse_utc(
        config["periods"][
            "baseline_start_utc"
        ]
    )

    baseline_end = parse_utc(
        config["periods"][
            "baseline_end_utc"
        ]
    )

    threshold_config = config[
        "baseline_thresholds"
    ]

    thresholds: dict[str, Any] = {}

    for window_id in config[
        "windows"
    ]["complete_window_order"]:

        baseline_mask = (
            (
                metrics["window_id"]
                == window_id
            )
            & (
                metrics["endpoint_utc"]
                >= baseline_start
            )
            & (
                metrics["endpoint_utc"]
                <= baseline_end
            )
        )

        thresholds[window_id] = {}

        for metric_name in [
            "Gamma_CF",
            "SCI",
            "CRM",
        ]:
            metric_config = threshold_config[
                metric_name
            ]

            baseline_values = metrics.loc[
                baseline_mask,
                metric_name,
            ].dropna()

            if baseline_values.empty:
                fail(
                    "No baseline values for threshold: "
                    f"{window_id}/{metric_name}"
                )

            values = {
                "attention": float(
                    baseline_values.quantile(
                        float(
                            metric_config[
                                "attention_percentile"
                            ]
                        ) / 100.0
                    )
                ),
                "anomaly": float(
                    baseline_values.quantile(
                        float(
                            metric_config[
                                "anomaly_percentile"
                            ]
                        ) / 100.0
                    )
                ),
                "structural": float(
                    baseline_values.quantile(
                        float(
                            metric_config[
                                "structural_percentile"
                            ]
                        ) / 100.0
                    )
                ),
            }

            thresholds[window_id][
                metric_name
            ] = {
                "direction":
                    metric_config["direction"],
                "values":
                    values,
            }

        conventional = threshold_config[
            "conventional_upper_tail"
        ]

        thresholds[window_id][
            "conventional"
        ] = {}

        for metric_name in [
            "event_count",
            "maximum_magnitude",
            "log10_cumulative_energy_joule",
        ]:
            baseline_values = metrics.loc[
                baseline_mask,
                metric_name,
            ].dropna()

            thresholds[window_id][
                "conventional"
            ][metric_name] = {
                "attention": float(
                    baseline_values.quantile(
                        float(
                            conventional[
                                "attention_percentile"
                            ]
                        ) / 100.0
                    )
                ),
                "anomaly": float(
                    baseline_values.quantile(
                        float(
                            conventional[
                                "anomaly_percentile"
                            ]
                        ) / 100.0
                    )
                ),
                "structural": float(
                    baseline_values.quantile(
                        float(
                            conventional[
                                "structural_percentile"
                            ]
                        ) / 100.0
                    )
                ),
            }

    return thresholds


def threshold_condition(
    values: pd.Series,
    threshold: float,
    direction: str,
) -> pd.Series:
    if direction == "lower":
        return values <= threshold

    if direction == "upper":
        return values >= threshold

    fail(f"Unknown threshold direction: {direction}")

    return values.astype(bool)


def persistent_flags(
    flags: pd.Series,
    required: int,
) -> pd.Series:
    result: list[bool] = []
    run_length = 0

    for flag in flags.fillna(False):
        if bool(flag):
            run_length += 1
        else:
            run_length = 0

        result.append(
            run_length >= required
        )

    return pd.Series(
        result,
        index=flags.index,
        dtype=bool,
    )


def build_window_alerts(
    metrics: pd.DataFrame,
    thresholds: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "endpoint_utc",
        "window_id",
        "Gamma_CF",
        "SCI",
        "CRM",
        "event_count",
        "maximum_magnitude",
        "log10_cumulative_energy_joule",
    ]

    records: list[pd.DataFrame] = []

    for window_id in config[
        "windows"
    ]["complete_window_order"]:

        group = metrics.loc[
            metrics["window_id"] == window_id,
            columns,
        ].sort_values(
            "endpoint_utc",
            kind="stable",
        ).copy()

        for level in [
            "attention",
            "anomaly",
            "structural",
        ]:
            condition_columns: list[str] = []

            for metric_name in [
                "Gamma_CF",
                "SCI",
                "CRM",
            ]:
                threshold_data = thresholds[
                    window_id
                ][metric_name]

                column_name = (
                    f"{metric_name}_{level}"
                )

                group[column_name] = (
                    threshold_condition(
                        group[metric_name],
                        float(
                            threshold_data[
                                "values"
                            ][level]
                        ),
                        threshold_data[
                            "direction"
                        ],
                    )
                    .fillna(False)
                )

                condition_columns.append(
                    column_name
                )

            minimum_conditions = int(
                config[
                    "window_alert_rule"
                ][level][
                    "minimum_conditions"
                ]
            )

            raw_column = f"raw_{level}"

            group[raw_column] = (
                group[condition_columns]
                .sum(axis=1)
                >= minimum_conditions
            )

            required_persistence = int(
                config["persistence"][
                    f"{level}"
                    "_consecutive_endpoints"
                ]
            )

            group[
                f"persistent_{level}"
            ] = persistent_flags(
                group[raw_column],
                required_persistence,
            )

        def choose_level(row: pd.Series) -> str:
            if row[
                "persistent_structural"
            ]:
                return "structural"

            if row[
                "persistent_anomaly"
            ]:
                return "anomaly"

            if row[
                "persistent_attention"
            ]:
                return "attention"

            return "normal"

        group["window_alert_level"] = (
            group.apply(
                choose_level,
                axis=1,
            )
        )

        for comparator in [
            "event_count",
            "maximum_magnitude",
            "log10_cumulative_energy_joule",
        ]:
            for level in [
                "attention",
                "anomaly",
                "structural",
            ]:
                group[
                    f"{comparator}_{level}"
                ] = (
                    group[comparator]
                    >= float(
                        thresholds[
                            window_id
                        ]["conventional"][
                            comparator
                        ][level]
                    )
                )

        records.append(group)

    output = pd.concat(
        records,
        ignore_index=True,
    )

    window_order = {
        "1d": 0,
        "3d": 1,
        "7d": 2,
        "30d": 3,
    }

    output["_window_order"] = (
        output["window_id"].map(
            window_order
        )
    )

    return output.sort_values(
        by=[
            "endpoint_utc",
            "_window_order",
        ],
        kind="stable",
    ).drop(
        columns=["_window_order"]
    ).reset_index(drop=True)


def build_global_alerts(
    window_alerts: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    required_windows = int(
        config["global_alert_rule"][
            "required_confirming_windows"
        ]
    )

    primary_window = str(
        config["global_alert_rule"][
            "primary_window"
        ]
    )

    records: list[dict[str, Any]] = []

    for endpoint, group in window_alerts.groupby(
        "endpoint_utc",
        sort=True,
    ):
        levels = {
            str(row.window_id):
                str(row.window_alert_level)
            for row in group.itertuples()
        }

        structural_windows = [
            window_id
            for window_id, level
            in levels.items()
            if LEVEL_PRIORITY[level]
            >= LEVEL_PRIORITY["structural"]
        ]

        anomaly_windows = [
            window_id
            for window_id, level
            in levels.items()
            if LEVEL_PRIORITY[level]
            >= LEVEL_PRIORITY["anomaly"]
        ]

        attention_windows = [
            window_id
            for window_id, level
            in levels.items()
            if LEVEL_PRIORITY[level]
            >= LEVEL_PRIORITY["attention"]
        ]

        if (
            len(structural_windows)
            >= required_windows
            and primary_window
            in structural_windows
        ):
            global_level = "structural"
            confirming = structural_windows

        elif (
            len(anomaly_windows)
            >= required_windows
        ):
            global_level = "anomaly"
            confirming = anomaly_windows

        elif (
            len(attention_windows)
            >= required_windows
        ):
            global_level = "attention"
            confirming = attention_windows

        else:
            global_level = "normal"
            confirming = []

        record: dict[str, Any] = {
            "endpoint_utc":
                utc_string(endpoint),
            "global_alert_level":
                global_level,
            "confirming_window_count":
                len(confirming),
            "confirming_windows":
                ",".join(confirming),
        }

        for window_id in [
            "1d",
            "3d",
            "7d",
            "30d",
        ]:
            record[
                f"window_{window_id}_level"
            ] = levels.get(
                window_id,
                "missing",
            )

        records.append(record)

    return pd.DataFrame.from_records(
        records
    )


def build_alert_episodes(
    global_alerts: pd.DataFrame,
    reset_required: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    active = False
    start: pd.Timestamp | None = None
    last_alert: pd.Timestamp | None = None
    levels: list[str] = []
    normal_run = 0
    episode_id = 0

    for row in global_alerts.itertuples(
        index=False
    ):
        endpoint = parse_utc(
            row.endpoint_utc
        )

        level = str(
            row.global_alert_level
        )

        if level != "normal":
            normal_run = 0

            if not active:
                active = True
                episode_id += 1
                start = endpoint
                levels = []

            last_alert = endpoint
            levels.append(level)

        elif active:
            normal_run += 1

            if normal_run >= reset_required:
                maximum_level = max(
                    levels,
                    key=lambda item:
                        LEVEL_PRIORITY[item],
                )

                duration_hours = (
                    last_alert - start
                ).total_seconds() / 3600.0

                records.append(
                    {
                        "episode_id":
                            episode_id,
                        "start_utc":
                            utc_string(start),
                        "end_utc":
                            utc_string(last_alert),
                        "duration_hours":
                            float(duration_hours),
                        "maximum_level":
                            maximum_level,
                    }
                )

                active = False
                start = None
                last_alert = None
                levels = []
                normal_run = 0

    if active:
        maximum_level = max(
            levels,
            key=lambda item:
                LEVEL_PRIORITY[item],
        )

        duration_hours = (
            last_alert - start
        ).total_seconds() / 3600.0

        records.append(
            {
                "episode_id":
                    episode_id,
                "start_utc":
                    utc_string(start),
                "end_utc":
                    utc_string(last_alert),
                "duration_hours":
                    float(duration_hours),
                "maximum_level":
                    maximum_level,
            }
        )

    return pd.DataFrame.from_records(
        records,
        columns=[
            "episode_id",
            "start_utc",
            "end_utc",
            "duration_hours",
            "maximum_level",
        ],
    )


def first_alert_timestamp(
    alerts: pd.DataFrame,
    minimum_level: str,
) -> pd.Timestamp | None:
    minimum_priority = LEVEL_PRIORITY[
        minimum_level
    ]

    priorities = alerts[
        "global_alert_level"
    ].map(LEVEL_PRIORITY)

    qualifying = alerts.loc[
        priorities >= minimum_priority
    ]

    if qualifying.empty:
        return None

    return parse_utc(
        qualifying.iloc[0][
            "endpoint_utc"
        ]
    )


def classify_lead_time(
    lead_seconds: float | None,
    config: dict[str, Any],
) -> str:
    if lead_seconds is None:
        return "no_signal"

    classes = config[
        "lead_time"
    ]["classes"]

    for class_name in [
        "coincident",
        "short_hours",
        "short_days",
        "structural",
    ]:
        bounds = classes[class_name]

        if (
            lead_seconds
            >= float(
                bounds["minimum_seconds"]
            )
            and lead_seconds
            <= float(
                bounds["maximum_seconds"]
            )
        ):
            return class_name

    return "remote_or_nondiscriminating"


def build_summary(
    global_alerts: pd.DataFrame,
    episodes: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    target_event = parse_utc(
        config["target_event"]["utc_time"]
    )

    test_start = parse_utc(
        config["periods"]["test_start_utc"]
    )

    test_alerts = global_alerts.copy()

    test_alerts["endpoint_utc"] = pd.to_datetime(
        test_alerts["endpoint_utc"],
        utc=True,
        errors="raise",
        format="mixed",
    )

    test_alerts = test_alerts.loc[
        test_alerts["endpoint_utc"]
        >= test_start
    ].copy()

    first_attention = first_alert_timestamp(
        test_alerts,
        "attention",
    )

    first_anomaly = first_alert_timestamp(
        test_alerts,
        "anomaly",
    )

    first_structural = first_alert_timestamp(
        test_alerts,
        "structural",
    )

    first_qualifying = (
        first_structural
        or first_anomaly
        or first_attention
    )

    lead_seconds: float | None = None

    if first_qualifying is not None:
        lead_seconds = float(
            (
                target_event
                - first_qualifying
            ).total_seconds()
        )

    alert_fraction = (
        float(
            (
                test_alerts[
                    "global_alert_level"
                ]
                != "normal"
            ).mean()
        )
        if not test_alerts.empty
        else 0.0
    )

    permanent_threshold = float(
        config[
            "false_positive_evaluation"
        ][
            "permanent_alert_fraction_threshold"
        ]
    )

    permanent_alert = (
        alert_fraction
        >= permanent_threshold
    )

    lead_class = classify_lead_time(
        lead_seconds,
        config,
    )

    if permanent_alert:
        lead_class = (
            "remote_or_nondiscriminating"
        )

    if episodes.empty:
        median_duration = 0.0
        maximum_duration = 0.0
    else:
        median_duration = float(
            episodes[
                "duration_hours"
            ].median()
        )

        maximum_duration = float(
            episodes[
                "duration_hours"
            ].max()
        )

    return {
        "experiment_id":
            "CF_RETRO_01",
        "target_event_utc":
            utc_string(target_event),
        "first_attention_utc":
            (
                utc_string(first_attention)
                if first_attention is not None
                else None
            ),
        "first_anomaly_utc":
            (
                utc_string(first_anomaly)
                if first_anomaly is not None
                else None
            ),
        "first_structural_utc":
            (
                utc_string(first_structural)
                if first_structural is not None
                else None
            ),
        "first_qualifying_alert_utc":
            (
                utc_string(first_qualifying)
                if first_qualifying is not None
                else None
            ),
        "lead_time_seconds":
            lead_seconds,
        "lead_time_hours":
            (
                lead_seconds / 3600.0
                if lead_seconds is not None
                else None
            ),
        "lead_time_days":
            (
                lead_seconds / 86400.0
                if lead_seconds is not None
                else None
            ),
        "lead_time_class":
            lead_class,
        "test_alert_fraction":
            alert_fraction,
        "alert_episode_count":
            int(len(episodes)),
        "median_episode_duration_hours":
            median_duration,
        "maximum_episode_duration_hours":
            maximum_duration,
        "permanent_alert_threshold":
            permanent_threshold,
        "permanent_alert_detected":
            permanent_alert,
        "prospective_prediction_claim":
            False,
        "interpretation":
            (
                "Retrospective structural-instability "
                "evaluation only."
            ),
    }


def write_report(
    path: Path,
    summary: dict[str, Any],
) -> None:
    lead_hours = summary[
        "lead_time_hours"
    ]

    if lead_hours is None:
        lead_text = (
            "No qualifying pre-event alert."
        )
    else:
        lead_text = (
            f"{lead_hours:.2f} hours "
            f"({lead_hours / 24.0:.2f} days)"
        )

    report = f"""# CF-RETRO-01 — Retrospective Inference Report

## Scientific status

Experiment: CF-RETRO-01

Analysis type: retrospective structural-instability validation

Prospective prediction claim: no

## Target event

UTC:

{summary["target_event_utc"]}

## First qualifying alerts

Attention:

{summary["first_attention_utc"]}

Anomaly:

{summary["first_anomaly_utc"]}

Structural:

{summary["first_structural_utc"]}

## Lead time

{lead_text}

Classification:

{summary["lead_time_class"]}

## Alert burden

Test-period alert fraction:

{summary["test_alert_fraction"]:.6f}

Alert episodes:

{summary["alert_episode_count"]}

Median episode duration:

{summary["median_episode_duration_hours"]:.2f} hours

Maximum episode duration:

{summary["maximum_episode_duration_hours"]:.2f} hours

Permanent alert detected:

{summary["permanent_alert_detected"]}

## Interpretation boundary

This experiment evaluates whether frozen pre-event seismic observations
contained a detectable structural transition under rules committed
before inference.

A positive retrospective result does not demonstrate deterministic
earthquake prediction.

A no-signal, permanent-alert, excessive-alert or non-discriminating
result must be retained as a falsification or limitation.
"""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 78)
    print("CF-RETRO-01 — FROZEN RETROSPECTIVE INFERENCE")
    print("=" * 78)

    feature_config = load_yaml(
        FEATURE_CONFIG_PATH
    )

    inference_config = load_yaml(
        INFERENCE_CONFIG_PATH
    )

    validate_configuration(
        feature_config,
        inference_config,
    )

    features, input_hash = load_features(
        inference_config
    )

    print(f"Feature rows: {len(features)}")
    print(f"Input feature SHA-256: {input_hash}")
    print(
        "Feature interval: "
        f"{utc_string(features['endpoint_utc'].min())} "
        "through "
        f"{utc_string(features['endpoint_utc'].max())}"
    )

    normalized, baseline_parameters = (
        normalize_features(
            features,
            inference_config,
        )
    )

    metrics = calculate_gamma_and_sci(
        normalized,
        inference_config,
    )

    metrics, crm_baseline = calculate_crm(
        metrics,
        inference_config,
    )

    thresholds = estimate_thresholds(
        metrics,
        inference_config,
    )

    window_alerts = build_window_alerts(
        metrics,
        thresholds,
        inference_config,
    )

    global_alerts = build_global_alerts(
        window_alerts,
        inference_config,
    )

    episodes = build_alert_episodes(
        global_alerts,
        int(
            inference_config[
                "persistence"
            ][
                "reset_consecutive_endpoints"
            ]
        ),
    )

    summary = build_summary(
        global_alerts,
        episodes,
        inference_config,
    )

    outputs = inference_config[
        "outputs"
    ]

    normalized_path = resolve_path(
        outputs["normalized_features"]
    )

    baseline_parameters_path = resolve_path(
        outputs["baseline_parameters"]
    )

    metrics_path = resolve_path(
        outputs["metrics"]
    )

    thresholds_path = resolve_path(
        outputs["thresholds"]
    )

    window_alerts_path = resolve_path(
        outputs["window_alerts"]
    )

    global_alerts_path = resolve_path(
        outputs["global_alerts"]
    )

    episodes_path = resolve_path(
        outputs["alert_episodes"]
    )

    summary_path = resolve_path(
        outputs["summary"]
    )

    report_path = resolve_path(
        outputs["report"]
    )

    metrics_hash_path = resolve_path(
        outputs["metrics_hash"]
    )

    manifest_path = resolve_path(
        outputs["manifest"]
    )

    float_precision = int(
        inference_config[
            "reproducibility"
        ]["float_precision"]
    )

    write_csv(
        normalized,
        normalized_path,
        float_precision,
    )

    write_csv(
        metrics,
        metrics_path,
        float_precision,
    )

    write_csv(
        window_alerts,
        window_alerts_path,
        float_precision,
    )

    write_csv(
        global_alerts,
        global_alerts_path,
        float_precision,
    )

    write_csv(
        episodes,
        episodes_path,
        float_precision,
    )

    write_json(
        baseline_parameters_path,
        {
            "experiment_id":
                "CF_RETRO_01",
            "normalization_parameters":
                baseline_parameters,
            "crm_baseline":
                crm_baseline,
        },
    )

    write_json(
        thresholds_path,
        thresholds,
    )

    write_json(
        summary_path,
        summary,
    )

    write_report(
        report_path,
        summary,
    )

    metrics_hash = sha256_file(
        metrics_path
    )

    metrics_hash_path.write_text(
        f"{metrics_hash}  {metrics_path.name}\n",
        encoding="utf-8",
    )

    manifest = {
        "experiment_id":
            "CF_RETRO_01",
        "stage":
            "inference",
        "configuration_status":
            inference_config[
                "experiment"
            ]["status"],
        "execution_timestamp_utc":
            datetime.now(
                timezone.utc
            )
            .isoformat()
            .replace("+00:00", "Z"),
        "source_commit":
            get_git_commit(),
        "input_features_sha256":
            input_hash,
        "feature_configuration_sha256":
            sha256_file(
                FEATURE_CONFIG_PATH
            ),
        "inference_configuration_sha256":
            sha256_file(
                INFERENCE_CONFIG_PATH
            ),
        "inference_script_sha256":
            sha256_file(
                SCRIPT_PATH
            ),
        "metrics_sha256":
            metrics_hash,
        "feature_rows":
            int(len(features)),
        "metrics_rows":
            int(len(metrics)),
        "global_alert_rows":
            int(len(global_alerts)),
        "alert_episode_count":
            int(len(episodes)),
        "target_event_utc":
            inference_config[
                "target_event"
            ]["utc_time"],
        "frozen_cutoff_utc":
            inference_config[
                "target_event"
            ][
                "frozen_cutoff_utc"
            ],
        "target_event_included":
            False,
        "future_data_used":
            False,
        "normalization_period":
            "baseline_only",
        "threshold_period":
            "baseline_only",
        "summary":
            summary,
    }

    write_json(
        manifest_path,
        manifest,
    )

    print()
    print("=" * 78)
    print("INFERENCE COMPLETED")
    print("=" * 78)
    print(
        "First qualifying alert: "
        f"{summary['first_qualifying_alert_utc']}"
    )
    print(
        "Lead time hours: "
        f"{summary['lead_time_hours']}"
    )
    print(
        "Lead-time class: "
        f"{summary['lead_time_class']}"
    )
    print(
        "Test alert fraction: "
        f"{summary['test_alert_fraction']:.6f}"
    )
    print(
        "Alert episodes: "
        f"{summary['alert_episode_count']}"
    )
    print(
        "Permanent alert: "
        f"{summary['permanent_alert_detected']}"
    )
    print()
    print(f"Metrics: {metrics_path}")
    print(f"Metrics SHA-256: {metrics_hash}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    print(f"Manifest: {manifest_path}")
    print()
    print(
        "Interpretation boundary: retrospective "
        "structural-instability evaluation only."
    )


if __name__ == "__main__":
    main()

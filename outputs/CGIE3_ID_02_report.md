# CGIE3-ID-02 — Eligible Relation Discovery Report

## Status

`COMPLETED`

## Scientific outcome

`eligible_relations_identified`

## Purpose

CGIE3-ID-02 evaluates whether relations among the frozen
Campi Flegrei seismic feature set satisfy a predetermined
baseline eligibility protocol.

The experiment does not test deterministic earthquake prediction.

## Dataset boundary

- Selection interval: frozen 2025 baseline only
- Evaluation data used for selection: `false`
- Target-event labels used for selection: `false`
- Candidate relations evaluated: `136`

## Overall classification

| Status | Count |
|---|---:|
| `eligible` | 45 |
| `candidate` | 29 |
| `rejected` | 62 |
| `non_estimable` | 0 |

## Classification by temporal window

### Window `1d`

| Status | Count |
|---|---:|
| `eligible` | 9 |
| `candidate` | 3 |
| `rejected` | 16 |
| `non_estimable` | 0 |

### Window `30d`

| Status | Count |
|---|---:|
| `eligible` | 10 |
| `candidate` | 14 |
| `rejected` | 12 |
| `non_estimable` | 0 |

### Window `3d`

| Status | Count |
|---|---:|
| `eligible` | 11 |
| `candidate` | 6 |
| `rejected` | 19 |
| `non_estimable` | 0 |

### Window `7d`

| Status | Count |
|---|---:|
| `eligible` | 15 |
| `candidate` | 6 |
| `rejected` | 15 |
| `non_estimable` | 0 |


## Eligible relations

- `1d` — `depth_mad_km` ↔ `spatial_dispersion_km`; Spearman = `0.3004`; persistence = `0.8333`.
- `1d` — `event_count` ↔ `interevent_time_mad_hours`; Spearman = `0.4972`; persistence = `0.7500`.
- `1d` — `event_count` ↔ `log10_cumulative_energy_joule`; Spearman = `0.7587`; persistence = `1.0000`.
- `1d` — `event_count` ↔ `maximum_magnitude`; Spearman = `0.6751`; persistence = `1.0000`.
- `1d` — `event_count` ↔ `median_interevent_time_hours`; Spearman = `-0.5284`; persistence = `0.8333`.
- `1d` — `log10_cumulative_energy_joule` ↔ `maximum_magnitude`; Spearman = `0.9859`; persistence = `1.0000`.
- `1d` — `log10_cumulative_energy_joule` ↔ `median_interevent_time_hours`; Spearman = `-0.4366`; persistence = `0.8333`.
- `1d` — `maximum_magnitude` ↔ `median_interevent_time_hours`; Spearman = `-0.4042`; persistence = `0.8333`.
- `1d` — `median_interevent_time_hours` ↔ `spatial_dispersion_km`; Spearman = `0.4994`; persistence = `1.0000`.
- `30d` — `event_count` ↔ `interevent_time_mad_hours`; Spearman = `-0.7624`; persistence = `0.7500`.
- `30d` — `event_count` ↔ `log10_cumulative_energy_joule`; Spearman = `0.3148`; persistence = `0.8333`.
- `30d` — `interevent_time_mad_hours` ↔ `median_interevent_time_hours`; Spearman = `0.9768`; persistence = `1.0000`.
- `30d` — `interevent_time_mad_hours` ↔ `spatial_dispersion_km`; Spearman = `0.6003`; persistence = `0.7500`.
- `30d` — `interevent_time_mad_hours` ↔ `temporal_burstiness`; Spearman = `-0.8709`; persistence = `0.9167`.
- `30d` — `log10_cumulative_energy_joule` ↔ `maximum_magnitude`; Spearman = `0.9722`; persistence = `0.9167`.
- `30d` — `maximum_magnitude` ↔ `temporal_burstiness`; Spearman = `0.3920`; persistence = `0.8333`.
- `30d` — `median_depth_km` ↔ `spatial_dispersion_km`; Spearman = `0.7668`; persistence = `0.8333`.
- `30d` — `median_interevent_time_hours` ↔ `spatial_dispersion_km`; Spearman = `0.6228`; persistence = `0.8333`.
- `30d` — `median_interevent_time_hours` ↔ `temporal_burstiness`; Spearman = `-0.8710`; persistence = `0.8333`.
- `3d` — `event_count` ↔ `log10_cumulative_energy_joule`; Spearman = `0.7643`; persistence = `1.0000`.
- `3d` — `event_count` ↔ `maximum_magnitude`; Spearman = `0.6850`; persistence = `0.9167`.
- `3d` — `event_count` ↔ `median_interevent_time_hours`; Spearman = `-0.6986`; persistence = `1.0000`.
- `3d` — `event_count` ↔ `temporal_burstiness`; Spearman = `0.7263`; persistence = `0.8333`.
- `3d` — `interevent_time_mad_hours` ↔ `temporal_burstiness`; Spearman = `-0.6502`; persistence = `0.8333`.
- `3d` — `log10_cumulative_energy_joule` ↔ `maximum_magnitude`; Spearman = `0.9858`; persistence = `1.0000`.
- `3d` — `log10_cumulative_energy_joule` ↔ `median_interevent_time_hours`; Spearman = `-0.6193`; persistence = `0.8333`.
- `3d` — `log10_cumulative_energy_joule` ↔ `temporal_burstiness`; Spearman = `0.6035`; persistence = `0.9167`.
- `3d` — `maximum_magnitude` ↔ `median_interevent_time_hours`; Spearman = `-0.5691`; persistence = `0.8333`.
- `3d` — `maximum_magnitude` ↔ `temporal_burstiness`; Spearman = `0.5555`; persistence = `0.9167`.
- `3d` — `median_interevent_time_hours` ↔ `temporal_burstiness`; Spearman = `-0.8154`; persistence = `1.0000`.
- `7d` — `depth_mad_km` ↔ `median_depth_km`; Spearman = `-0.3650`; persistence = `0.9167`.
- `7d` — `event_count` ↔ `interevent_time_mad_hours`; Spearman = `-0.6884`; persistence = `0.7500`.
- `7d` — `event_count` ↔ `log10_cumulative_energy_joule`; Spearman = `0.7666`; persistence = `0.9167`.
- `7d` — `event_count` ↔ `maximum_magnitude`; Spearman = `0.6896`; persistence = `0.7500`.
- `7d` — `event_count` ↔ `median_interevent_time_hours`; Spearman = `-0.8110`; persistence = `0.9167`.
- `7d` — `event_count` ↔ `temporal_burstiness`; Spearman = `0.7136`; persistence = `0.9167`.
- `7d` — `interevent_time_mad_hours` ↔ `maximum_magnitude`; Spearman = `-0.5470`; persistence = `0.8333`.
- `7d` — `interevent_time_mad_hours` ↔ `median_interevent_time_hours`; Spearman = `0.8876`; persistence = `1.0000`.
- `7d` — `interevent_time_mad_hours` ↔ `temporal_burstiness`; Spearman = `-0.8310`; persistence = `1.0000`.
- `7d` — `log10_cumulative_energy_joule` ↔ `maximum_magnitude`; Spearman = `0.9821`; persistence = `1.0000`.
- `7d` — `log10_cumulative_energy_joule` ↔ `temporal_burstiness`; Spearman = `0.6072`; persistence = `0.8333`.
- `7d` — `maximum_magnitude` ↔ `temporal_burstiness`; Spearman = `0.5509`; persistence = `0.8333`.
- `7d` — `median_depth_km` ↔ `spatial_dispersion_km`; Spearman = `0.5173`; persistence = `0.8333`.
- `7d` — `median_interevent_time_hours` ↔ `temporal_burstiness`; Spearman = `-0.8976`; persistence = `1.0000`.
- `7d` — `spatial_dispersion_km` ↔ `temporal_burstiness`; Spearman = `-0.3687`; persistence = `0.7500`.

## Interpretation

An `eligible` relation is one that satisfies the frozen requirements
for baseline strength, estimability, sign preservation, persistence,
bootstrap uncertainty, leave-one-block-out robustness and
missingness robustness.

Eligibility does **not** establish that a relation is:

- causal;
- indispensable;
- a primary identity relation;
- physically sufficient;
- predictive of a specific earthquake.

## Scientific claim boundary

Eligible relations satisfy only the frozen CGIE3-ID-02 baseline eligibility protocol. Eligibility does not establish causality, indispensability, primary identity status or prediction of a specific earthquake.

## Explicit negative claims

- Primary relations established: `false`
- Indispensable relations established: `false`
- Causality established: `false`
- Predictive capability established: `false`
- Earthquake prediction established: `false`

## Reproducibility

The official output package includes:

- candidate relations;
- monthly block estimates;
- bootstrap replications;
- frozen relation classifications;
- equivalence and redundancy flags;
- summary;
- report;
- manifest.

Rejected and non-estimable relations remain preserved.

## Generated

`2026-08-05T13:19:12.181838Z`

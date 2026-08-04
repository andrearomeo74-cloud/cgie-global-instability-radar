# CGIE3-ID-01 Identity Declaration Audit

## Status

VALID

## System

- System ID: `campi_flegrei_seismic_system`
- Name: Campi Flegrei Seismic System
- Domain: `geophysical_seismic_monitoring`
- Protocol version: `CGIE3_ID_01_v1.0`

## Functional purpose

Represent and evaluate the continuity of the multiscale spatiotemporal organization of seismic activity within the declared Campi Flegrei observation boundary.

## System boundary

The system includes seismic events retained by the frozen CF_RETRO_01 Campi Flegrei catalogue and the features derived from those events under the frozen geographical, temporal and quality-control rules. The individual target earthquake is not itself the system identity.

## Observation context

Retrospective experimental evaluation using the frozen CF_RETRO_01 lineage. The baseline interval is 2025-01-01 through 2025-12-31. The evaluation interval is 2026-01-01 through 2026-07-31T17:46:42Z. All estimates at time t must use only information available at or before t.

## Temporal scales

- `1d`
- `3d`
- `7d`
- `30d`

## Declared components

- `event_count`
- `maximum_magnitude`
- `log10_cumulative_energy_joule`
- `median_depth_km`
- `depth_mad_km`
- `spatial_dispersion_km`
- `median_interevent_time_hours`
- `interevent_time_mad_hours`
- `temporal_burstiness`

## Excluded interpretations

- `deterministic_earthquake_prediction`
- `exact_event_time_prediction`
- `exact_event_magnitude_prediction`
- `exact_event_location_prediction`
- `causal_precursor_claim`
- `operational_public_warning`
- `evacuation_advice`
- `replacement_of_official_authorities`
- `retrospective_result_presented_as_prospective`
- `non_estimable_state_interpreted_as_normal`

## Provenance

- Generated at UTC: `2026-08-04T15:36:05.355595Z`
- Source commit: `7d668067bda7ecb8e9210f8e5c85d74c4ed35bf8`
- Declaration SHA-256: `4c68d1d701238f3d96aaa9a3e1bc7a5a4556437a72390d9c0750eabc4ac7e180`
- Validator SHA-256: `e3239500d6bc87359caf8a2d3cdc739e6798aedc52bdcd4f09e2abcf9ac4991e`

## Interpretation boundary

This audit confirms only that the YAML declaration satisfies the
Congruity Core structural contract.

It does not establish that the declared candidate identity is
physically correct, indispensable, predictive or scientifically
validated.

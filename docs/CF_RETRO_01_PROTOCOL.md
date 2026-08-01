# CF-RETRO-01
## Frozen Retrospective Validation Protocol

Version: 1.0

Status:
FROZEN

------------------------------------------------------------

# Objective

Evaluate whether the Campi Flegrei volcanic system showed a measurable
loss of structural continuity before the earthquake of:

31 July 2026
19:46:43 Europe/Rome

This protocol does NOT attempt to predict earthquakes.

It evaluates whether Congruity metrics identify
a structural transition before the observable event.

------------------------------------------------------------

# Target Event

Local time

2026-07-31
19:46:43

UTC

2026-07-31
17:46:43

------------------------------------------------------------

# Data Cutoff

No information occurring AFTER

2026-07-31 17:46:43 UTC

may be used during feature generation.

------------------------------------------------------------

# Geographic Domain

Reference coordinates

Latitude
40.8315

Longitude
14.1402

Radius

15 km

------------------------------------------------------------

# Baseline

2025-01-01

through

2025-12-31

------------------------------------------------------------

# Test Interval

2026-01-01

through

2026-07-31
17:46:42 UTC

------------------------------------------------------------

# Initial Dataset

Only public earthquake catalogue information.

Required variables

event_id

time

latitude

longitude

depth

magnitude

magnitude_type

------------------------------------------------------------

# Initial Time Windows

1 day

3 days

7 days

30 days

------------------------------------------------------------

# Conventional Metrics

Event count

Maximum magnitude

Median magnitude

Cumulative energy

Median depth

Depth dispersion

Spatial dispersion

Median interevent interval

------------------------------------------------------------

# Congruity Metrics

Gamma_CF

CRM

SCI

------------------------------------------------------------

# Frozen Thresholds

Attention

90th percentile

Anomaly

95th percentile

Structural Alert

99th percentile

Persistence

Two consecutive windows

Reset

Three consecutive normal windows

------------------------------------------------------------

# Mandatory Outputs

CF_RETRO_01_catalog_frozen.csv

CF_RETRO_01_features.csv

CF_RETRO_01_metrics.csv

CF_RETRO_01_alerts.csv

CF_RETRO_01_report.md

------------------------------------------------------------

# Outcome Classes

A
No signal

B
Coincident signal

C
Short anticipation

D
Structural anticipation

E
Permanent alert

F
Excessive false positives

------------------------------------------------------------

# Falsification

The experiment is considered falsified if

• no signal appears before the event

• the signal appears only after the earthquake

• the signal is equivalent to simple event counting

• the alert remains permanently active

• excessive false positives occur

• parameters are modified after inspecting results

------------------------------------------------------------

# Reproducibility

Every execution must record

protocol_version

execution_timestamp

input_hash

configuration_hash

code_commit

baseline_period

test_period

event_cutoff

------------------------------------------------------------

# Interpretation

A positive retrospective result does NOT constitute
prospective earthquake prediction.

Prospective validation begins only after activation
of a timestamped live monitoring pipeline.

End of Protocol.

# CF-PROS-02 Frozen Prospective Decision Protocol

## Purpose

Define the deterministic decision process that transforms prospective monitoring outputs into operational evidence.

Unlike CF-PROS-01, which specifies how monitoring is performed, CF-PROS-02 specifies when evidence becomes actionable.

---

# Inputs

The protocol receives only frozen outputs produced by previous workflows.

Required inputs:

- prospective monitoring outputs
- event catalogue
- alert episodes
- configuration
- evidence summaries

No external information may be introduced.

---

# Decision Principle

The protocol never predicts earthquakes.

It evaluates whether the current observational state satisfies predefined evidence criteria.

The decision process is deterministic.

---

# Decision Levels

Level 0

Normal monitoring.

No operational evidence.

---

Level 1

Weak anomaly.

Continue monitoring.

No public interpretation.

---

Level 2

Persistent anomaly.

Evidence increases.

Internal review permitted.

---

Level 3

Strong prospective evidence.

Frozen evidence report generated.

Prospective record archived.

---

# Forbidden Operations

The protocol must never:

- modify thresholds
- modify historical data
- tune parameters after observing outputs
- introduce expert judgement
- remove negative evidence
- remove failed runs

---

# Evidence Preservation

Every execution must preserve:

- workflow version
- configuration hash
- input hashes
- execution time
- git commit
- generated outputs

Nothing may be overwritten.

---

# Interpretation Boundary

The protocol evaluates only statistical evidence.

No physical causality is claimed.

No deterministic earthquake prediction is claimed.

---

# Reproducibility

Given identical frozen inputs, identical configuration and identical software revision, the protocol must always produce identical decisions.

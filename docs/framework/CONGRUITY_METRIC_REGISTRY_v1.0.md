# Congruity Metric Registry v1.0

## Status

DRAFT — CANONICAL METRIC REGISTRY

This document records the official meaning, interpretation,
compatibility status and implementation boundary of every metric
used within the Congruity research programme.

No software module may assign a new formula to an existing acronym
unless the change is explicitly versioned in this registry.

---

# 1. Purpose

The Metric Registry prevents semantic drift across:

- domains;
- experiments;
- software versions;
- publications;
- reports;
- patent-related implementations.

Every metric entry must distinguish:

1. canonical conceptual meaning;
2. mathematical implementation;
3. direction of interpretation;
4. valid range;
5. missingness behavior;
6. domain-specific adaptations;
7. compatibility with previous versions;
8. validation status.

---

# 2. Metric status classes

Each metric must be classified as one of the following.

## `canonical`

The metric has an established meaning that must not be changed
without a versioned revision.

## `domain_implementation`

The metric is a domain-specific implementation of a canonical
concept.

## `experimental`

The metric is under evaluation and must not be presented as
universally validated.

## `deprecated`

The metric or formula is retained only for historical
reproducibility.

## `reserved`

The acronym is protected from reuse while its canonical
definition is being finalized.

---

# 3. ICᵀ

## Full name

`Indice di Congruità Totale`

## Status

`canonical`

## Conceptual meaning

ICᵀ evaluates the proportional congruity of an action, state or
configuration by relating generated value and enabling factors to
cost, distance, energy, incoherence and normalization terms.

## Canonical formula

\[
IC^T =
\frac{V \cdot F \cdot CCI}
{(C+1)(D+1)(E+1)}
\cdot
\frac{1}{ICC}
\cdot
N
\]

where:

- \(V\): generated or preserved value;
- \(F\): enabling or functional factor;
- \(CCI\): contextual congruity coefficient;
- \(C\): cost;
- \(D\): distance;
- \(E\): energy or effort;
- \(ICC\): incoherence or critical incompatibility coefficient;
- \(N\): normalization factor.

## Direction

Higher values indicate greater proportional congruity under the
declared implementation.

## Range

Not universally bounded unless the implementation explicitly
normalizes the variables.

## CGIE-3 use

ICᵀ must not be calculated in CGIE-3 until seismic-domain meanings
for all variables have been independently justified and frozen.

No direct substitution of seismic features into the formula is
permitted merely to obtain a numerical score.

## Validation status

Canonical as a framework metric.

Its seismic implementation remains unvalidated.

---

# 4. Γ

## Full name

`Gamma continuity operator`

## Status

`canonical_concept / domain_implementation_required`

## Conceptual meaning

Γ represents continuity under the aggregation rule appropriate to
the declared system and failure geometry.

In some validated implementations, Γ has represented the weakest
or limiting continuity component.

## Interpretation boundary

Γ is not universally equal to:

- a raw minimum;
- a mean;
- a probability;
- an anomaly score.

The aggregation rule must be justified by the system architecture.

## Direction

Higher Γ generally represents greater continuity.

Lower Γ generally represents a limiting loss of continuity.

## CGIE-2 implementation

The frozen CGIE-2 implementation must remain documented under its
original configuration and must not be silently replaced.

## CGIE-3 use

CGIE-3 may define a relational Γ only through a separate frozen
metric specification.

The relational implementation must remain distinguishable from
the feature-level CGIE-2 Γ.

---

# 5. SCI

## Full name

`Structural Continuity Index`

## Status

`canonical_concept / implementation_version_required`

## Conceptual meaning

SCI measures the preservation of system structure through time.

The structure may refer to:

- feature organization;
- relational organization;
- topology;
- functional organization;
- cross-scale coherence.

The declared structural object must always be specified.

## Direction

Higher SCI indicates greater structural continuity.

Lower SCI indicates greater structural degradation.

## Prohibited uses

SCI must not be used as an undefined synonym for:

- correlation;
- identity;
- stability;
- resilience;
- average feature continuity.

## CGIE-3 use

CGIE-3 must explicitly label the structural object, for example:

- `SCI_feature`;
- `SCI_relation`;
- `SCI_identity_candidate`.

The canonical acronym `SCI` alone must not be emitted until the
implementation has passed an equivalence and compatibility audit.

---

# 6. CRM

## Full name

`Congruity Relational Metric`

## Status

`reserved_pending_canonical_formula_audit`

## Conceptual meaning

CRM represents relational organization or relational change within
the Congruity research programme.

Previous implementations have used CRM to characterize changes in
the organization of relations among system variables.

## Current restriction

The acronym CRM must not yet be assigned a new universal formula
inside CGIE-3.

The seismic implementation must initially use explicit technical
names such as:

- `edge_strength_deviation`;
- `relational_sign_loss`;
- `network_geometry_distance`;
- `topology_change_score`.

## Direction

Not frozen universally because previous implementations may use
different orientations.

Each historical implementation must be audited before the
canonical direction is declared.

## Required action

Before CRM becomes operational in CGIE-3, document:

- previous formulas;
- domains in which each formula was used;
- whether high values represented continuity or change;
- empirical equivalences;
- incompatibilities;
- proposed canonical version.

---

# 7. APS

## Full name

`Reserved canonical acronym`

## Status

`reserved_pending_semantic_audit`

## Problem

APS has been used or proposed with multiple possible meanings,
including:

- admissible phase space;
- acceleration-related quantities;
- alert persistence;
- other domain-specific interpretations.

This ambiguity must be resolved before implementation.

## Current restriction

CGIE-3 must not output a column named `APS`.

Use explicit temporary names instead:

- `continuity_velocity`;
- `continuity_acceleration`;
- `admissible_state_fraction`;
- `alert_persistence_hours`.

## Required action

The canonical APS definition must be recovered from the completed
cross-domain work and frozen in a future registry revision.

---

# 8. MFAC

## Full name

`Minimum Functional Admissible Continuity`

## Status

`canonical_concept / formula_not_universal`

## Conceptual meaning

MFAC represents the continuity of the weakest functionally
admissible structure necessary for preservation of the declared
system function.

MFAC is not automatically the minimum of every measured quantity.

The minimum must operate over a justified admissible functional
set.

## Direction

Higher MFAC indicates that limiting functional continuity remains
preserved.

Lower MFAC indicates degradation of at least one limiting
functional structure.

## Required elements

Every MFAC implementation must define:

- functional declaration;
- admissible relation set;
- aggregation geometry;
- redundancy rule;
- missingness rule;
- non-identifiability rule;
- threshold interpretation.

## CGIE-3 use

The first CGIE-3 experiment may calculate only a provisional
technical metric named:

`minimum_primary_relation_continuity`

It may be mapped to MFAC only after:

- primary relations are reproducibly identified;
- redundancy is evaluated;
- leave-one-edge-out ablations are completed;
- placebo discrimination is demonstrated;
- the functional admissibility claim is justified.

---

# 9. IEC

## Full name

`Identity Edge Continuity`

## Status

`experimental`

## Conceptual meaning

IEC describes continuity of one eligible candidate identity
relation relative to its frozen reference behavior.

## Required components

IEC may consider:

- relational-strength preservation;
- sign preservation;
- direction preservation;
- persistence;
- uncertainty;
- estimability.

## Formula status

No universal formula is frozen in Registry v1.0.

The first formula must be specified in:

`CGIE3_ID_01_METRIC_SPECIFICATION.md`

before experimental execution.

## Direction

Higher IEC must indicate greater edge continuity.

## Missingness

A non-estimable edge must not automatically receive IEC equal to
zero or one.

It must receive:

- `value: null`;
- `estimability_status: non_estimable`;
- an explicit reason.

---

# 10. INC

## Full name

`Identity Network Continuity`

## Status

`experimental`

## Conceptual meaning

INC describes continuity of the candidate identity network as a
whole.

## Aggregation requirements

INC must account for:

- primary relations;
- secondary relations;
- redundancy;
- concentration of degradation;
- connected components;
- uncertainty;
- non-estimability.

## Formula status

No universal formula is frozen in Registry v1.0.

A simple unweighted mean is not accepted by default.

## Direction

Higher INC indicates greater network continuity.

---

# 11. RCR

## Full name

`Relational Closure Residue`

## Status

`experimental_reserved`

## Conceptual meaning

RCR is intended to represent residual relational inconsistency
after accounting for the preserved candidate identity network.

## Restriction

RCR must not be used for alert generation in CGIE3-ID-01.

It may be calculated only as an exploratory diagnostic after a
formal metric specification is frozen.

---

# 12. Technical metrics permitted before canonical mapping

CGIE-3 may initially calculate the following explicitly named
technical quantities:

- `edge_strength_deviation`;
- `edge_sign_preservation`;
- `edge_estimability`;
- `edge_temporal_persistence`;
- `network_density_change`;
- `connected_component_change`;
- `node_participation_change`;
- `network_geometry_distance`;
- `continuity_velocity`;
- `continuity_acceleration`;
- `minimum_primary_relation_continuity`;
- `multiscale_confirmation_count`;
- `alert_persistence_hours`;
- `recovery_duration_hours`.

These names must be used until evidence supports mapping them onto
canonical Congruity metrics.

---

# 13. Metric implementation requirements

Every implemented metric must declare:

```yaml
metric_id:
full_name:
registry_version:
status:
conceptual_definition:
formula:
input_variables:
valid_range:
direction:
missingness_rule:
estimability_rule:
baseline_dependency:
future_data_allowed: false
uncertainty_method:
domain:
implementation_version:
compatibility:
```

No official output may omit the metric implementation version.

---

# 14. Equivalence audit

Before accepting a new metric, the implementation must test
whether it is equivalent to an existing quantity through:

- exact equality;
- monotonic dependence;
- rank correlation;
- deterministic transformation;
- identical alert classification;
- conditional information analysis.

A complex metric that does not add measurable information must be
reported as equivalent.

---

# 15. Naming boundary

The following names are prohibited unless defined by this registry
or an approved metric specification:

- stability score;
- identity score;
- resilience score;
- congruity score;
- predictive score;
- earthquake probability.

Generic names may conceal undefined formulas and must not appear in
official CGIE-3 outputs.

In particular, `ISS` is not an approved CGIE-3 metric.

---

# 16. Initial CGIE3-ID-01 metric set

The first experiment may implement:

1. `edge_strength_deviation`;
2. `edge_sign_preservation`;
3. `edge_estimability`;
4. `edge_temporal_persistence`;
5. experimental `IEC`;
6. `network_geometry_distance`;
7. `minimum_primary_relation_continuity`;
8. `continuity_velocity`;
9. `continuity_acceleration`;
10. `multiscale_confirmation_count`.

The experiment must not initially claim validated implementations
of:

- ICᵀ;
- CRM;
- APS;
- MFAC;
- SCI.

Mapping to those metrics requires later evidence and an updated
registry.

---

# 17. Versioning rule

Any semantic or mathematical change requires:

- a new registry version;
- a change log;
- compatibility notes;
- migration instructions;
- identification of affected experiments;
- preservation of previous formulas for reproducibility.

---

# 18. Registry boundary

This registry preserves conceptual and semantic consistency.

It does not claim that every listed metric has already been
scientifically validated.

A metric becomes validated only through frozen experiments,
negative controls, independent replication and explicit
falsification testing.

# Congruity Framework Specification v1.0

## Status

DRAFT — PRE-IMPLEMENTATION SPECIFICATION

This document defines the conceptual architecture, invariants,
interfaces and validation boundaries of the Congruity Framework.

It must be reviewed before being marked `FROZEN`.

No experimental result may be used to modify the framework
after the corresponding implementation protocol has been frozen.

---

# 1. Purpose

The Congruity Framework is a general methodology for evaluating
whether a complex system preserves the relational organization
required to maintain its functional identity through time.

The framework does not assume that:

- every system possesses one unique identity;
- every measured variable is functionally relevant;
- every correlation is a necessary relation;
- instability necessarily produces a critical event;
- identity degradation predicts the exact time or form of failure.

The framework tests whether relational continuity provides
measurable information beyond conventional state variables and
domain-specific indicators.

---

# 2. Core scientific question

For a system observed through time:

> Which relationships must remain sufficiently preserved for the
> system to continue functioning as the same system?

The operational subquestions are:

1. What is the system's functional identity?
2. Which observable relations participate in that identity?
3. Which relations are stable, redundant, replaceable or
   non-identifiable?
4. How can relational continuity be measured?
5. Does degradation precede an observable critical transition?
6. Does the Congruity Framework add information beyond existing
   methods?
7. Under which conditions does the hypothesis fail?

---

# 3. System representation

At time \(t\), a system is represented as:

\[
S_t = (X_t, R_t, C_t, B_t)
\]

where:

- \(X_t\) is the set of observable variables or components;
- \(R_t\) is the set of estimated relations among them;
- \(C_t\) is the observational and environmental context;
- \(B_t\) is the admissible operational boundary.

The framework distinguishes the observed system state from its
functional identity.

The observed state may change substantially while functional
identity remains continuous.

Conversely, marginal feature values may remain apparently normal
while the relational organization begins to degrade.

---

# 4. Functional identity

The functional identity of a system is defined operationally as:

> The minimal relational organization whose preservation is
> sufficient for the system to continue performing its declared
> function within the declared observational boundary.

Functional identity is therefore conditional on:

- the declared system;
- the declared function;
- the observation scale;
- the temporal resolution;
- the context;
- the available evidence.

Identity must not be inferred from outcome labels alone.

---

# 5. Identity declaration

Every implementation must provide an Identity Declaration
containing:

- system name;
- system boundary;
- functional purpose;
- observation interval;
- temporal scale;
- component or feature set;
- relevant external context;
- known failure or transition modes;
- excluded interpretations;
- authoritative domain references.

For seismic monitoring, the identity is not the individual
earthquake.

The initial candidate identity is:

> The multiscale spatiotemporal organization of seismic activity
> within a declared geographical and temporal boundary.

This definition remains a testable working declaration, not an
established physical law.

---

# 6. Relational representation

Relations may include:

- statistical dependence;
- temporal dependence;
- directional influence;
- phase coordination;
- spatial organization;
- conditional dependence;
- topological participation;
- functional coupling;
- domain-specific constraints.

A relation must include at least:

- source;
- target;
- estimator;
- sign or direction, where applicable;
- strength;
- uncertainty;
- estimability;
- persistence;
- temporal support;
- provenance.

A raw correlation is not automatically a functional relation.

---

# 7. Relation classes

The framework distinguishes five relation classes.

## 7.1 Candidate relation

A relation measurable from the declared inputs.

## 7.2 Eligible relation

A candidate relation that satisfies frozen requirements for:

- observational support;
- estimability;
- persistence;
- reproducibility;
- missingness;
- absence of future-label leakage.

## 7.3 Primary identity relation

An eligible relation retained by the frozen deterministic
selection procedure as part of the candidate identity network.

The term `primary` does not yet mean indispensable.

## 7.4 Secondary identity relation

An eligible relation retained for redundancy, sensitivity or
alternative-path analysis.

## 7.5 Non-identifiable relation

A relation for which available evidence cannot support a stable
classification.

Non-identifiability must not be converted into zero continuity.

---

# 8. Indispensability boundary

A relation may be called indispensable only after evidence from
ablation or intervention shows that its removal causes a
reproducible loss of functional identity that cannot be
compensated by alternative relations.

Indispensability cannot be established by:

- high correlation alone;
- high centrality alone;
- temporal proximity to an event;
- selection by predictive performance on one dataset;
- one retrospective case study.

Until this condition is satisfied, CGIE-3 must use the terms:

- candidate relation;
- eligible relation;
- primary relation;
- secondary relation.

---

# 9. Reference identity

A reference identity is estimated from a frozen baseline interval:

\[
I^{ref} = (V^{ref}, E^{ref}, W^{ref}, G^{ref})
\]

where:

- \(V^{ref}\) is the frozen node set;
- \(E^{ref}\) is the eligible edge set;
- \(W^{ref}\) contains reference relational properties;
- \(G^{ref}\) contains reference network geometry.

The reference identity must be estimated without access to future
target-event labels.

The baseline must be:

- temporally prior to evaluation;
- explicitly declared;
- hashed;
- reproducible;
- tested for internal stability.

If no stable reference identity can be estimated, the correct
result is:

`unstable_identity`

---

# 10. Identity continuity

Identity continuity describes the degree to which the current
relational organization remains compatible with the frozen
reference identity.

For each eligible relation \(e\), an edge-continuity quantity must
consider, where applicable:

- strength preservation;
- sign preservation;
- directional preservation;
- temporal persistence;
- estimability;
- uncertainty;
- contextual admissibility.

The general form is:

\[
IEC_e(t) =
F_e(
\Delta strength,
sign,
direction,
persistence,
estimability,
uncertainty
)
\]

The exact function \(F_e\) must be frozen in the experiment
configuration before outcome inspection.

No universal numerical formula is asserted in this specification.

---

# 11. Network continuity

Identity Network Continuity evaluates preservation of the
candidate identity network as a whole.

It must not be reduced automatically to a simple arithmetic mean.

The aggregation procedure must account for:

- primary versus secondary relations;
- redundancy;
- connected components;
- alternative relational paths;
- uncertainty;
- missing or non-estimable relations;
- concentration of degradation.

The implementation must distinguish:

- diffuse weak degradation;
- localized critical degradation;
- global identity deformation;
- loss of estimability;
- recovery.

---

# 12. MFAC

MFAC is reserved for the canonical cross-domain metric:

`Minimum Functional Admissible Continuity`

Within CGIE-3, MFAC must represent the continuity of the weakest
functionally admissible relational structure under the frozen
aggregation rule.

MFAC must not be defined as:

- an arbitrary transformation of another metric;
- the minimum of all raw correlations;
- the minimum of all measured features;
- a post hoc selected relation;
- a quantity chosen using the target event.

The exact CGIE-3 implementation of MFAC must be stated in a
separate metric specification.

---

# 13. CRM

CRM retains the canonical meaning already established in the
Congruity research programme.

CGIE-3 must not silently redefine CRM.

A separate Metric Registry must specify:

- full name;
- canonical formula;
- direction of interpretation;
- valid range;
- missingness behavior;
- relation to previous implementations;
- version history.

Until that registry is frozen, implementations must use explicit
technical names such as:

- `network_geometry_distance`;
- `edge_strength_deviation`;
- `relational_sign_loss`;

rather than assigning them to CRM.

---

# 14. APS

APS retains the canonical meaning already established in the
Congruity research programme.

APS must not be used interchangeably for:

- acceleration;
- admissible phase space;
- anomaly probability;
- alert persistence.

Its exact meaning and formula must be imported from the canonical
Metric Registry before software implementation.

Temporary technical variables must use unambiguous names such as:

- `continuity_velocity`;
- `continuity_acceleration`;
- `admissible_state_fraction`;
- `alert_persistence_hours`.

---

# 15. SCI

SCI retains the canonical cross-domain definition.

The CGIE-3 implementation must explicitly distinguish between:

- feature-level continuity;
- relational continuity;
- structural continuity;
- identity continuity.

An identity metric must not be declared equivalent to SCI without
an empirical equivalence test.

---

# 16. Metric naming rule

No existing acronym may receive a new formula without:

1. a canonical definition;
2. a version number;
3. compatibility notes;
4. a migration rule;
5. an explicit equivalence or non-equivalence declaration.

This rule prevents semantic drift across domains and experiments.

---

# 17. Temporal evaluation

CGIE-3 must evaluate relational identity through time using only
information available at each timestamp.

For a timestamp \(t\):

\[
I_t = H(X_{\leq t}, C_{\leq t})
\]

Future observations must not contribute to \(I_t\).

Rolling calculations must explicitly declare:

- lookback period;
- minimum observations;
- update cadence;
- warm-up interval;
- missingness rule;
- reset rule;
- recovery rule.

---

# 18. Multiscale identity

A system may preserve identity at one temporal scale while
degrading at another.

CGIE-3 must therefore represent each frozen scale independently.

For the initial seismic implementation:

- 1 day;
- 3 days;
- 7 days;
- 30 days.

The 7-day scale is the frozen primary window inherited from
CF_RETRO_01.

Cross-scale states include:

- isolated;
- confirmed;
- nested;
- contradictory;
- persistent;
- recovering.

No additional window may be introduced after target inspection in
the frozen experiment.

---

# 19. Identity state hierarchy

CGIE-3 uses the following interpretation states:

## `normal`

Reference identity is sufficiently preserved.

## `identity_watch`

Early or isolated degradation exists, but evidence is not yet
multiscale or persistent.

## `identity_degradation`

Relational identity loss is persistent, concentrated or confirmed
across frozen scales.

## `identity_transition`

The current organization is no longer compatible with the frozen
reference identity under the declared decision rule.

These states describe the measured system representation.

They do not constitute earthquake predictions or public-safety
instructions.

---

# 20. Admissibility

Admissibility determines which interpretations or actions remain
justified by the available evidence.

An analytical result may be:

- computable but not interpretable;
- statistically unusual but not functionally relevant;
- temporally associated but not discriminating;
- structurally coherent but not predictive;
- informative but insufficient for action.

The framework therefore separates:

\[
measurement \neq interpretation \neq action
\]

CGIE-3 may generate an identity state without authorizing a
prediction claim.

---

# 21. LIMEN boundary

LIMEN acts after analytical inference and before external action
or communication.

For CGIE-3, LIMEN must block:

- deterministic earthquake predictions;
- claims of exact timing;
- claims of exact magnitude;
- claims of exact location;
- evacuation advice;
- replacement of official authorities;
- retrospective language presented as prospective evidence;
- alerts derived from non-frozen or incomplete inputs.

Permitted outputs include:

- observational state;
- relational degradation;
- uncertainty;
- provenance;
- comparison with controls;
- falsification status.

---

# 22. Comparison requirement

CGIE-3 must be evaluated against:

- frozen CGIE-2 outputs;
- conventional seismic activity measures;
- event-time placebo controls;
- randomized relational controls;
- simplified baselines;
- ablated network variants.

Increased complexity is not evidence of improvement.

CGIE-3 adds information only when it demonstrates measurable
benefit conditional on simpler alternatives.

---

# 23. Primary validation question

The primary validation question is:

> Does relational identity continuity improve temporal
> discrimination relative to the frozen CGIE-2 engine and its own
> placebo distribution?

Success requires both:

1. improvement relative to CGIE-2;
2. superiority to frozen controls.

Lead time alone is insufficient.

---

# 24. Required negative controls

Each frozen implementation must include, where mathematically
admissible:

- circular event-time shifts;
- random event-time shifts;
- node-label permutations;
- edge-label permutations;
- baseline perturbation;
- primary-edge leave-one-out;
- feature-family ablation;
- secondary-edge removal;
- simplified aggregation baseline;
- missingness stress test.

---

# 25. Falsification outcomes

A CGIE-3 experiment must be allowed to conclude:

- `supported_preliminary`;
- `equivalent`;
- `non_discriminating`;
- `unstable_identity`;
- `inconclusive_low_power`;
- `falsified_in_current_configuration`.

The hypothesis is not supported when:

- no stable identity is estimable;
- identity relations are not reproducible;
- performance does not exceed placebo;
- apparent improvement results from alert inflation;
- improvement depends on target-informed selection;
- results collapse under minor admissible perturbations;
- identity metrics duplicate CGIE-2;
- missingness drives the state classification.

---

# 26. Equivalence

CGIE-3 must test whether a proposed metric is operationally
equivalent to an existing metric.

Equivalence includes:

- exact equality;
- monotonic transformation;
- near-perfect deterministic dependence;
- identical alert decisions;
- no conditional information gain;
- no improved discrimination.

An equivalent result must be reported as equivalence, not novelty.

---

# 27. Provenance

Every official run must record:

- experiment ID;
- protocol version;
- framework version;
- metric-registry version;
- Git commit;
- configuration hash;
- input file hashes;
- baseline interval;
- evaluation interval;
- feature list;
- relation estimator;
- eligible edges;
- primary edges;
- random seed;
- dependency versions;
- output hashes;
- execution timestamp.

---

# 28. Modular architecture

The implementation architecture is:

```text
CGIE-3
│
├── IDENTITY
│   └── system and function declaration
│
├── INPUT
│   └── frozen data validation
│
├── RELATION
│   └── relational estimation
│
├── SELECTION
│   └── eligible and primary relation selection
│
├── CONTINUITY
│   └── edge and network continuity
│
├── DYNAMICS
│   └── degradation, persistence and recovery
│
├── MULTISCALE
│   └── cross-window confirmation
│
├── ADMISSIBILITY
│   └── interpretation boundary
│
├── VALIDATION
│   └── comparison, placebo and ablation
│
├── LIMEN
│   └── communication and action boundary
│
└── REPORT
    └── evidence, uncertainty and falsifications

# 29. Software interface contracts

Every module must receive declared inputs and produce declared
outputs.

Modules must not read hidden global state.

Each output must contain:

- timestamp;
- window;
- value;
- estimability status;
- uncertainty where available;
- provenance reference.

A downstream module must not convert `non_estimable` into
`normal`.

---

# 30. Initial seismic implementation

The first implementation is:

- Experiment: `CGIE3_ID_01`
- Domain: Campi Flegrei
- Frozen source: `CF_RETRO_01`
- Baseline:
  `2025-01-01T00:00:00Z`
  to
  `2025-12-31T23:59:59Z`
- Evaluation:
  `2026-01-01T00:00:00Z`
  to
  `2026-07-31T17:46:42Z`
- Primary window: `7d`
- Secondary windows:
  `1d`, `3d`, `30d`
- Required multiscale confirmations: `2`

Frozen feature candidates:

- `event_count`;
- `maximum_magnitude`;
- `log10_cumulative_energy_joule`;
- `median_depth_km`;
- `depth_mad_km`;
- `spatial_dispersion_km`;
- `median_interevent_time_hours`;
- `interevent_time_mad_hours`;
- `temporal_burstiness`.

`temporal_burstiness` is unavailable or excluded in the `1d`
window according to the frozen CF_RETRO_01 configuration.

---

# 31. Development separation

CGIE-3 must not modify:

- CGIE-2 source code;
- CGIE-2 configurations;
- frozen CF_RETRO_01 inputs;
- frozen CGIE-2 outputs;
- frozen negative-control results.

CGIE-3 must use separate:

- source files;
- configurations;
- workflows;
- outputs;
- manifests;
- reports.

---

# 32. Framework claim boundary

Version 1.0 defines a transferable analytical architecture.

It does not establish that:

- one universal identity metric already exists;
- the same relations define every domain;
- seismic transitions are predictable;
- relational degradation is causal;
- CGIE-3 outperforms CGIE-2;
- the Congruity Framework is universally valid.

These are experimental questions.

---

# 33. Advancement rule

The framework may advance beyond v1.0 only when a proposed change
is supported by:

- an explicit unresolved limitation;
- evidence from at least one frozen experiment;
- compatibility analysis across completed domains;
- updated falsification criteria;
- a versioned migration record.

Changes made solely to improve one observed outcome are
prohibited.

---

# 34. Foundational statement

The Congruity Framework evaluates whether a system preserves the
relational organization required for continuity of its declared
functional identity.

It treats failure, equivalence, non-identifiability and lack of
discrimination as valid scientific outcomes.

Its purpose is not to force prediction from data, but to discover
whether relational continuity contains reproducible information
about systemic transition.

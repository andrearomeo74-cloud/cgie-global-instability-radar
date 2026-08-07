# CGIE3-ID-04 — Relational Identity Continuity Audit

## Protocol version

CGIE3_ID_04_v1.0

## Configuration status

FROZEN

## Framework

Congruity Framework — CGIE-3

## Experiment purpose

CGIE3-ID-04 tests whether the observed system preserves continuity
of its relational organization through time even when CGIE3-ID-03
does not identify reproducible static relational families.

The experiment is explicitly downstream of CGIE3-ID-03.

CGIE3-ID-03 is treated as frozen evidence.

Its scientific outcome is not modified, reinterpreted or optimized
retroactively by ID-04.

## Scientific question

Does relational organization preserve temporal continuity beyond
frozen null expectations even when individual relations or static
relation families are not reproducibly preserved?

The experiment therefore distinguishes:

- persistence of individual relations;
- persistence of relation families;
- continuity of relational organization.

The third property is the primary object of ID-04.

## Scientific hypothesis

### Null hypothesis H0

Observed temporal continuity of relational organization is compatible
with continuity produced by frozen null and surrogate procedures.

Under H0, the experiment does not establish relational identity
continuity.

### Alternative hypothesis H1

Observed relational organization preserves temporal continuity beyond
the frozen null expectation on at least the number of temporal scales
required by the preregistered decision rule.

Support for H1 establishes only evidence for relational continuity.

It does not establish:

- a Minimum Identity Core;
- indispensable relations;
- causal organization;
- predictive capability;
- earthquake prediction;
- universal transferability.

## Input boundary

ID-04 consumes only previously frozen inputs.

Required inputs include:

- the operational identity declaration;
- the frozen feature table;
- the complete ID-02 relation classification;
- ID-03 relation states;
- ID-03 family membership;
- ID-03 relation families;
- ID-03 official summary.

The expected ID-02 relation population remains:

- 136 total relations;
- 45 eligible;
- 29 candidate;
- 62 rejected reference relations.

The primary ID-04 relation population contains the 74 ID-02
eligible or candidate relations preserved through ID-03.

Rejected ID-02 relations may be retained only as reference evidence
and are not included in the primary relational continuity score.

ID-02 classifications must not be modified.

ID-03 states must not be modified.

## Required upstream scientific state

ID-04 is preregistered against the observed frozen ID-03 outcome:

`no_reproducible_families`

This is not treated as a failure to be repaired.

ID-04 tests a distinct hypothesis:

a system may fail to preserve a static reproducible relation family
while preserving continuity of relational organization.

## Relational snapshot

For each temporal position and each frozen temporal scale, the
experiment constructs a relational snapshot:

G_t = (V, E_t, W_t, S_t)

where:

- V is the fixed declared component set;
- E_t is the set of estimable frozen primary relations;
- W_t contains estimated relation strengths;
- S_t contains relation signs where estimable.

The node set is fixed by the identity declaration.

The relation universe is fixed by the frozen primary relation
population.

Relation strength is re-estimated inside each temporal snapshot.

No new relations may be selected after inspection of ID-04 results.

## Estimability

Non-estimability remains an explicit scientific state.

A non-estimable relation must not be silently converted to:

- zero strength;
- absent evidence;
- negative evidence;
- normality.

The experiment distinguishes unavailable information from observed
loss of continuity.

If a transition contains fewer than the frozen minimum number of
estimable continuity components, its Relational Continuity Score is
non-estimable.

If a temporal scale contains fewer than the frozen minimum number of
estimable transitions, that scale is classified as non-estimable.

## Temporal ordering

Snapshots are ordered chronologically within each temporal scale.

The primary analysis compares only consecutive snapshots:

G_t versus G_(t+1)

Cross-scale snapshots are never mixed inside one primary transition.

Each temporal scale is evaluated independently before any cross-scale
scientific conclusion is assigned.

## Continuity component 1 — Edge Continuity

Edge Continuity is denoted EC.

For two consecutive snapshots:

EC_t =
|E_t ∩ E_(t+1)|
/
|E_t ∪ E_(t+1)|

using Jaccard similarity.

The calculation uses only relations whose estimability status permits
the edge comparison under the frozen implementation contract.

EC lies inside [0, 1].

A value of 1 indicates complete preservation of the compared edge set.

A value of 0 indicates no shared compared edges.

An undefined comparison remains non-estimable.

## Continuity component 2 — Weight Continuity

Weight Continuity is denoted WC.

WC evaluates whether common relations preserve their relative strength
ordering between consecutive snapshots.

The primary estimator is Spearman rank correlation.

Only edges estimable in both snapshots enter the comparison.

At least three common estimable edges are required.

If fewer than three are available, WC is non-estimable.

The experiment does not require absolute relation strengths to remain
constant.

WC therefore evaluates continuity of relational ordering rather than
identity of numerical values.

## Continuity component 3 — Sign Continuity

Sign Continuity is denoted SC.

For common estimable signed relations:

SC_t =
N(sign_t = sign_(t+1))
/
N(common signed relations)

At least one common signed relation is required.

If none exists, SC is non-estimable.

An observed sign reversal counts as loss of sign continuity.

No sign is inferred for a relation whose strength is non-estimable.

## Continuity component 4 — Topological Continuity

Topological Continuity is denoted TC.

TC tests whether declared components preserve comparable relational
roles even when individual edges change.

For every snapshot, weighted node degree/strength is calculated from
the estimable primary relation graph.

Node-role vectors from consecutive snapshots are compared using
Spearman rank correlation.

At least three common estimable nodes are required.

TC therefore asks whether the relative structural role of system
components persists through time.

It does not require identical edge membership.

## Relational Continuity Score

The primary composite measure is the Relational Continuity Score:

RCS

The four preregistered components are:

- EC;
- WC;
- SC;
- TC.

Each carries frozen nominal weight:

0.25

No weight optimization is permitted.

If every component is estimable:

RCS_t =
0.25 EC_t
+ 0.25 WC_t
+ 0.25 SC_t
+ 0.25 TC_t

If one or more components are non-estimable, available component
weights are renormalized.

At least two continuity components must remain estimable.

Otherwise RCS is non-estimable for that transition.

Missing values are never replaced by zero.

## Multiscale analysis

Each declared temporal scale is evaluated separately.

The primary analysis does not average temporal scales before testing.

For each scale, the experiment estimates:

- number of snapshots;
- number of transitions;
- number of estimable transitions;
- mean observed RCS;
- median observed RCS;
- dispersion of RCS;
- empirical null distribution;
- corrected significance;
- primary effect size.

Only after scale-specific testing may the experiment determine whether
evidence generalizes across scales.

## Primary effect

For each estimable temporal scale:

Delta_RCS =
observed mean RCS
-
median null mean RCS

The primary effect must be positive for a scale to support relational
continuity.

A statistically significant result with a non-positive primary effect
does not count as supporting continuity.

## Frozen null controls

Three primary null controls are preregistered.

Each primary null uses 1000 repetitions.

Randomization uses the frozen seed specified by the ID-04
configuration.

### NULL-1 — Temporal snapshot permutation

This null:

- preserves complete observed snapshot contents;
- preserves the marginal distribution of snapshot properties;
- destroys the observed temporal succession.

The null tests whether observed continuity depends on actual temporal
ordering.

### NULL-2 — Relation-label permutation

This null:

- preserves relation weight distributions;
- preserves snapshot density;
- permutes relation identity.

It destroys the correspondence between an observed edge and its
declared source/target identity.

The null tests whether continuity follows from generic weight
distributions rather than preservation of relational identity.

### NULL-3 — Constrained weight/sign surrogate

This null preserves, as far as specified by the frozen implementation:

- edge density;
- marginal weight distribution;
- marginal sign distribution.

It does not preserve original relation identity.

The null tests whether the observed continuity can be reproduced by a
network with comparable low-order distributions but without the
specific observed relational organization.

## Empirical significance

Empirical significance uses an upper-tail test.

Finite-sample empirical p-values use the plus-one correction.

Multiple testing across temporal scales is corrected using:

Benjamini-Hochberg false discovery rate.

Frozen FDR threshold:

0.05

No uncorrected scale-specific p-value may independently establish the
primary scientific conclusion.

## Robustness analysis

ID-04 includes a leave-one-transition-out robustness audit.

Each estimable transition is removed once and the scale-level
conclusion is recomputed.

The robustness fraction is:

R =
number of leave-one-transition-out analyses retaining the conclusion
/
number of admissible leave-one-transition-out analyses

Frozen minimum robustness requirement:

R >= 0.80

A full continuity conclusion must not depend critically on one
transition.

## Primary scientific outcome classes

Exactly four scientific outcomes are permitted.

### CONTINUITY_SUPPORTED

Assigned only when:

- at least two temporal scales are estimable;
- at least two scales meet corrected FDR significance;
- every supporting scale has positive Delta_RCS;
- required robustness is at least 0.80;
- no prohibited single-transition dependency is detected.

This outcome establishes evidence for relational identity continuity
under the frozen ID-04 operational definition.

It establishes nothing stronger.

### PARTIAL_OR_SCALE_SPECIFIC_EVIDENCE

Assigned when:

- at least one scale supports continuity;
- the full cross-scale requirement is not met.

Possible reasons include:

- only one significant temporal scale;
- inconsistent scale direction;
- insufficient robustness on one or more otherwise supportive scales.

This outcome must not be reported as full continuity support.

### CONTINUITY_NOT_SUPPORTED

Assigned when:

- at least one temporal scale is scientifically estimable;
- no temporal scale satisfies the corrected support rule.

This is a valid negative result.

Thresholds, temporal scales and weights may not be changed
retroactively to convert this outcome into support.

### NON_IDENTIFIABLE

Assigned when:

- no temporal scale is sufficiently estimable to test the primary
  hypothesis.

This outcome means the available evidence cannot identify the tested
property.

It is not equivalent to evidence that continuity is absent.

## Explicit falsification rules

ID-04 falsifies the preregistered continuity hypothesis for the tested
dataset when sufficient estimability exists but observed relational
continuity fails to exceed the null expectation under the frozen
decision rule.

The following are explicitly recorded as negative or limiting results:

- no significant temporal scale;
- non-positive Delta_RCS;
- continuity compatible with temporal permutation;
- continuity compatible with relation-label permutation;
- continuity compatible with constrained surrogates;
- failure of leave-one-transition-out robustness;
- support isolated to one temporal scale;
- excessive non-estimability.

All such results remain in the official report.

## Forbidden analyses

The following analyses are outside CGIE3-ID-04 and must not be
performed as part of the primary experiment:

- earthquake-event alignment;
- earthquake lead-time optimization;
- target-event threshold optimization;
- retrospective temporal-scale selection;
- post-hoc weight optimization;
- precursor identification;
- epicenter estimation;
- magnitude estimation;
- alarm generation;
- causal inference;
- indispensable-relation claims;
- Minimum Identity Core claims;
- predictive claims.

## Event blindness

ID-04 is deliberately event-blind.

Earthquake occurrence is not used:

- to select snapshots;
- to select temporal scales;
- to tune thresholds;
- to optimize weights;
- to define nulls;
- to classify continuity.

Any future event-alignment study requires a separate frozen experiment.

## Relation to CGIE3-ID-03

CGIE3-ID-03 tested whether primary relations could be compressed into
reproducible static relational families.

Its frozen result is preserved.

CGIE3-ID-04 instead tests whether relational organization exhibits
temporal continuity.

Therefore:

failure of static family reproducibility

does not logically imply

failure of relational continuity.

Conversely:

support for relational continuity

does not retroactively make ID-03 families reproducible.

The experiments answer different questions.

## Scientific claim boundary

Even a positive CGIE3-ID-04 result does not establish:

- that the operational identity declaration is uniquely correct;
- that any individual relation is essential;
- that any relation is causal;
- that an identified structure constitutes a Minimum Identity Core;
- that continuity predicts system failure;
- that continuity predicts earthquakes;
- that the result transfers universally to other systems.

Those hypotheses require independent experiments.

## Valid negative result principle

A negative result is scientifically admissible and final under this
protocol.

If ID-04 concludes:

CONTINUITY_NOT_SUPPORTED

the experiment is not reopened merely because the result conflicts
with the theoretical expectation.

If ID-04 concludes:

NON_IDENTIFIABLE

additional data may motivate a future experiment, but the existing
experiment remains non-identifiable.

## Official outputs

The experiment produces the following official outputs:

- CGIE3_ID_04_snapshot_relations.csv
- CGIE3_ID_04_transition_continuity.csv
- CGIE3_ID_04_multiscale_continuity.csv
- CGIE3_ID_04_null_controls.csv
- CGIE3_ID_04_null_summaries.csv
- CGIE3_ID_04_leave_one_transition_out.csv
- CGIE3_ID_04_estimability.csv
- CGIE3_ID_04_summary.json
- CGIE3_ID_04_report.md
- CGIE3_ID_04_manifest.json
- CGIE3_ID_04_workflow_provenance.json
- CGIE3_ID_04_execution_status.json

## Advancement boundary

CGIE3-ID-04 does not automatically authorize a subsequent event
prediction experiment.

A positive ID-04 result would justify only the next scientific
question:

whether observed degradation of relational continuity can itself be
defined, measured and independently validated without target-event
information.

That question belongs to a separate frozen experiment.

## Final preregistered interpretation

CGIE3-ID-04 asks whether identity can be operationally observed as
continuity of relational organization rather than as persistence of a
fixed relation set.

The experiment is considered scientifically successful if it produces
a valid, reproducible and auditable conclusion under the frozen rules,
regardless of whether that conclusion supports or rejects relational
continuity.

# CGIE3-ID-01 — Identity Continuity Baseline

## Status

FROZEN PROTOCOL — VERSION 1.0

This protocol must not be modified after inspection of the
CGIE3-ID-01 experimental results.

---

## 1. Experiment identity

- Experiment ID: `CGIE3_ID_01`
- Engine family: `CGIE-3`
- Engine name: `Identity Congruity Radar`
- Protocol version: `1.0`
- Experimental role: baseline identity-continuity test
- Comparison engine: `CGIE-2`
- Primary domain: seismic-system monitoring
- Initial frozen dataset: `CF_RETRO_01`

---

## 2. Scientific question

Does the continuity of a system's relational identity provide
information that is not already contained in:

- individual seismic features;
- minimum feature continuity;
- aggregate structural continuity;
- correlation-matrix change;
- multiscale alert confirmation?

---

## 3. Primary hypothesis

A seismic system may preserve ordinary marginal feature values
while losing continuity in the relationships that maintain its
dynamic identity.

Therefore, degradation of relational identity may occur before
or independently from a conventional anomaly state.

---

## 4. Null hypothesis

The proposed identity-continuity indicators do not add
discriminating information beyond the frozen CGIE-2 indicators.

Possible null outcomes include:

- equivalence with CGIE-2;
- lower temporal specificity;
- excessive false-positive rate;
- no improvement against placebo controls;
- no stable relational identity;
- no reproducibility across windows.

All such outcomes must be retained and reported.

---

## 5. Interpretation boundary

CGIE-3 is not a deterministic earthquake-prediction system.

CGIE-3 evaluates:

- relational continuity;
- structural identity degradation;
- multiscale coherence;
- transition dynamics;
- residual failure of relational closure.

CGIE-3 must not claim:

- the exact time of an earthquake;
- the exact location of a future earthquake;
- the magnitude of a future earthquake;
- causal prediction from temporal proximity alone.

---

## 6. Frozen input data

CGIE3-ID-01 must use the same frozen input lineage already used
for the CF_RETRO_01 and CGIE-2 experiments.

Required inputs:

- frozen seismic catalogue;
- frozen seismic-feature table;
- frozen analysis interval;
- frozen temporal windows;
- frozen target-event catalogue;
- frozen CGIE-2 outputs.

No future event may be added to the frozen retrospective dataset
after analysis begins.

---

## 7. Base feature set

The initial identity network must be built only from features
already present in the frozen CGIE-2 feature table.

Candidate feature families include:

- event count;
- maximum magnitude;
- cumulative energy;
- median depth;
- depth dispersion;
- spatial dispersion;
- median inter-event interval;
- inter-event interval dispersion;
- temporal burstiness.

The exact available columns must be validated programmatically.

No new external geophysical variable may be introduced in
CGIE3-ID-01.

---

## 8. Relational representation

For every frozen temporal window, CGIE-3 must construct a
relational representation among the available features.

The initial representation must include:

1. signed pairwise association;
2. absolute relational strength;
3. relational persistence through time;
4. relational direction consistency;
5. missingness and estimability status.

The primary relation estimator must be frozen before outcome
inspection.

The initial estimator is:

- Spearman rank association.

Pearson correlation may be retained only as a secondary
sensitivity analysis.

---

## 9. Reference identity

The reference identity must be learned exclusively from a frozen
baseline interval that precedes the evaluation interval.

The reference identity consists of:

- eligible relational edges;
- baseline edge sign;
- baseline edge-strength distribution;
- edge persistence;
- network density;
- connected components;
- node participation;
- relational redundancy.

The target event must not be used to select edges.

---

## 10. Edge eligibility

An edge is eligible only when all of the following are satisfied:

- sufficient paired observations;
- estimable association;
- minimum baseline persistence;
- stable sign or explicitly recorded sign ambiguity;
- no use of future labels;
- no retrospective outcome-based selection.

Exact numerical thresholds must be stored in the frozen
configuration file.

---

## 11. Relational hierarchy

CGIE3-ID-01 must distinguish:

- primary identity edges;
- secondary identity edges;
- unstable or non-identifiable edges.

Primary edges are those with the strongest baseline persistence
and participation in network connectivity.

Secondary edges are retained for sensitivity analysis.

No edge may be called indispensable on the basis of this single
experiment.

The term `indispensable` requires later ablation evidence across
independent datasets.

---

## 12. Primary indicators

CGIE3-ID-01 must calculate at least the following indicators.

### 12.1 Identity Edge Continuity — IEC

Continuity of each eligible relation relative to its frozen
baseline identity.

IEC must account for:

- strength deviation;
- sign preservation;
- temporal persistence;
- estimability.

### 12.2 Identity Network Continuity — INC

Aggregate continuity of the primary relational identity network.

INC must not be a simple unweighted mean unless explicitly
validated as equivalent.

### 12.3 Minimum Functional Admissible Continuity — MFAC-ID

Continuity of the weakest primary identity relation or relational
substructure under the frozen aggregation rule.

MFAC-ID must remain distinguishable from the original CGIE-2
minimum feature-continuity measure.

### 12.4 Identity Acceleration — APS-ID

Rate and acceleration of identity-continuity degradation.

APS-ID must distinguish:

- low continuity with recovery;
- rapid active degradation;
- stable low-estimability states.

### 12.5 Identity Geometry Change — CRM-ID

Change in the geometry of the eligible relational identity
network relative to baseline.

CRM-ID must compare relational structure, not only raw feature
correlation matrices.

### 12.6 Relational Closure Residue — RCR

Residual relational inconsistency after accounting for the
preserved identity network.

RCR is exploratory in CGIE3-ID-01 and must not be used as the
primary success criterion.

---

## 13. Multiscale structure

Indicators must be calculated independently for the frozen
temporal windows already used by CGIE-2.

No window may be added after outcome inspection.

Cross-window confirmation must distinguish:

- isolated degradation;
- persistent degradation;
- nested degradation;
- contradictory windows;
- multiscale recovery.

---

## 14. Primary comparison

CGIE3-ID-01 must be compared directly with frozen CGIE-2 outputs
on identical timestamps and events.

The primary comparison must evaluate whether CGIE-3 adds:

- earlier transition detection;
- fewer unassociated alert episodes;
- better event-cluster discrimination;
- greater temporal specificity;
- greater robustness under placebo shifts;
- information conditional on CGIE-2 state.

---

## 15. Primary outcome

The primary outcome is not raw lead time alone.

The primary outcome is:

> improvement in temporal discrimination relative to CGIE-2
> under frozen event-time placebo controls.

Improvement requires both:

1. observed association better than the CGIE-2 observed result;
2. observed result superior to its own frozen placebo
   distribution.

A longer lead time without improved specificity is not success.

---

## 16. Secondary outcomes

Secondary outcomes include:

- number of non-normal episodes;
- fraction of evaluation time flagged;
- event-cluster detection fraction;
- unassociated episode fraction;
- median lead time;
- maximum lead time;
- duration of degradation;
- recovery time;
- agreement with CGIE-2;
- disagreement with CGIE-2;
- identity loss while CGIE-2 remains normal;
- CGIE-2 alert while identity remains continuous.

---

## 17. Negative controls

CGIE3-ID-01 must include:

- circular event-time shifts;
- random temporal shifts;
- edge-label permutation;
- node-label permutation where mathematically admissible;
- baseline-interval sensitivity;
- secondary-edge exclusion;
- leave-one-primary-edge-out analysis.

Controls must not alter the frozen target event catalogue.

---

## 18. Falsification criteria

The identity hypothesis is not supported in this experiment if
one or more of the following occurs:

- no stable reference identity can be estimated;
- primary edges are not reproducible;
- CGIE-3 is equivalent to CGIE-2;
- CGIE-3 performs worse than CGIE-2;
- observed associations are not superior to placebo;
- improvements depend on one edge only;
- improvements disappear under minor baseline perturbations;
- alert burden rises without improved discrimination;
- identity indicators are dominated by missingness;
- target-event knowledge is required to obtain the result.

All falsifications must be explicitly reported.

---

## 19. Equivalence criteria

A result must be reported as equivalent when CGIE-3 indicators
are monotonic transformations or operational duplicates of
existing CGIE-2 indicators without measurable additional
information.

Complexity alone does not constitute novelty.

---

## 20. Safety and communication boundary

All outputs must use the following hierarchy:

- `normal`
- `identity_watch`
- `identity_degradation`
- `identity_transition`

No output may be communicated publicly as:

- earthquake prediction;
- imminent earthquake warning;
- evacuation advice;
- official civil-protection guidance.

Official seismic and civil-protection authorities remain the
authoritative sources for public safety decisions.

---

## 21. Reproducibility requirements

Every run must record:

- Git commit;
- configuration hash;
- input hashes;
- baseline interval;
- evaluation interval;
- feature columns;
- eligible edges;
- primary edges;
- thresholds;
- random seed;
- software versions;
- generated-output hashes.

---

## 22. Required outputs

The experiment must generate:

- `CGIE3_ID_01_reference_edges.csv`
- `CGIE3_ID_01_edge_continuity.csv`
- `CGIE3_ID_01_identity_metrics.csv`
- `CGIE3_ID_01_identity_alerts.csv`
- `CGIE3_ID_01_event_associations.csv`
- `CGIE3_ID_01_placebos.csv`
- `CGIE3_ID_01_ablations.csv`
- `CGIE3_ID_01_summary.json`
- `CGIE3_ID_01_report.md`
- `CGIE3_ID_01_manifest.json`
- `CGIE3_ID_01_workflow_provenance.json`

---

## 23. Decision rule

CGIE3-ID-01 may be classified as:

- `supported_preliminary`
- `equivalent`
- `non_discriminating`
- `unstable_identity`
- `inconclusive_low_power`
- `falsified_in_current_configuration`

No stronger classification is permitted from this experiment.

---

## 24. Development boundary

CGIE-2 files, configurations and frozen outputs must not be
modified by CGIE3-ID-01.

CGIE-3 must use separate:

- source files;
- configuration files;
- workflow;
- outputs;
- protocol;
- manifest.

This separation is mandatory for valid comparison.

---

## 25. Frozen statement

CGIE3-ID-01 tests whether relational identity continuity adds
measurable and reproducible information beyond the frozen CGIE-2
engine.

It does not assume that relational identity exists, that the
selected relations are indispensable, or that degradation
predicts a specific earthquake.

Failure to outperform the frozen controls is an admissible and
scientifically valuable result.

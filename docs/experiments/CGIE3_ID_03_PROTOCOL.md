# CGIE3-ID-03

## Relational Dependency and Family Audit Protocol

Version: 1.0 Draft

Status: PRE-FREEZE

---

# 1. Purpose

CGIE3-ID-03 evaluates whether the relations classified as
`eligible` or `candidate` by CGIE3-ID-02 represent distinct
structural information or multiple projections of the same
underlying feature-generating process.

The experiment aims to:

- identify definitional dependencies among features;
- evaluate multiscale consistency;
- test sensitivity to overlapping rolling windows;
- identify relational redundancy;
- organize relations into evidence-supported families;
- select family-representative candidates for later experiments.

CGIE3-ID-03 does not establish:

- primary identity relations;
- indispensable relations;
- causal mechanisms;
- earthquake precursors;
- predictive capability.

---

# 2. Scientific question

Do the relations retained by CGIE3-ID-02 contain distinct and
reproducible structural information, or are they primarily produced
by shared definitions, overlapping observations and redundant
feature construction?

---

# 3. Input lineage

CGIE3-ID-03 uses only frozen CGIE3-ID-02 outputs and the frozen
CF_RETRO_01 feature lineage.

Required inputs:

```text
outputs/CGIE3_ID_02_candidate_relations.csv
outputs/CGIE3_ID_02_block_relations.csv
outputs/CGIE3_ID_02_bootstrap_relations.csv
outputs/CGIE3_ID_02_relation_classification.csv
outputs/CGIE3_ID_02_equivalence_flags.csv
outputs/CGIE3_ID_02_summary.json
outputs/CF_RETRO_01_features.csv
engines/cgie3/config/campi_flegrei_identity.yaml
engines/cgie3/config/cgie3_id_02_relation_discovery.yaml
```

The ID-02 population is preserved without retrospective
reclassification:

- 45 `eligible`;
- 29 `candidate`;
- 62 `rejected`;
- 0 `non_estimable`.

ID-03 may add new audit labels but must not rewrite the original
ID-02 status.

---

# 4. Analysis population

## Primary audit population

All relations classified as:

- `eligible`;
- `candidate`.

Expected count:

```text
74 relations
```

## Negative-reference population

Relations classified as:

- `rejected`.

Rejected relations remain available for comparison and calibration.

They must not be removed from official outputs.

---

# 5. Audit structure

CGIE3-ID-03 contains five mandatory audits:

1. Feature Dependency Audit;
2. Multiscale Relation Audit;
3. Rolling-Window Overlap Audit;
4. Redundancy and Null Audit;
5. Relational Family Audit.

No family-representative candidate may be selected before all five
audits are completed.

---

# 6. Feature Dependency Audit

## 6.1 Objective

Determine whether each relation joins features that are
mathematically, procedurally or observationally dependent before
any statistical association is estimated.

## 6.2 Feature provenance classes

Each feature receives one provenance class:

- `primitive_observation`;
- `aggregate_observation`;
- `distribution_summary`;
- `derived_physical_quantity`;
- `derived_temporal_quantity`;
- `derived_spatial_quantity`;
- `unknown_provenance`.

## 6.3 Pairwise dependency classes

Every relation receives exactly one dependency status:

- `independent_source`;
- `shared_event_source`;
- `shared_temporal_source`;
- `shared_spatial_source`;
- `partially_derived`;
- `directly_derived`;
- `unknown_dependency`.

## 6.4 Initial feature lineage declarations

The following declarations are candidate protocol inputs and must
be checked against the actual CF_RETRO_01 feature-generation code
before freezing.

### `event_count`

Derived from the number of retained seismic events inside the
declared window.

### `maximum_magnitude`

Derived from the maximum magnitude among retained events inside the
declared window.

### `log10_cumulative_energy_joule`

Derived from event magnitudes through the frozen seismic-energy
conversion and aggregation procedure.

### `median_depth_km`

Derived from the event-depth distribution inside the window.

### `depth_mad_km`

Derived from dispersion in the event-depth distribution.

### `spatial_dispersion_km`

Derived from the spatial distribution of retained event locations.

### `median_interevent_time_hours`

Derived from ordered event timestamps.

### `interevent_time_mad_hours`

Derived from dispersion in event interarrival times.

### `temporal_burstiness`

Derived from the distribution of event interarrival times.

These declarations do not establish redundancy automatically.

They identify potential sources of non-independence that must
remain visible in interpretation.

---

# 7. Multiscale Relation Audit

## 7.1 Objective

Determine whether the same component pair preserves a coherent
relation across:

- `1d`;
- `3d`;
- `7d`;
- `30d`.

## 7.2 Pair alignment

Relations are aligned by their unordered component pair,
independently of window.

For every aligned pair, record:

- ID-02 status in each scale;
- signed Spearman strength;
- absolute strength;
- persistence;
- estimable block fraction;
- sign-preservation fraction;
- bootstrap confidence interval;
- leave-one-block-out result;
- missingness-stress result.

## 7.3 Multiscale support classes

Each aligned relation receives one class:

- `single_scale`;
- `multi_scale_2`;
- `multi_scale_3`;
- `multi_scale_4`;
- `scale_inconsistent`;
- `insufficient_scale_support`.

## 7.4 Sign consistency

A relation is sign-consistent when every estimable supported scale
has the same sign.

A relation is `scale_inconsistent` when:

- supported scales show opposite signs; or
- one or more strong supported scales contradict the dominant sign.

Sign inconsistency must not be hidden by absolute-strength
aggregation.

## 7.5 Scale-specific relations

A single-scale relation may remain scientifically relevant when:

- the scale specificity is declared before outcome interpretation;
- the relation passes overlap and null controls;
- the observed specificity is not caused by insufficient sample
  support at the other scales.

Single-scale status does not automatically imply rejection.

---

# 8. Rolling-Window Overlap Audit

## 8.1 Objective

Determine whether relations identified from strongly overlapping
rolling windows remain observable when temporal dependence between
adjacent feature rows is reduced.

## 8.2 Required sampling schemes

For each temporal scale, evaluate:

### `rolling_full`

Use every frozen feature-table endpoint.

### `half_window_stride`

Use a stride equal to half the declared temporal window.

### `non_overlapping`

Use a stride equal to the full declared temporal window.

## 8.3 Example strides

Where the feature table has hourly endpoints:

```text
1d:
  half-window stride = 12 hours
  non-overlapping stride = 24 hours

3d:
  half-window stride = 36 hours
  non-overlapping stride = 72 hours

7d:
  half-window stride = 84 hours
  non-overlapping stride = 168 hours

30d:
  half-window stride = 360 hours
  non-overlapping stride = 720 hours
```

The implementation must derive actual row strides from timestamp
spacing rather than assuming that every input table is complete.

## 8.4 Required comparisons

For every relation and sampling scheme, record:

- paired observation count;
- signed strength;
- sign;
- uncertainty;
- rank among relations in the same scale;
- deviation from the rolling-full estimate;
- estimability.

## 8.5 Overlap-sensitivity classes

Each relation receives one class:

- `overlap_robust`;
- `moderately_overlap_sensitive`;
- `strongly_overlap_sensitive`;
- `non_estimable_non_overlapping`;
- `inconclusive_overlap_audit`.

A relation cannot become a family representative solely from
rolling-full evidence when it is strongly overlap-sensitive.

---

# 9. Effective temporal support

Nominal row count must not be treated as independent sample size.

For each feature and relation, ID-03 must report at least:

- nominal paired observation count;
- lag-1 autocorrelation;
- declared stride;
- non-overlapping observation count;
- an explicit effective-support diagnostic.

The experiment must not claim a universally valid effective sample
size formula unless separately specified and validated.

Temporary technical names must be used, such as:

- `nominal_sample_count`;
- `nonoverlap_sample_count`;
- `lag1_autocorrelation`;
- `effective_support_diagnostic`.

---

# 10. Redundancy and Null Audit

## 10.1 Objective

Determine whether apparent relational structure exceeds results
obtainable from simplified or randomized alternatives.

## 10.2 Required null controls

Where mathematically admissible:

- circular temporal shifts of one component;
- independent block permutations;
- component-label permutation;
- within-window value permutation;
- phase-randomized or autocorrelation-preserving surrogate;
- strength-matched random-edge comparison.

## 10.3 Required redundancy checks

Evaluate whether relations provide distinct information after
conditioning on:

- `event_count`;
- `maximum_magnitude`;
- total or cumulative energy;
- their declared feature family;
- a simpler relation joining the same family.

Methods may include:

- partial rank association;
- residual association;
- conditional information diagnostics;
- deterministic transformation checks;
- relation-decision equivalence.

No single conditional method is considered universally
authoritative.

## 10.4 Null-audit outcomes

Each relation receives one outcome:

- `exceeds_null`;
- `equivalent_to_null`;
- `partially_exceeds_null`;
- `null_test_inconclusive`;
- `null_test_not_admissible`.

---

# 11. Relational Family Audit

## 11.1 Objective

Organize relations into evidence-supported groups without assuming
that each eligible edge represents an independent aspect of system
identity.

## 11.2 Graph representation

The family audit uses a graph in which:

- nodes represent declared features;
- edges represent ID-02 eligible or candidate relations;
- edge attributes contain all ID-02 and ID-03 audit evidence.

## 11.3 Permitted family evidence

Families may be supported by:

- shared feature provenance;
- graph community structure;
- multiscale co-occurrence;
- common sign and strength dynamics;
- similar null-test behavior;
- similar overlap sensitivity;
- conditional redundancy.

## 11.4 Prohibited procedure

The number and names of final families must not be chosen solely to
match the five preliminary conceptual groups proposed after ID-02.

The preliminary labels:

- energy;
- frequency and rhythm;
- temporal organization;
- depth structure;
- spatial geometry;

may be used as interpretive candidates only after the data-driven
family structure is obtained.

## 11.5 Family outputs

Every family must report:

- family identifier;
- member components;
- member relations;
- scales represented;
- eligible-relation count;
- candidate-relation count;
- dependency composition;
- overlap robustness;
- null support;
- internal redundancy;
- unresolved residue.

---

# 12. Representative-candidate selection

A relation may become a
`family_representative_candidate` only when all mandatory
conditions are satisfied.

## 12.1 Mandatory conditions

The relation must:

1. be `eligible` or `candidate` in ID-02;
2. belong to an identified relational family;
3. not be `directly_derived`;
4. not be `strongly_overlap_sensitive`;
5. preserve sign under an admissible reduced-overlap scheme;
6. have support in at least two temporal scales, unless a frozen
   scale-specific justification applies;
7. exceed at least one admissible null control;
8. retain information after conditioning on the dominant activity
   feature or be explicitly classified as unresolved;
9. not be fully equivalent to a simpler relation in the same
   family.

## 12.2 Tie-breaking rule

When multiple relations satisfy the conditions inside one family,
rank them deterministically by:

1. number of supported scales;
2. overlap robustness;
3. null-control margin;
4. conditional residual information;
5. ID-02 persistence;
6. bootstrap interval width;
7. lexicographic relation identifier.

No human interpretability criterion may be used as an unrecorded
tie-break.

---

# 13. ID-03 relation states

Every audited relation receives exactly one final ID-03 state:

- `family_representative_candidate`;
- `supporting_relation`;
- `definitionally_constrained`;
- `redundant_relation`;
- `overlap_sensitive`;
- `scale_inconsistent`;
- `insufficient_evidence`.

The ID-03 state is added beside the frozen ID-02 status.

It does not replace or modify the ID-02 status.

---

# 14. Classification precedence

The frozen precedence is:

1. `scale_inconsistent`;
2. `overlap_sensitive`;
3. `definitionally_constrained`;
4. `redundant_relation`;
5. `family_representative_candidate`;
6. `supporting_relation`;
7. `insufficient_evidence`.

A relation satisfying multiple conditions receives the
highest-precedence applicable state, while all secondary flags
remain recorded.

---

# 15. Scientific success criteria

CGIE3-ID-03 succeeds technically when:

- every ID-02 eligible and candidate relation is audited;
- feature dependencies are explicitly registered;
- multiscale pairs are aligned deterministically;
- reduced-overlap analyses are completed where estimable;
- null and redundancy outcomes are preserved;
- every audited relation receives exactly one ID-03 state;
- complete provenance is produced.

A scientifically positive result requires:

- at least one reproducible relational family;
- and at least one family-representative candidate satisfying all
  frozen mandatory conditions.

---

# 16. Valid negative outcomes

The following are valid scientific outcomes:

- `no_reproducible_families`;
- `all_relations_definitionally_constrained`;
- `all_relations_overlap_sensitive`;
- `no_relation_exceeds_null`;
- `families_identified_without_representatives`;
- `insufficient_nonoverlap_power`;
- `multiscale_structure_inconsistent`.

These outcomes must not trigger retrospective threshold changes.

---

# 17. Falsification conditions

The structural interpretation derived from ID-02 is weakened when:

- eligible relations collapse under non-overlapping sampling;
- relation rankings are unstable under minor admissible changes;
- most relations are explained by direct feature derivation;
- relation families do not exceed null controls;
- no conditional information remains beyond activity magnitude;
- apparent families depend entirely on one arbitrary clustering
  choice;
- representative selection changes under equivalent deterministic
  implementations.

---

# 18. Claim boundary

Permitted claim:

> CGIE3-ID-03 identified one or more relational families and
> representative candidates that survived the frozen dependency,
> multiscale, overlap, redundancy and null audits.

Prohibited claims:

- the representatives are primary identity relations;
- the representatives are indispensable;
- the families are causal physical mechanisms;
- the relations predict a specific earthquake;
- the identified structure is universally transferable;
- the Minimum Identity Core has been discovered.

---

# 19. Required outputs

```text
outputs/CGIE3_ID_03_feature_dependencies.csv
outputs/CGIE3_ID_03_multiscale_relations.csv
outputs/CGIE3_ID_03_overlap_sensitivity.csv
outputs/CGIE3_ID_03_null_controls.csv
outputs/CGIE3_ID_03_conditional_redundancy.csv
outputs/CGIE3_ID_03_relation_families.csv
outputs/CGIE3_ID_03_family_membership.csv
outputs/CGIE3_ID_03_representative_candidates.csv
outputs/CGIE3_ID_03_summary.json
outputs/CGIE3_ID_03_report.md
outputs/CGIE3_ID_03_manifest.json
outputs/CGIE3_ID_03_workflow_provenance.json
outputs/CGIE3_ID_03_execution_status.json
```

---

# 20. Development separation

CGIE3-ID-03 must not modify:

- CGIE3-ID-01 outputs;
- CGIE3-ID-02 outputs;
- frozen CF_RETRO_01 inputs;
- CGIE-2 source code;
- ID-02 classifications;
- ID-02 thresholds.

ID-03 must use separate:

- configuration;
- source modules;
- workflow;
- outputs;
- manifest;
- provenance.

---

# 21. Advancement boundary

CGIE3-ID-04 may begin only after ID-03 produces:

- at least one reproducible family;
- at least one representative candidate;
- no unresolved workflow or provenance failure;
- a frozen report of negative and inconclusive outcomes.

Even then, ID-04 may test candidate structural importance but must
not assume indispensability in advance.

---

# 22. Foundational statement

CGIE3-ID-03 tests whether statistically eligible relations can be
compressed into distinct, multiscale and non-trivially derived
relational families.

Its purpose is to reduce apparent relational complexity without
forcing a positive identity structure.

Failure to identify stable families or representatives is a valid
scientific result.

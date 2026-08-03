# CF-POS-01 — Frozen Positive Temporal Association Test

## 1. Experiment identity

- Experiment ID: CF_POS_01
- Status: FROZEN
- Purpose: positive temporal-association validation
- Domain: Campi Flegrei retrospective seismic analysis
- Interpretation boundary: association only; no earthquake-prediction claim

## 2. Primary question

Do the original, unshifted CGIE alert episodes show a stronger temporal
association with significant seismic event clusters than expected under
frozen event-time placebo controls?

## 3. Frozen inputs

CF-POS-01 must use without retrospective modification:

- the frozen CF-RETRO-01 earthquake catalogue;
- the frozen CF-RETRO-01 alert episodes;
- the frozen CF-NEG-01 significant event clusters;
- the same catalogue cutoff used by the preceding experiments;
- the same alert-episode boundaries;
- the same significant-event definition;
- the same primary association window;
- the same episode-association rule.

No alert episode, event cluster, timestamp, threshold, or association
boundary may be removed or changed after inspection of CF-POS-01 results.

## 4. Primary observed statistic

The primary statistic is:

    number of significant event clusters associated with at least one
    original unshifted alert episode

A cluster is counted at most once.

An alert episode is associated with a cluster when the frozen association
rule is satisfied.

## 5. Frozen primary parameters

- Primary event magnitude threshold: 3.0
- Primary maximum lead window: 168 hours
- Placebo repetitions: 1000
- Random seed: 20260803
- Significance level: 0.05
- Placebo method: circular alert-episode shift

The observed alert episodes remain fixed for the primary observed result.

For each placebo repetition, the complete alert-episode structure is shifted
circularly within the frozen analysis interval while preserving:

- episode count;
- episode duration;
- relative spacing between episode boundaries;
- internal alert structure;
- total observation interval.

## 6. Primary empirical test

Let:

- O = observed number of associated significant event clusters;
- P_b = associated-cluster count in placebo repetition b;
- B = 1000.

The one-sided empirical p-value is:

    p = (1 + number of placebo counts greater than or equal to O) / (B + 1)

## 7. Frozen evidence classification

The result is classified as:

### temporally_specific

All conditions must hold:

- empirical p-value < 0.05;
- observed count is greater than the placebo median;
- observed count is greater than the placebo 95th percentile;
- at least two significant event clusters are associated.

### preliminary

All conditions must hold:

- empirical p-value >= 0.05 and < 0.10;
- observed count is greater than the placebo median;
- at least two significant event clusters are associated.

### non_discriminating

Assigned when the requirements for temporally_specific or preliminary are
not met.

### inconclusive_low_power

Assigned when fewer than five significant event clusters exist in the
frozen evaluation set.

## 8. Secondary analyses

Secondary analyses are descriptive only and cannot replace the primary test:

- lead-time distribution;
- association precision;
- alert-episode coverage;
- sensitivity at 24, 72, 168, 336, and 720 hours;
- magnitude thresholds 2.5, 3.0, 3.5, and 4.0;
- episode-level and cluster-level association tables.

No secondary result may be presented as confirmatory.

## 9. Required negative findings

The report must explicitly state:

- whether the observed statistic is compatible with the placebo
  distribution;
- whether significance depends on one event cluster;
- whether significance disappears under nearby frozen sensitivity checks;
- whether alert coverage alone can explain the observed association;
- whether CF-POS-01 contradicts CF-NEG-01 or CF-NEG-02.

## 10. Prohibited practices

The following practices are prohibited:

- changing the alert threshold after observing results;
- changing the magnitude threshold after observing results;
- changing the association window after observing results;
- deleting unfavourable alert episodes;
- deleting unfavourable event clusters;
- selecting only favourable lead times;
- redefining episode boundaries;
- changing the random seed after observing results;
- presenting overlapping detection as earthquake prediction;
- using the target event to construct the input alerts;
- replacing the primary statistic with a more favourable secondary metric.

## 11. Required outputs

The workflow must generate:

- CF_POS_01_observed_associations.csv
- CF_POS_01_placebo_distribution.csv
- CF_POS_01_sensitivity.csv
- CF_POS_01_summary.json
- CF_POS_01_report.md
- CF_POS_01_manifest.json
- CF_POS_01_workflow_provenance.json

## 12. Interpretation boundary

CF-POS-01 can test temporal specificity of retrospective alert-event
association.

It cannot establish:

- earthquake prediction;
- causal precursors;
- prospective forecasting ability;
- operational warning capability;
- generalisation outside the frozen Campi Flegrei dataset.

A positive result must be followed by a genuinely prospective,
pre-registered evaluation.

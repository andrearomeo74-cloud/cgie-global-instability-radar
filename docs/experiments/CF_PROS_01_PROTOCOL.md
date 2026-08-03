# CF-PROS-01 — Frozen Prospective Monitoring Protocol

## 1. Experiment identity

- Experiment ID: CF_PROS_01
- Version: 1.0
- Status: FROZEN
- Domain: Campi Flegrei
- Analysis type: prospective structural-instability monitoring
- Start condition: first successful execution after protocol freeze
- End condition: explicit closeout under a separately committed protocol

## 2. Primary objective

Determine whether the frozen Congruity radar detects prospective
structural transitions using only seismic observations available at the
time of each execution.

The experiment must preserve all outputs, including:

- normal states;
- attention states;
- anomaly states;
- structural states;
- failed executions;
- unavailable-data states;
- subsequent corrections supplied by the data source.

## 3. Interpretation boundary

CF-PROS-01 evaluates prospective structural monitoring.

It does not establish:

- deterministic earthquake prediction;
- prediction of exact event time;
- prediction of exact magnitude;
- prediction of exact location;
- causal seismic precursors;
- operational civil-protection warning capability;
- safety advice for the public.

Any radar alert is a research output and must not replace information
issued by INGV, the Italian Civil Protection Department or other
competent authorities.

## 4. Frozen inputs

Each execution may use only catalogue observations satisfying:

    observation_time_utc <= execution_cutoff_utc

The execution cutoff must be generated automatically at runtime.

Future observations are prohibited.

The primary input source, geographic boundary, magnitude handling,
quality rules and feature definitions must be frozen before the first
prospective run.

## 5. Frozen geographic domain

The monitored geographic domain must remain identical to the domain
used by the committed Campi Flegrei feature-generation configuration.

The domain must not be enlarged, reduced or repositioned after observing
prospective outcomes.

## 6. Execution schedule

Primary cadence:

    once every hour

Each scheduled execution must produce one immutable prospective record.

If input data are unavailable, delayed or invalid, the execution must
record:

    DATA_UNAVAILABLE

It must not silently reuse old data as though they were current.

## 7. Prospective feature generation

Each execution must rebuild the permitted feature windows from the
catalogue available at the execution cutoff.

Frozen windows:

- 1 day;
- 3 days;
- 7 days;
- 30 days.

Feature-window exclusions already committed before prospective execution
must remain unchanged.

No future-centred or centred rolling windows are allowed.

## 8. Frozen normalization and thresholds

Normalization parameters and alert thresholds must be estimated only
from the previously committed baseline period.

They must not be recomputed from prospective observations.

Prospective observations must never alter:

- baseline medians;
- scale parameters;
- alert percentiles;
- persistence requirements;
- multi-window confirmation rules;
- feature directions;
- feature-window exclusions.

## 9. Frozen primary metrics

The prospective engine must calculate, where applicable:

- Gamma_CF;
- SCI;
- CRM;
- per-window alert level;
- global alert level;
- confirming-window count;
- confirming-window identities.

Every value must be stored with its execution timestamp and cutoff.

## 10. Prospective state classes

Allowed global states:

- normal;
- attention;
- anomaly;
- structural;
- data_unavailable;
- invalid_execution.

No additional state may be introduced after prospective results have
been observed without closing the current experiment and opening a new
version.

## 11. Alert persistence

The frozen persistence and reset rules already committed for the radar
must be applied without retrospective alteration.

A later execution may update the current state, but it must never delete
or overwrite a previous state record.

## 12. Immutable prospective record

Every execution must generate a record containing at least:

- experiment ID;
- protocol version;
- execution timestamp UTC;
- execution cutoff UTC;
- source commit;
- workflow run ID;
- configuration hashes;
- script hashes;
- input catalogue hash;
- latest included event timestamp;
- input event count;
- Gamma_CF by window;
- SCI by window;
- CRM by window;
- alert level by window;
- global alert level;
- confirming windows;
- data-quality state;
- error state, when applicable.

## 13. Append-only requirement

Prospective records must be append-only.

A previously committed record may not be:

- deleted;
- rewritten;
- reordered to alter chronology;
- replaced by a regenerated value;
- removed because it was a false alert;
- removed because no significant event followed;
- removed because execution failed.

If a technical correction is required, it must be added as a new
correction record referring explicitly to the original record.

## 14. Timestamp integrity

Each prospective record must contain timestamps from at least two
independent provenance layers:

1. timestamp written inside the generated record;
2. GitHub Actions workflow-run timestamp and commit history.

The workflow run ID and source commit must be stored.

## 15. Data revisions

If the seismic catalogue later revises an event, the original prospective
record must remain unchanged.

A later record may report the revised catalogue state, but it must include:

- revision detected: true;
- previous input hash;
- current input hash;
- revision timestamp.

## 16. Event evaluation

Future event evaluation must be performed by a separate workflow and
must not alter monitoring records.

Primary significant-event threshold:

    magnitude >= 3.0

Primary association window:

    168 hours after the beginning of an alert episode

These values are retained only for later evaluation and do not change
the prospective monitoring state.

## 17. Primary prospective questions

CF-PROS-01 must answer:

1. How often does the radar leave the normal state?
2. How much time is spent in each alert state?
3. Are alerts persistent or transient?
4. How many significant event clusters are preceded by an alert?
5. How many alerts are not followed by significant events?
6. Is alert-event association stronger than frozen placebo controls?
7. Does performance remain stable across time?
8. Does the radar add information beyond event count, magnitude and energy?
9. Are results reproducible from the committed inputs and code?
10. Does the prospective result confirm or falsify retrospective findings?

## 18. Minimum observation period

No confirmatory conclusion may be issued before both conditions hold:

- at least 180 days of prospective monitoring;
- at least 5 significant event clusters under the frozen definition.

If fewer than 5 significant clusters occur, the result must be classified:

    INCONCLUSIVE_LOW_POWER

Longer monitoring is permitted without changing the protocol.

## 19. Evidence classes

### prospectively_specific

All must hold:

- minimum observation requirements satisfied;
- empirical p-value below 0.05;
- observed association exceeds the placebo median;
- false-discovery fraction below 0.50;
- result is not dependent on one alert or one event cluster;
- no critical provenance violation.

### preliminary

Applied when:

- the direction is favourable;
- minimum observation requirements are satisfied;
- statistical significance is not reached;
- no critical provenance violation exists.

### non_discriminating

Applied when:

- observed association does not exceed the placebo median;
- or false-discovery fraction is excessive;
- or the radar remains persistently active without useful discrimination.

### inconclusive_low_power

Applied when:

- fewer than 180 monitoring days are available;
- or fewer than 5 significant event clusters are available.

### invalid

Applied when:

- future observations entered an execution;
- timestamps cannot be verified;
- prior records were overwritten;
- thresholds were altered during the experiment;
- provenance is incomplete in a way that compromises interpretation.

## 20. Required controls

The final evaluation must include:

- circular alert-time shifts;
- circular event-time shifts;
- conventional-metric comparison;
- alert-time fraction;
- false-discovery fraction;
- leave-one-alert-out analysis;
- leave-one-event-cluster-out analysis;
- sensitivity analyses labelled exploratory;
- missing-data and catalogue-delay audit.

## 21. Required falsifications

The final report must retain explicitly:

- every unassociated alert;
- every significant event not preceded by an alert;
- every invalid or unavailable execution;
- every interval of excessive alert burden;
- every equivalence with conventional metrics;
- every result compatible with chance;
- every protocol deviation.

## 22. Prohibited practices

The following are prohibited during CF-PROS-01:

- altering thresholds after prospective monitoring begins;
- altering feature directions;
- altering selected features;
- altering feature-window exclusions;
- altering persistence rules;
- altering multi-window confirmation;
- deleting false alerts;
- deleting missed events;
- replacing failed executions;
- backfilling old timestamps as prospective records;
- using observations later than the execution cutoff;
- selecting only favourable monitoring intervals;
- restarting the experiment after an unfavourable result;
- presenting a research alert as a public warning;
- presenting structural monitoring as earthquake prediction.

## 23. Version changes

Any material change requires:

1. closure of CF-PROS-01 version 1.0;
2. preservation of all version 1.0 records;
3. creation of a new experiment identifier or version;
4. a committed rationale written before new results are observed.

Version 1.0 must never be retroactively rewritten.

## 24. Required prospective outputs

Each hourly run must update or generate:

- CF_PROS_01_latest.json;
- CF_PROS_01_latest.md;
- CF_PROS_01_records.csv;
- CF_PROS_01_records.jsonl;
- CF_PROS_01_state_history.csv;
- CF_PROS_01_data_quality.csv;
- CF_PROS_01_manifest.json;
- CF_PROS_01_workflow_provenance.json.

The historical CSV and JSONL files must be append-only.

## 25. Public and scientific communication

Permitted language:

    The radar recorded a prospective structural state under a frozen
    research protocol.

Prohibited language:

    The radar predicted an earthquake.

Any public-facing interpretation must defer to official institutional
sources for hazard and safety information.

## 26. Final interpretation boundary

CF-PROS-01 can provide evidence about the prospective temporal behaviour
and possible discriminative value of a structural-instability radar.

Only subsequent independent replication could support generalisation.

No result from CF-PROS-01 alone can justify operational earthquake
warnings or claims of deterministic prediction.

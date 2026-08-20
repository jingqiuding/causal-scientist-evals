# Methodology

## 1. Evaluation objective

This benchmark studies **causal ontology discovery**: whether an AI-scientist agent can introduce an operational concept that makes the causal structure of a problem easier to state, test, and use.

This differs from asking whether an agent can:

- fit observational data;
- guess the next observation;
- restate variables in different language;
- generate a plausible mechanism after seeing the answer; or
- write a convincing scientific narrative.

The object of evaluation is the proposal together with its measurement and intervention semantics. A good proposal must earn value on evidence unavailable when it was produced.

### Current implementation boundary

Version 0.1 implements the typed concept proposal, an executable measurement and linear outcome model, one deterministic structural-causal environment family, observational/calibration-intervention/runtime-hidden-intervention/counterfactual/transport splits, component scoring, and reference anti-gaming proposals. The scorer uses evaluator-controlled tolerances and rejects non-finite or structurally malformed submissions. Independent-measurement, human-review, matched-baseline, and statistical protocols below are design specifications for later benchmark suites; they are not empirical capabilities of the default generator today.

## 2. Unit of evaluation

An evaluation episode contains four logical parts:

1. **Visible problem:** the scientific setting, available variables, observational evidence, and allowed tools.
2. **Submission contract:** the structured fields an agent must return.
3. **Sealed evidence:** interventions, counterfactual queries, held-out environments, or independent measurements hidden during proposal generation.
4. **Reference comparisons:** null concepts, known concepts, ablations, or candidates generated with matched information and compute.

Synthetic structural causal models are useful in an initial prototype because their ground truth and interventions can be generated exactly. They are not a substitute for later semi-synthetic and real scientific tasks.

### 2.1 Concept-card contract

Each proposed concept should specify the following fields.

| Field | Required content | Why it matters |
| --- | --- | --- |
| Domain and type | Units, support, level of aggregation, and applicable entities | Prevents an attractive label from remaining semantically undefined |
| Measurement or encoder | A reproducible mapping from observations to the concept | Makes the representation executable and exposes measurement cost |
| Outcome model | A runnable mapping from measured inputs and the concept to the scored outcome | Mechanically ties predictions to the proposed representation |
| Uncertainty | Measurement error, posterior uncertainty, or a calibrated confidence model | Prevents point estimates from masquerading as certain facts |
| Causal role | Proposed parents, children, mediator/moderator role, or abstraction relation | Connects the concept to a causal model |
| Intervention map | How an intervention on the concept maps to lower-level actions | Distinguishes causal use from correlation |
| Non-manipulability note | If direct intervention is incoherent, why, and which downstream tests remain valid | Avoids forcing all useful constructs into an invalid intervention semantics |
| Scope | Environments, populations, and regimes where the proposal is expected to hold | Enables transport and boundary testing |
| Predictions | Quantitative or rank-order consequences not used to form the proposal | Creates prospective tests |
| Falsifiers | Outcomes that would count as evidence against the proposal | Rules out unfalsifiable storytelling |

A parser may enforce structural completeness, but syntactic validity is not substantive correctness. Missing fields, ambiguous encoders, and unexecutable intervention descriptions should be reported explicitly rather than silently repaired by the evaluator.

## 3. Evidence separation

The core protection is a strict boundary between proposal evidence and evaluation evidence.

### 3.1 Sealed interventions

The agent may see observational data or a subset of interventions. The scorer retains interventions on relevant variables, placebo interventions, and interventions that break nuisance correlations. A useful concept should improve predictions of these outcomes without being refit to them.

### 3.2 Counterfactual queries

Counterfactual questions should share latent background conditions with an observed case while changing a specified action. They test whether the proposal supports an internally consistent causal model, not merely a population-level association. Where individual counterfactuals are not identifiable, the benchmark should test only identified quantities and state that restriction.

### 3.3 Held-out environments

Held-out environments can vary nuisance distributions, measurement channels, mechanism parameters, or intervention policies. The split should preserve the proposed invariant mechanism while changing shortcuts available in the visible environment. Performance should be reported per shift type.

### 3.4 Independent measurements

Whenever possible, the sealed set should include a second sensor, assay, annotator, or encoder that was not used to define the concept. Agreement across independent measurement paths is stronger evidence than evaluating a latent variable with the same model that produced it.

## 4. Distinguishing forms of novelty

“Novel” should never be a single binary label.

### 4.1 Historical novelty

Historical novelty asks whether an idea existed in the relevant record before a cutoff. Establishing it requires a scoped corpus, search protocol, date boundary, domain experts, and a documented rule for partial precedents. A benchmark generated from known causal models cannot, by itself, establish historical novelty.

### 4.2 Representational novelty

A representation is not substantively new if it is an invertible transformation of existing variables and has the same predictions, interventions, and complexity once the transformation is counted. Equivalence tests should include explicit algebraic checks where possible and behavioral probes otherwise.

### 4.3 Granularity novelty

A proposal can add value by choosing a better scale: aggregating microstates into a stable macro-variable or decomposing a coarse variable into causally distinct subtypes. The evaluation should test whether the chosen granularity improves modularity, transport, or intervention targeting.

### 4.4 Operational novelty

A concept may be operationally new when it supplies a new feasible measurement or intervention procedure. The cost, error model, and domain of that procedure are part of the proposal—not free implementation details.

### 4.5 Causal utility

Causal utility asks whether the concept improves decisions or explanations on sealed causal tests. It is the prototype's primary empirical target and does not imply the other forms of novelty.

## 5. Causal-value measurements

The benchmark should retain a vector of component measurements. A single aggregate, if used, must be declared before inspecting model outputs.

Potential components include:

- **Interventional prediction:** change in proper scoring rule or loss for sealed intervention outcomes.
- **Counterfactual consistency:** error on identified counterfactual quantities and consistency across related queries.
- **Transport:** performance and calibration in held-out environments, including worst-group or worst-environment results.
- **Causal-abstraction consistency:** whether high-level interventions commute with their stated low-level realization.
- **Independent-measurement agreement:** convergence across measurement paths with their errors propagated.
- **Modularity and sparsity:** whether the concept isolates mechanisms without adding unnecessary dependencies.
- **Stability:** sensitivity to resampling, nuisance perturbations, prompt variation, and equivalent input representations.
- **Total description length:** concept specification plus encoder, measurement, exceptions, and downstream model—not the latent dimension alone.
- **Experiment information gain:** expected reduction in uncertainty from an experiment proposed using the concept.

For a mature benchmark and component utility vector \(U\), incremental causal value should be evaluated by paired comparison:

$$
\Delta U(c; b) = U(c) - U(b),
$$

where \(c\) is the proposed concept and \(b\) is a baseline with matched access and budget. This is a comparison, not proof that the concept is the true ontology. Version 0.1 does **not** implement this matched delta: it reports an absolute diagnostic vector and separately scores hand-authored fixtures, some of which use generator ground truth.

### 5.1 Executability gate

A proposal should not receive a positive causal-utility claim unless:

1. its required concept-card fields are present;
2. its encoder or measurement is runnable or precisely reproducible;
3. it entails at least one prospective causal consequence;
4. its falsifier could occur in the benchmark; and
5. its sealed score improves on a declared baseline without an offsetting, hidden complexity increase.

The gate should be reported separately from the quality score so that parse and specification failures are visible.

## 6. Baselines and degeneracy checks

In a mature evaluation, baselines should receive the same observations, tools, and effective compute budget as the evaluated agent. Relevant comparisons include:

- an observational predictor with no causal concept;
- the supplied variables without a proposed abstraction;
- a renamed-variable baseline;
- an invertible reparameterization;
- a high-capacity lookup encoder;
- a simple hand-authored causal concept; and
- ablations that remove the candidate concept but keep downstream model capacity fixed.

The v0.1 `empty`, `treatment_renaming`, `lookup_table`, and `compact_causal_oracle` implementations are deterministic reference fixtures for testing scorer behavior. They are not matched agent baselines and their absolute score differences are not estimates of incremental causal value.

The evaluator should explicitly challenge four common degeneracies.

**Renaming.** Substitute synonyms and canonicalize formulas. If predictions and intervention semantics are unchanged, the proposal has not added representational value.

**Invertible reparameterization.** Recover the original variables and charge the transform's description length. A coordinate change may aid optimization, but it should not be reported as a newly discovered causal entity without additional consequences.

**Lookup-table latents.** Evaluate on unseen combinations and count encoder capacity, stored examples, retrieval data, and exceptions. Generalization from an opaque memorizer is not ontology discovery.

**Post-hoc stories.** Freeze proposals before revealing sealed outcomes. Log retries and prohibit evaluator feedback from entering a revised proposal unless the revision is scored as a separate adaptive round.

## 7. Recommended run protocol

1. Version and freeze the episode generator, seeds, visible/sealed split, and scoring rules.
2. Record the agent's model identifier, prompt, sampling settings, tools, token budget, retry budget, and wall-clock or compute limits.
3. Generate one structured proposal using visible evidence only.
4. Validate the submission without semantic repair; retain the raw response and all validation errors.
5. Execute the measurement/encoder in a restricted environment when code is allowed.
6. Evaluate predictions on sealed interventions, counterfactuals, environments, and independent measurements.
7. Run matched baselines and degeneracy probes.
8. Report all components, per-episode outputs, exclusions, parser failures, uncertainty intervals, and sensitivity analyses.
9. Keep exploratory analyses clearly separate from preregistered or confirmatory results.

For comparisons across systems, use paired episodes and confidence intervals over episodes. Report effect sizes rather than win counts alone. If many metrics or prompt variants are explored, distinguish selection data from final test data.

## 8. Optional Inspect AI integration

Inspect AI provides a reproducible execution layer for model calls, task definitions, scorers, and logs. The optional adapter included here:

- loads benchmark episodes without exposing sealed fields to the model;
- requires a structured concept-card response;
- treats format failures as observable outcomes;
- keeps deterministic causal scoring outside the model grader;
- executes the submitted measurement and outcome model with evaluator-controlled tolerances;
- stores raw transcripts and run metadata in Inspect logs; and
- uses the deterministic core scorer rather than a model grader.

After installing the `inspect` extra, run the registered task with:

```bash
inspect eval causal_scientist_evals/sealed_interventions \
  --model <provider/model>
```

The adapter is optional. Its presence demonstrates a reproducible execution path; it does not imply that a provider model has been run or that any particular model result exists.

## 9. Interpretation

Passing a v0.1 benchmark episode supports a narrow conclusion: under the public development generator, evidence split, and scoring rules, the submitted executable hypothesis fit the specified runtime-hidden causal cases. The absolute score does not establish incremental value over a matched agent baseline. It also does not establish consciousness, general causal understanding, historical discovery, scientific autonomy, alignment, or safety. See [Limitations](LIMITATIONS.md) for the main threats to validity.

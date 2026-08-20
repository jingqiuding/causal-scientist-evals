# Limitations and Responsible Interpretation

This repository is a new research prototype. Its outputs should be treated as evidence about a particular task construction—not as a certification of scientific ability or model safety.

## What the prototype cannot establish

A strong score does **not** by itself show that a system:

- discovered a historically new scientific idea;
- possesses a generally correct causal world model;
- can conduct autonomous science in an open-ended domain;
- will remain reliable outside the tested environments;
- is aligned, honest, or non-deceptive;
- understands the natural-language explanation it produced; or
- is safe to deploy in a laboratory or consequential decision process.

Likewise, a weak score may reflect an unclear prompt, parser failure, tool restriction, insufficient budget, or a mismatch between the benchmark ontology and a valid alternative—not an absence of causal reasoning.

## Construct validity

“Conceptual novelty” and “causal understanding” are contested constructs. The executable concept card deliberately narrows them to observable commitments, but it may exclude valuable tacit, qualitative, or non-manipulable concepts. Conversely, a system can satisfy a schema without having a coherent underlying representation.

The benchmark therefore separates historical, representational, granularity, operational, and causal-utility claims. Even taken together, these measurements are proxies rather than a complete theory of scientific discovery.

## Synthetic-to-real gap

Synthetic structural causal models provide known interventions and counterfactual ground truth, but real science rarely offers that access. Real systems contain measurement error, feedback, selection effects, nonstationarity, interference, partial observability, ethical constraints, and genuine ambiguity about the right variables. Success on generated episodes can reward familiarity with benchmark patterns rather than transferable scientific judgment.

Before real-world use, tasks should include independently designed semi-synthetic and empirical cases, domain-expert review, and prospective experiments where ethically appropriate.

## Ground-truth and identifiability assumptions

A generated “true” ontology is only one representation of a data-generating process. Multiple causal models may be observationally or interventionally equivalent over the available tests. Some individual counterfactuals are not identifiable. The scorer can mistakenly reject a useful alternative abstraction if it compares names or graphs too literally.

Evaluation should focus on declared, identifiable consequences and allow multiple semantically equivalent solutions. Unidentifiable claims should not be scored as facts.

## Novelty assessment

Historical novelty cannot be automated reliably by searching a small corpus or asking another language model. It requires a stated field and time boundary, systematic retrieval, expert adjudication, and uncertainty about missing precedents. This prototype centers causal utility and should not label a proposal “scientifically novel” from benchmark performance alone.

Semantic novelty is also difficult to judge. Renaming and invertible-transform checks catch only simple equivalences; learned encoders can hide an old representation inside a complex implementation.

## Evaluator dependence and gaming

Any fixed benchmark creates incentives to target its generator, output schema, scorer, or model grader. A candidate can exploit parser behavior, memorize generated families, infer sealed variables from leakage, or optimize a scalar aggregate while degrading important components.

Mitigations include private test generators, adversarially authored episodes, code review for leakage, canary tasks, component-level reporting, held-out benchmark families, and periodic benchmark retirement. These reduce risk but do not eliminate it.

The bundled v0.1 JSONL is a public development case. Its targets are hidden from the model at runtime by the Inspect adapter, but they are not private and provide no protection against prior inspection, fine-tuning contamination, or deliberate memorization. Credible system comparisons require non-public generated families and a clean final split.

## Encoder and measurement complexity

A low-dimensional latent can conceal a large lookup table, pretrained model, retrieval corpus, or manual exception list. If only the final representation is counted, description-length and sparsity metrics become misleading. A fair accounting must include the measurement procedure, encoder parameters, external data, computation, and exception handling. Such accounting is inherently approximate across different model families.

## Intervention semantics

Not every scientifically useful concept can be directly manipulated. Interventions on high-level variables may have several incompatible low-level realizations, and naive “set the concept to x” operations can violate the underlying system. The concept card permits a non-manipulability justification, but scoring these cases requires care: predictive value under downstream interventions is weaker evidence than a well-defined intervention on the concept itself.

## Counterfactual scoring

Counterfactual ground truth depends on structural assumptions and shared exogenous variables. If those assumptions are hidden or unrealistic, the benchmark may reward guessing the generator rather than causal reasoning. Counterfactual scores should be restricted to identified queries and reported separately from interventional prediction.

## Distribution shift and scope

Held-out environments sample only shifts imagined by benchmark authors. Strong transport within those shifts does not imply robustness to new mechanisms, adversarial perturbations, or regime changes. Each proposal's claimed scope should be recorded, and evaluation should distinguish justified abstention from incorrect prediction.

## Model graders and human review

Language-model graders may share biases, training data, or failure modes with the evaluated agent. They can prefer fluent explanations, miss mathematical equivalence, and be vulnerable to prompt injection. Deterministic tests should be used where possible. Model-graded judgments should retain rationales and be checked against blinded human adjudication with inter-rater agreement.

Human review is not infallible either. Domain experts can favor familiar concepts, infer author identity from style, or disagree about ontology. Blinding, multiple adjudicators, written rubrics, and documented disagreements are important.

## Statistical uncertainty

Small episode sets, correlated task variants, prompt selection, and repeated experimentation can produce unstable rankings. Reports should include paired uncertainty intervals, all tested configurations, parser and execution failure rates, and sensitivity to aggregation choices. Benchmark-development data must not be reused as a final test set without disclosure.

Version 0.1 contains one repeated deterministic mechanism family and an absolute hand-weighted score. Its oracle, renaming, lookup, and empty fixtures test intended scorer behavior; they are not matched model baselines, and their score gaps are not causal-effect estimates.

## Reproducibility limits

Hosted models, provider defaults, and tool APIs change over time. Even with a fixed seed, sampling and external services may be nondeterministic. Reproduction requires the code revision, environment lockfile, model/version identifier, prompts, sampling settings, tool permissions, budget, seeds, raw transcripts, and scorer outputs. Where model snapshots are unavailable, exact reproduction may be impossible.

## Safety and dual use

Better evaluations of AI-scientist agents can support safer deployment, but the same infrastructure may accelerate autonomous experimentation or help optimize systems against oversight. This prototype should not be connected to wet-lab equipment, production systems, or consequential actions without independent risk assessment, access controls, sandboxing, and human authorization.

The benchmark measures displayed task behavior. It does not infer deceptive intent or rule out strategic behavior. Evaluation awareness is itself a possible confounder.

## Reporting checklist

Every empirical report should state:

- the repository revision and benchmark version;
- whether episodes or generators were public during development;
- model identifiers and access dates;
- prompts, tools, budgets, retries, and sampling settings;
- visible versus sealed information;
- baseline matching rules;
- component metrics and aggregation choices;
- exclusions, parser failures, and execution failures;
- uncertainty intervals and sensitivity analyses;
- model-grader and human-review procedures; and
- known data contamination or leakage risks.

Until those conditions are met, results are exploratory. No measured language-model results are included or implied by this repository's documentation.

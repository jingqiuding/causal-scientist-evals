"""Prompts for the optional Inspect AI evaluation task."""

from __future__ import annotations

SYSTEM_PROMPT = """You are an AI-scientist agent in a causal concept evaluation.

Your job is to propose one operational concept that clarifies the causal structure of
the supplied system. A persuasive explanation is not sufficient: the concept must be
measurable, causally situated, prospectively predictive, scoped, and falsifiable.

Use only the evidence in the user message. Outcomes for prospective cases are sealed.
Return exactly one JSON object, with no Markdown fence or prose outside the object.
Do not claim historical novelty; this task measures executability and causal utility.
"""


OUTPUT_CONTRACT = """Required JSON contract:
- name: non-empty string
- domain: non-empty string
- concept_type: non-empty string
- measurement: object with operator, inputs, output_domain, description,
  encoder_complexity, and lookup_table. operator must be one of identity, product,
  sum, difference, threshold, lookup, or custom.
- uncertainty: object with kind, calibration_statement, and parameters. kind must be
  deterministic, interval, or probabilistic.
- causal_role: object with variable, role, parents, children, and structural_equation.
  role must be cause, mediator, moderator, confounder, effect, proxy, or unknown.
- outcome_model: executable linear model with target, finite intercept, and a
  coefficients object. It must use causal_role.variable as one predictor; other
  predictors may be measured inputs such as context.
- intervention_map: list of objects with target, values, predicted_change, and
  concept_is_directly_manipulable.
- scope: object with environments, population, and assumptions.
- predictions: one object per listed case, each with case_id, target="outcome",
  predicted_value, tolerance, and rationale.
- falsifiers: list of objects with statement, case_ids, and rejection_rule.

All predictions must be finite numbers and must agree with the executable measurement
plus outcome_model. Prediction tolerance is recorded but never relaxes the evaluator's
scoring tolerance. Use a compact measurement rule: computed lookup/input capacity and
declared encoder complexity are both counted.
"""

"""Deterministic scoring for executable causal concept proposals."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from .schema import ConceptProposal, MeasurementOperator, PredictionSpec
from .simulator import CausalBenchmark, EvaluationCase, Split


@dataclass(frozen=True)
class ScoreWeights:
    """Weights for causal utility and anti-gaming penalties."""

    visible_fit_prediction: float = 0.05
    interventional_prediction: float = 0.25
    counterfactual_prediction: float = 0.15
    transport_prediction: float = 0.15
    reference_ontology_match: float = 0.20
    executability: float = 0.20
    complexity_penalty: float = 0.10
    renaming_penalty: float = 0.12
    lookup_penalty: float = 0.18


@dataclass(frozen=True)
class ScoreBreakdown:
    """Auditable component scores and final score, all in ``[0, 1]``."""

    visible_fit_prediction: float
    interventional_prediction: float
    counterfactual_prediction: float
    transport_prediction: float
    reference_ontology_match: float
    executability: float
    complexity_penalty: float
    renaming_penalty: float
    lookup_penalty: float
    executability_gate: bool
    total: float

    def to_dict(self) -> dict[str, float | bool]:
        """Return a JSON-compatible report."""

        return {
            key: value if isinstance(value, bool) else round(float(value), 6)
            for key, value in asdict(self).items()
        }


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _prediction_accuracy(
    proposal: ConceptProposal,
    cases: tuple[EvaluationCase, ...],
    error_scale: float,
) -> float:
    """Execute the declared concept/model and score with evaluator tolerances."""

    if not cases:
        return 0.0
    scores: list[float] = []
    for case in cases:
        try:
            predicted_value = derived_prediction(proposal, case)
        except (KeyError, NotImplementedError, ValueError):
            scores.append(0.0)
            continue
        error = abs(predicted_value - case.expected_value)
        excess_error = max(0.0, error - case.tolerance)
        scores.append(max(0.0, 1.0 - excess_error / error_scale))
    return sum(scores) / len(scores)


def derived_prediction(proposal: ConceptProposal, case: EvaluationCase) -> float:
    """Derive a case outcome from the submitted measurement and outcome model."""

    values = dict(case.inputs)
    values.update(case.intervention)
    concept_value = proposal.measurement.evaluate(values, case_id=case.case_id)
    variable = proposal.causal_role.variable
    if not variable:
        raise ValueError("causal role has no concept variable")
    values[variable] = concept_value
    return proposal.outcome_model.evaluate(values)


def reference_ontology_match(
    proposal: ConceptProposal, benchmark: CausalBenchmark
) -> float:
    """Compare fields with the generator's reference ontology, not semantic truth."""

    truth = benchmark.truth
    claimed_targets = tuple(item.target for item in proposal.intervention_map)
    components = (
        (0.30, float(proposal.measurement.operator is truth.measurement_operator)),
        (0.25, _jaccard(proposal.measurement.inputs, truth.measurement_inputs)),
        (0.20, float(proposal.causal_role.role is truth.causal_role.role)),
        (0.10, _jaccard(proposal.causal_role.parents, truth.causal_role.parents)),
        (0.05, _jaccard(proposal.causal_role.children, truth.causal_role.children)),
        (0.10, _jaccard(claimed_targets, truth.intervention_targets)),
    )
    return sum(weight * value for weight, value in components)


# Compatibility alias for early notebooks; reports use the calibrated name above.
causal_role_correctness = reference_ontology_match


def benchmark_validation_issues(
    proposal: ConceptProposal, benchmark: CausalBenchmark
) -> tuple[str, ...]:
    """Return authoritative executable-card issues for this benchmark instance."""

    issues = list(proposal.executability_issues())
    expected = {case.case_id: case for case in benchmark.cases}
    if proposal.outcome_model.target != "outcome":
        issues.append("outcome model target must be 'outcome'")
    predictions: dict[str, PredictionSpec] = {}
    duplicates: set[str] = set()
    for prediction in proposal.predictions:
        if prediction.case_id in predictions:
            duplicates.add(prediction.case_id)
        predictions[prediction.case_id] = prediction
    if duplicates:
        issues.append(f"duplicate prediction IDs: {sorted(duplicates)}")
    missing = sorted(set(expected) - set(predictions))
    extra = sorted(set(predictions) - set(expected))
    if missing:
        issues.append(f"missing required prediction IDs: {missing}")
    if extra:
        issues.append(f"unknown prediction IDs: {extra}")

    inconsistent: list[str] = []
    for case_id, case in expected.items():
        prediction = predictions.get(case_id)
        if prediction is None:
            continue
        if prediction.target != case.target:
            issues.append(f"wrong prediction target for {case_id}")
            continue
        try:
            derived = derived_prediction(proposal, case)
        except (KeyError, NotImplementedError, ValueError) as exc:
            issues.append(f"model is not executable for {case_id}: {exc}")
            break
        if abs(prediction.predicted_value - derived) > max(case.tolerance, 1e-6):
            inconsistent.append(case_id)
    if inconsistent:
        issues.append(f"declared predictions disagree with executable model: {inconsistent}")

    valid_case_ids = set(expected)
    prospective_ids = {
        case.case_id
        for case in benchmark.cases
        if case.split
        not in {Split.OBSERVATIONAL, Split.VISIBLE_INTERVENTION}
    }
    falsifier_ids = {
        case_id for falsifier in proposal.falsifiers for case_id in falsifier.case_ids
    }
    unknown_falsifiers = sorted(falsifier_ids - valid_case_ids)
    if unknown_falsifiers:
        issues.append(f"falsifiers reference unknown cases: {unknown_falsifiers}")
    if proposal.falsifiers and not (falsifier_ids & prospective_ids):
        issues.append("falsifier does not reference a prospective case")

    known_environments = set(benchmark.environments)
    unknown_environments = sorted(set(proposal.scope.environments) - known_environments)
    if unknown_environments:
        issues.append(f"scope references unknown environments: {unknown_environments}")
    return tuple(dict.fromkeys(issues))


def executability_score(proposal: ConceptProposal) -> float:
    """Score whether each required proposal component can be tested."""

    measurement_executable = (
        bool(proposal.measurement.inputs)
        and bool(proposal.measurement.output_domain.strip())
        and proposal.measurement.operator is not MeasurementOperator.CUSTOM
    )
    checks = (
        bool(proposal.name.strip()),
        bool(proposal.domain.strip() and proposal.concept_type.strip()),
        measurement_executable,
        bool(proposal.uncertainty.calibration_statement.strip()),
        bool(
            proposal.causal_role.role.value != "unknown"
            and proposal.causal_role.variable.strip()
            and proposal.causal_role.structural_equation.strip()
        ),
        bool(
            proposal.outcome_model.target.strip()
            and proposal.outcome_model.coefficients
            and proposal.causal_role.variable in proposal.outcome_model.coefficients
        ),
        bool(proposal.intervention_map),
        bool(proposal.scope.environments and proposal.scope.population.strip()),
        bool(proposal.predictions),
        bool(
            proposal.falsifiers
            and all(
                item.statement.strip()
                and item.case_ids
                and item.rejection_rule.strip()
                for item in proposal.falsifiers
            )
        ),
    )
    # Measurement and causal graph claims carry two shares; textual labels carry one.
    shares = (1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0)
    return sum(
        share for share, passed in zip(shares, checks, strict=True) if passed
    ) / sum(shares)


def complexity_penalty(proposal: ConceptProposal, benchmark: CausalBenchmark) -> float:
    """Penalize effective complexity, including computed lookup/input capacity."""

    operator_floor = max(1.0, float(len(proposal.measurement.inputs) - 1))
    if proposal.measurement.operator is MeasurementOperator.LOOKUP:
        operator_floor = max(operator_floor, float(len(proposal.measurement.lookup_table)))
    effective_complexity = max(
        proposal.measurement.encoder_complexity, operator_floor
    )
    excess = max(
        0.0,
        effective_complexity - benchmark.truth.minimum_encoder_complexity,
    )
    # Smooth bounded transform: five extra units produce a 0.5 penalty.
    return excess / (excess + 5.0) if excess else 0.0


def renaming_penalty(proposal: ConceptProposal) -> float:
    """Detect a new label attached to an unchanged observed variable."""

    measurement = proposal.measurement
    if measurement.operator is not MeasurementOperator.IDENTITY or len(measurement.inputs) != 1:
        return 0.0
    input_name = measurement.inputs[0].strip().lower().replace("_", " ")
    concept_name = proposal.name.strip().lower().replace("_", " ")
    return 1.0 if concept_name and concept_name != input_name else 0.5


def score_proposal(
    proposal: ConceptProposal,
    benchmark: CausalBenchmark,
    weights: ScoreWeights | None = None,
) -> ScoreBreakdown:
    """Compute an absolute prototype score across visible and prospective cases.

    Perfect row memorization can earn predictive credit, but lookup-table and
    description-length penalties prevent it from matching a compact causal
    abstraction.  Similarly, renaming a raw input receives no novelty credit.
    """

    weights = weights or ScoreWeights()
    visible_cases = (
        benchmark.cases_for(Split.OBSERVATIONAL)
        + benchmark.cases_for(Split.VISIBLE_INTERVENTION)
    )
    visible = _prediction_accuracy(
        proposal, visible_cases, benchmark.error_scale
    )
    interventional = _prediction_accuracy(
        proposal,
        benchmark.cases_for(Split.SEALED_INTERVENTION),
        benchmark.error_scale,
    )
    counterfactual = _prediction_accuracy(
        proposal, benchmark.cases_for(Split.COUNTERFACTUAL), benchmark.error_scale
    )
    transport = _prediction_accuracy(
        proposal, benchmark.cases_for(Split.TRANSPORT), benchmark.error_scale
    )
    ontology_match = reference_ontology_match(proposal, benchmark)
    executable = executability_score(proposal)
    complexity = complexity_penalty(proposal, benchmark)
    renaming = renaming_penalty(proposal)
    lookup = float(proposal.measurement.operator is MeasurementOperator.LOOKUP)

    positive = (
        weights.visible_fit_prediction * visible
        + weights.interventional_prediction * interventional
        + weights.counterfactual_prediction * counterfactual
        + weights.transport_prediction * transport
        + weights.reference_ontology_match * ontology_match
        + weights.executability * executable
    )
    penalties = (
        weights.complexity_penalty * complexity
        + weights.renaming_penalty * renaming
        + weights.lookup_penalty * lookup
    )
    gate_passed = not benchmark_validation_issues(proposal, benchmark)
    # Incomplete concept cards remain visible in component diagnostics but may
    # not support a positive headline causal-utility claim.
    total = min(1.0, max(0.0, positive - penalties)) if gate_passed else 0.0
    return ScoreBreakdown(
        visible_fit_prediction=visible,
        interventional_prediction=interventional,
        counterfactual_prediction=counterfactual,
        transport_prediction=transport,
        reference_ontology_match=ontology_match,
        executability=executable,
        complexity_penalty=complexity,
        renaming_penalty=renaming,
        lookup_penalty=lookup,
        executability_gate=gate_passed,
        total=total,
    )

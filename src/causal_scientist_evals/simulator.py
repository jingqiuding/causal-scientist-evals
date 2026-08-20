"""Deterministic structural-causal environments for concept evaluation.

The default benchmark is intentionally small and auditable.  In the training
environment, ``treatment`` and ``gate`` are always aligned, so the raw treatment
is observationally indistinguishable from the causal concept
``activated_signal = treatment * gate``. Disclosed calibration interventions
identify the interaction before separate runtime-hidden interventions test it;
a transported environment then reverses the treatment policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from .schema import (
    CausalRole,
    CausalRoleSpec,
    ConceptProposal,
    FalsifierSpec,
    InterventionSpec,
    MeasurementOperator,
    MeasurementSpec,
    OutcomeModelSpec,
    PredictionSpec,
    ScopeSpec,
    UncertaintyKind,
    UncertaintySpec,
)


class Split(StrEnum):
    """Evaluation split, including hidden intervention and transport tests."""

    OBSERVATIONAL = "observational"
    VISIBLE_INTERVENTION = "visible_intervention"
    SEALED_INTERVENTION = "sealed_intervention"
    COUNTERFACTUAL = "counterfactual"
    TRANSPORT = "transport"


@dataclass(frozen=True)
class StructuralCausalEnvironment:
    """A deterministic SCM with an environment-specific treatment policy."""

    name: str
    policy: str = "aligned"
    contexts: tuple[float, ...] = (-1.0, 1.0, -1.0, 1.0)
    gates: tuple[float, ...] = (0.0, 1.0, 1.0, 0.0)
    signal_effect: float = 3.0
    context_effect: float = 2.0

    def __post_init__(self) -> None:
        if self.policy not in {"aligned", "opposed"}:
            raise ValueError("policy must be 'aligned' or 'opposed'")
        if not self.contexts or len(self.contexts) != len(self.gates):
            raise ValueError("contexts and gates must be non-empty and equally sized")

    def simulate(
        self, unit_id: int, interventions: Mapping[str, float] | None = None
    ) -> dict[str, float]:
        """Run the SCM for one unit under optional surgical interventions."""

        if unit_id < 0:
            raise ValueError("unit_id must be non-negative")
        interventions = dict(interventions or {})
        unknown = set(interventions) - {"context", "gate", "treatment"}
        if unknown:
            raise KeyError(f"unknown intervention targets: {sorted(unknown)}")

        index = unit_id % len(self.contexts)
        context = float(interventions.get("context", self.contexts[index]))
        natural_gate = float(self.gates[index])
        gate = float(interventions.get("gate", natural_gate))
        natural_treatment = natural_gate if self.policy == "aligned" else 1.0 - natural_gate
        treatment = float(interventions.get("treatment", natural_treatment))

        activated_signal = treatment * gate
        outcome = self.signal_effect * activated_signal + self.context_effect * context
        return {
            "context": context,
            "gate": gate,
            "treatment": treatment,
            "activated_signal": activated_signal,
            "outcome": outcome,
        }


@dataclass(frozen=True)
class EvaluationCase:
    """One observable input/intervention bundle with a held-out target."""

    case_id: str
    split: Split
    environment: str
    unit_id: int
    inputs: Mapping[str, float]
    intervention: Mapping[str, float]
    target: str
    expected_value: float
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", Split(self.split))
        object.__setattr__(self, "inputs", dict(self.inputs))
        object.__setattr__(self, "intervention", dict(self.intervention))


@dataclass(frozen=True)
class ConceptGroundTruth:
    """Minimal causal facts used for structural scoring."""

    canonical_name: str
    measurement_operator: MeasurementOperator
    measurement_inputs: tuple[str, ...]
    causal_role: CausalRoleSpec
    intervention_targets: tuple[str, ...]
    scope_environments: tuple[str, ...]
    minimum_encoder_complexity: float = 1.0


@dataclass(frozen=True)
class CausalBenchmark:
    """A deterministic benchmark with explicit visible and runtime-hidden splits."""

    environments: Mapping[str, StructuralCausalEnvironment]
    cases: tuple[EvaluationCase, ...]
    truth: ConceptGroundTruth
    error_scale: float = 7.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "environments", dict(self.environments))
        object.__setattr__(self, "cases", tuple(self.cases))
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("evaluation case IDs must be unique")

    def cases_for(self, split: Split) -> tuple[EvaluationCase, ...]:
        """Return cases belonging to one split in stable order."""

        split = Split(split)
        return tuple(case for case in self.cases if case.split is split)


def _make_case(
    environment: StructuralCausalEnvironment,
    split: Split,
    unit_id: int,
    intervention: Mapping[str, float],
    serial: int,
) -> EvaluationCase:
    values = environment.simulate(unit_id, intervention)
    # ``activated_signal`` is the latent abstraction to be discovered, never a
    # visible feature.  The harness exposes only directly measured variables.
    inputs = {
        key: values[key]
        for key in ("context", "gate", "treatment")
    }
    return EvaluationCase(
        case_id=f"{split.value}-{serial:02d}",
        split=split,
        environment=environment.name,
        unit_id=unit_id,
        inputs=inputs,
        intervention=dict(intervention),
        target="outcome",
        expected_value=values["outcome"],
    )


def _make_counterfactual_case(
    environment: StructuralCausalEnvironment,
    unit_id: int,
    intervention: Mapping[str, float],
    serial: int,
) -> EvaluationCase:
    """Hold a unit's background fixed while changing its action."""

    factual = environment.simulate(unit_id)
    counterfactual = environment.simulate(unit_id, intervention)
    return EvaluationCase(
        case_id=f"{Split.COUNTERFACTUAL.value}-{serial:02d}",
        split=Split.COUNTERFACTUAL,
        environment=environment.name,
        unit_id=unit_id,
        inputs={key: factual[key] for key in ("context", "gate", "treatment")},
        intervention=dict(intervention),
        target="outcome",
        expected_value=counterfactual["outcome"],
    )


def build_default_benchmark() -> CausalBenchmark:
    """Build the canonical calibration/sealed/counterfactual/transport benchmark."""

    lab = StructuralCausalEnvironment(name="aligned_lab", policy="aligned")
    field = StructuralCausalEnvironment(
        name="opposed_field",
        policy="opposed",
        contexts=(1.0, -1.0, 1.0, -1.0),
        gates=(0.0, 1.0, 1.0, 0.0),
    )

    cases: list[EvaluationCase] = []
    for serial, unit_id in enumerate(range(8)):
        cases.append(_make_case(lab, Split.OBSERVATIONAL, unit_id, {}, serial))

    # Two disclosed calibration interventions break the treatment==gate alias.
    # The agent sees these outcomes, then commits before separate sealed cases.
    visible_interventions = (
        (0, {"treatment": 1.0, "gate": 0.0}),
        (1, {"treatment": 0.0, "gate": 1.0}),
    )
    for serial, (unit_id, intervention) in enumerate(visible_interventions):
        cases.append(
            _make_case(
                lab,
                Split.VISIBLE_INTERVENTION,
                unit_id,
                intervention,
                serial,
            )
        )

    # Crossed do(treatment, gate) cases distinguish treatment from treatment*gate.
    interventions = (
        {"treatment": 0.0, "gate": 0.0},
        {"treatment": 0.0, "gate": 1.0},
        {"treatment": 1.0, "gate": 0.0},
        {"treatment": 1.0, "gate": 1.0},
    )
    for serial, intervention in enumerate(interventions):
        cases.append(
            _make_case(lab, Split.SEALED_INTERVENTION, serial, intervention, serial)
        )

    # Individual counterfactuals reuse each factual unit's context and gate but
    # flip its treatment.  This tests internally consistent alternate actions.
    for serial, unit_id in enumerate(range(4)):
        factual_treatment = lab.simulate(unit_id)["treatment"]
        cases.append(
            _make_counterfactual_case(
                lab,
                unit_id,
                {"treatment": 1.0 - factual_treatment},
                serial,
            )
        )

    for serial, unit_id in enumerate(range(8)):
        cases.append(_make_case(field, Split.TRANSPORT, unit_id, {}, serial))

    truth = ConceptGroundTruth(
        canonical_name="activated_signal",
        measurement_operator=MeasurementOperator.PRODUCT,
        measurement_inputs=("treatment", "gate"),
        causal_role=CausalRoleSpec(
            variable="activated_signal",
            role=CausalRole.MEDIATOR,
            parents=("treatment", "gate"),
            children=("outcome",),
            structural_equation="activated_signal = treatment * gate",
        ),
        intervention_targets=("treatment", "gate"),
        scope_environments=(lab.name, field.name),
    )
    return CausalBenchmark(
        environments={lab.name: lab, field.name: field},
        cases=tuple(cases),
        truth=truth,
    )


# Short alias convenient in notebooks and command-line harnesses.
default_benchmark = build_default_benchmark


def _truth_predictions(benchmark: CausalBenchmark) -> tuple[PredictionSpec, ...]:
    return tuple(
        PredictionSpec(
            case_id=case.case_id,
            target=case.target,
            predicted_value=case.expected_value,
            tolerance=case.tolerance,
            rationale="SCM prediction from activated signal and context",
        )
        for case in benchmark.cases
    )


def oracle_proposal(benchmark: CausalBenchmark | None = None) -> ConceptProposal:
    """Return an executable compact proposal matching the true abstraction."""

    benchmark = benchmark or build_default_benchmark()
    truth = benchmark.truth
    return ConceptProposal(
        name=truth.canonical_name,
        domain="binary treatment-gate systems",
        concept_type="derived causal mediator",
        measurement=MeasurementSpec(
            operator=truth.measurement_operator,
            inputs=truth.measurement_inputs,
            output_domain="{0, 1}",
            description="Product of administered treatment and an open gate",
            encoder_complexity=truth.minimum_encoder_complexity,
        ),
        uncertainty=UncertaintySpec(
            kind=UncertaintyKind.DETERMINISTIC,
            calibration_statement="Exact in the deterministic benchmark SCM",
        ),
        causal_role=truth.causal_role,
        outcome_model=OutcomeModelSpec(
            target="outcome",
            coefficients={truth.canonical_name: 3.0, "context": 2.0},
        ),
        intervention_map=tuple(
            InterventionSpec(
                target=target,
                values=(0.0, 1.0),
                predicted_change="Changes activated signal according to treatment * gate",
            )
            for target in truth.intervention_targets
        ),
        scope=ScopeSpec(
            environments=truth.scope_environments,
            population="Units generated by the treatment-gate SCM",
            assumptions=("stable outcome mechanism", "measured treatment and gate"),
        ),
        predictions=_truth_predictions(benchmark),
        falsifiers=(
            FalsifierSpec(
                statement="A crossed intervention contradicts the product mechanism",
                case_ids=tuple(
                    case.case_id
                    for case in benchmark.cases_for(Split.SEALED_INTERVENTION)
                ),
                rejection_rule="Reject if any deterministic prediction error exceeds 1e-9",
            ),
        ),
    )


def empty_proposal() -> ConceptProposal:
    """Return a no-information baseline."""

    return ConceptProposal()


def lookup_proposal(benchmark: CausalBenchmark | None = None) -> ConceptProposal:
    """Return a memorizing baseline with perfect predictions but no abstraction."""

    benchmark = benchmark or build_default_benchmark()
    table = {
        case.case_id: benchmark.environments[case.environment].simulate(
            case.unit_id, case.intervention
        )["activated_signal"]
        for case in benchmark.cases
    }
    return ConceptProposal(
        name="case code",
        domain="benchmark rows",
        concept_type="lookup latent",
        measurement=MeasurementSpec(
            operator=MeasurementOperator.LOOKUP,
            inputs=("case_id",),
            output_domain="real",
            description="One memorized latent value per benchmark row",
            encoder_complexity=float(len(table)),
            lookup_table=table,
        ),
        uncertainty=UncertaintySpec(
            kind=UncertaintyKind.DETERMINISTIC,
            calibration_statement="Claims certainty only on memorized rows",
        ),
        causal_role=CausalRoleSpec(
            variable="case code",
            role=CausalRole.PROXY,
            structural_equation="case code = table[case_id]",
        ),
        outcome_model=OutcomeModelSpec(
            target="outcome", coefficients={"case code": 3.0, "context": 2.0}
        ),
        intervention_map=(
            InterventionSpec(
                target="case_id",
                predicted_change="Selects another memorized row",
                concept_is_directly_manipulable=False,
            ),
        ),
        scope=ScopeSpec(
            environments=benchmark.truth.scope_environments,
            population="Only enumerated benchmark rows",
            assumptions=("case IDs are available",),
        ),
        predictions=_truth_predictions(benchmark),
        falsifiers=(
            FalsifierSpec(
                statement="A fresh transported row is not compressible by the table",
                case_ids=("transport-00",),
                rejection_rule="Reject if prediction requires a row-specific stored value",
            ),
        ),
    )


def renaming_proposal(benchmark: CausalBenchmark | None = None) -> ConceptProposal:
    """Return the treatment-only model that fits the observational split."""

    benchmark = benchmark or build_default_benchmark()
    return ConceptProposal(
        name="treatment intensity concept",
        domain="binary treatment-gate systems",
        concept_type="renamed observed variable",
        measurement=MeasurementSpec(
            operator=MeasurementOperator.IDENTITY,
            inputs=("treatment",),
            output_domain="{0, 1}",
            description="Treatment under a new label",
            encoder_complexity=1.0,
        ),
        uncertainty=UncertaintySpec(
            calibration_statement="Deterministic identity of measured treatment"
        ),
        causal_role=CausalRoleSpec(
            variable="treatment intensity concept",
            role=CausalRole.CAUSE,
            parents=(),
            children=("outcome",),
            structural_equation="treatment intensity concept = treatment",
        ),
        outcome_model=OutcomeModelSpec(
            target="outcome",
            coefficients={"treatment intensity concept": 3.0, "context": 2.0},
        ),
        intervention_map=(
            InterventionSpec(
                target="treatment", values=(0.0, 1.0), predicted_change="Identity"
            ),
        ),
        scope=ScopeSpec(
            environments=benchmark.truth.scope_environments,
            population="Units generated by the treatment-gate SCM",
        ),
        predictions=tuple(
            PredictionSpec(
                case_id=case.case_id,
                target=case.target,
                # In the aligned training policy, treatment == treatment*gate.
                # Extrapolating that observational fit fails for crossed do-cases
                # and for the opposed transport policy.
                predicted_value=(
                    3.0
                    * case.intervention.get("treatment", case.inputs["treatment"])
                    + 2.0 * case.inputs["context"]
                ),
                rationale="Observational treatment-only extrapolation",
            )
            for case in benchmark.cases
        ),
        falsifiers=(
            FalsifierSpec(
                statement="Treatment fails when gate is closed",
                case_ids=("sealed_intervention-02",),
                rejection_rule="Reject identity explanation when effect is zero at gate=0",
            ),
        ),
    )

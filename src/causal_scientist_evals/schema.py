"""Typed schemas for executable causal concept proposals.

The benchmark deliberately represents a proposed scientific concept as more
than a name and a paragraph.  A proposal must say how the concept is measured,
where uncertainty enters, what causal role it plays, how it can be intervened
on, where it applies, what it predicts, and what would falsify it.

Only Python's standard library is used so the schema remains easy to inspect,
serialize, and use in lightweight evaluation harnesses.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from typing import Any


class MeasurementOperator(StrEnum):
    """Small executable vocabulary for concept measurements."""

    IDENTITY = "identity"
    PRODUCT = "product"
    SUM = "sum"
    DIFFERENCE = "difference"
    THRESHOLD = "threshold"
    LOOKUP = "lookup"
    CUSTOM = "custom"


class UncertaintyKind(StrEnum):
    """How uncertainty attached to a proposed concept is represented."""

    DETERMINISTIC = "deterministic"
    INTERVAL = "interval"
    PROBABILISTIC = "probabilistic"


class CausalRole(StrEnum):
    """Canonical causal roles used by the compact benchmark."""

    CAUSE = "cause"
    MEDIATOR = "mediator"
    MODERATOR = "moderator"
    CONFOUNDER = "confounder"
    EFFECT = "effect"
    PROXY = "proxy"
    UNKNOWN = "unknown"


def _finite_float(value: object, field_name: str) -> float:
    """Return a finite real number without accepting booleans or numeric strings."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of strings")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must contain only strings")
    return tuple(value)


def _numeric_mapping(value: object, field_name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object of finite numbers")
    result: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} keys must be strings")
        result[key] = _finite_float(item, f"{field_name}.{key}")
    return result


@dataclass(frozen=True)
class MeasurementSpec:
    """An executable encoder or measurement for the proposed concept.

    ``encoder_complexity`` is an explicit description-length proxy.  A direct
    product of two measured variables can use complexity 1, whereas a table
    containing one value per evaluation case should report its number of rows.
    """

    operator: MeasurementOperator = MeasurementOperator.CUSTOM
    inputs: tuple[str, ...] = ()
    output_domain: str = ""
    description: str = ""
    encoder_complexity: float = 0.0
    lookup_table: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "operator", MeasurementOperator(self.operator))
        object.__setattr__(self, "inputs", _string_tuple(self.inputs, "measurement.inputs"))
        object.__setattr__(
            self,
            "lookup_table",
            _numeric_mapping(self.lookup_table, "measurement.lookup_table"),
        )
        complexity = _finite_float(
            self.encoder_complexity, "measurement.encoder_complexity"
        )
        object.__setattr__(self, "encoder_complexity", complexity)
        if complexity < 0:
            raise ValueError("encoder_complexity must be non-negative")

    def evaluate(self, values: Mapping[str, float], *, case_id: str = "") -> float:
        """Evaluate the measurement for the supported deterministic operators.

        ``CUSTOM`` is intentionally not silently executed: custom encoders need
        an external implementation and therefore raise ``NotImplementedError``.
        """

        if self.operator is MeasurementOperator.LOOKUP:
            if not case_id:
                raise ValueError("lookup measurement requires a case_id")
            try:
                return float(self.lookup_table[case_id])
            except KeyError as exc:
                raise KeyError(f"case is absent from lookup table: {case_id}") from exc

        try:
            xs = [float(values[name]) for name in self.inputs]
        except KeyError as exc:
            raise KeyError(f"measurement input is unavailable: {exc.args[0]}") from exc

        if self.operator is MeasurementOperator.IDENTITY:
            if len(xs) != 1:
                raise ValueError("identity measurement requires exactly one input")
            return xs[0]
        if self.operator is MeasurementOperator.PRODUCT:
            if not xs:
                raise ValueError("product measurement requires at least one input")
            result = 1.0
            for value in xs:
                result *= value
            return result
        if self.operator is MeasurementOperator.SUM:
            if not xs:
                raise ValueError("sum measurement requires at least one input")
            return sum(xs)
        if self.operator is MeasurementOperator.DIFFERENCE:
            if len(xs) != 2:
                raise ValueError("difference measurement requires exactly two inputs")
            return xs[0] - xs[1]
        if self.operator is MeasurementOperator.THRESHOLD:
            if len(xs) != 1:
                raise ValueError("threshold measurement requires exactly one input")
            return float(xs[0] > 0.0)
        raise NotImplementedError("custom measurements require an external encoder")


@dataclass(frozen=True)
class UncertaintySpec:
    """Uncertainty model and a statement describing its calibration."""

    kind: UncertaintyKind = UncertaintyKind.DETERMINISTIC
    calibration_statement: str = ""
    parameters: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", UncertaintyKind(self.kind))
        object.__setattr__(
            self,
            "parameters",
            _numeric_mapping(self.parameters, "uncertainty.parameters"),
        )


@dataclass(frozen=True)
class CausalRoleSpec:
    """Claim about a concept's location in a causal graph."""

    variable: str = ""
    role: CausalRole = CausalRole.UNKNOWN
    parents: tuple[str, ...] = ()
    children: tuple[str, ...] = ()
    structural_equation: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", CausalRole(self.role))
        object.__setattr__(self, "parents", _string_tuple(self.parents, "causal_role.parents"))
        object.__setattr__(
            self, "children", _string_tuple(self.children, "causal_role.children")
        )


@dataclass(frozen=True)
class OutcomeModelSpec:
    """Executable linear outcome model using observables and the proposed concept."""

    target: str = "outcome"
    intercept: float = 0.0
    coefficients: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "intercept",
            _finite_float(self.intercept, "outcome_model.intercept"),
        )
        object.__setattr__(
            self,
            "coefficients",
            _numeric_mapping(self.coefficients, "outcome_model.coefficients"),
        )

    def evaluate(self, values: Mapping[str, float]) -> float:
        """Evaluate the declared outcome equation without executing arbitrary code."""

        result = self.intercept
        for variable, coefficient in self.coefficients.items():
            try:
                value = _finite_float(values[variable], f"outcome input {variable}")
            except KeyError as exc:
                raise KeyError(f"outcome input is unavailable: {variable}") from exc
            result += coefficient * value
        if not isfinite(result):
            raise ValueError("outcome model produced a non-finite value")
        return result


@dataclass(frozen=True)
class InterventionSpec:
    """How a proposed concept is connected to manipulable variables."""

    target: str = ""
    values: tuple[float, ...] = ()
    predicted_change: str = ""
    concept_is_directly_manipulable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.values, (list, tuple)):
            raise ValueError("intervention values must be a list")
        object.__setattr__(
            self,
            "values",
            tuple(
                _finite_float(value, "intervention value") for value in self.values
            ),
        )


@dataclass(frozen=True)
class ScopeSpec:
    """Environments, population, and assumptions where the concept applies."""

    environments: tuple[str, ...] = ()
    population: str = ""
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "environments", _string_tuple(self.environments, "scope.environments")
        )
        object.__setattr__(
            self, "assumptions", _string_tuple(self.assumptions, "scope.assumptions")
        )


@dataclass(frozen=True)
class PredictionSpec:
    """A numeric, case-addressable prediction that the harness can score."""

    case_id: str = ""
    target: str = "outcome"
    predicted_value: float = 0.0
    tolerance: float = 0.0
    rationale: str = ""

    def __post_init__(self) -> None:
        predicted_value = _finite_float(self.predicted_value, "prediction.predicted_value")
        tolerance = _finite_float(self.tolerance, "prediction.tolerance")
        object.__setattr__(self, "predicted_value", predicted_value)
        object.__setattr__(self, "tolerance", tolerance)
        if tolerance < 0:
            raise ValueError("prediction tolerance must be non-negative")


@dataclass(frozen=True)
class FalsifierSpec:
    """A pre-registered observation and rule that could reject the concept."""

    statement: str = ""
    case_ids: tuple[str, ...] = ()
    rejection_rule: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_ids", _string_tuple(self.case_ids, "falsifier.case_ids"))


@dataclass(frozen=True)
class ConceptProposal:
    """Complete typed proposal for a candidate causal concept.

    Defaults make it possible to represent an intentionally empty baseline.
    Use :meth:`executability_issues` to distinguish a merely serializable
    proposal from one that a scientist could actually test.
    """

    name: str = ""
    domain: str = ""
    concept_type: str = ""
    measurement: MeasurementSpec = field(default_factory=MeasurementSpec)
    uncertainty: UncertaintySpec = field(default_factory=UncertaintySpec)
    causal_role: CausalRoleSpec = field(default_factory=CausalRoleSpec)
    outcome_model: OutcomeModelSpec = field(default_factory=OutcomeModelSpec)
    intervention_map: tuple[InterventionSpec, ...] = ()
    scope: ScopeSpec = field(default_factory=ScopeSpec)
    predictions: tuple[PredictionSpec, ...] = ()
    falsifiers: tuple[FalsifierSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "intervention_map", tuple(self.intervention_map))
        object.__setattr__(self, "predictions", tuple(self.predictions))
        object.__setattr__(self, "falsifiers", tuple(self.falsifiers))

    def executability_issues(self) -> tuple[str, ...]:
        """Return concrete missing or non-executable components."""

        issues: list[str] = []
        if not self.name.strip():
            issues.append("missing concept name")
        if not self.domain.strip() or not self.concept_type.strip():
            issues.append("missing domain/type")
        if not self.measurement.inputs:
            issues.append("measurement has no inputs")
        if not self.measurement.output_domain.strip():
            issues.append("measurement has no output domain")
        if self.measurement.operator is MeasurementOperator.CUSTOM:
            issues.append("custom measurement has no bundled implementation")
        if not self.uncertainty.calibration_statement.strip():
            issues.append("uncertainty model is not calibrated or justified")
        if self.causal_role.role is CausalRole.UNKNOWN:
            issues.append("causal role is unknown")
        if not self.causal_role.structural_equation.strip():
            issues.append("causal role has no structural equation")
        if not self.causal_role.variable.strip():
            issues.append("causal role has no concept variable")
        if not self.outcome_model.target.strip() or not self.outcome_model.coefficients:
            issues.append("missing executable outcome model")
        elif self.causal_role.variable not in self.outcome_model.coefficients:
            issues.append("outcome model does not use the proposed concept")
        if not self.intervention_map or any(
            not item.target.strip()
            or not item.predicted_change.strip()
            or (item.concept_is_directly_manipulable and not item.values)
            for item in self.intervention_map
        ):
            issues.append("missing intervention map")
        if not self.scope.environments or not self.scope.population.strip():
            issues.append("scope is incomplete")
        if not self.predictions:
            issues.append("missing testable predictions")
        if not self.falsifiers or any(
            not falsifier.statement.strip()
            or not falsifier.case_ids
            or not falsifier.rejection_rule.strip()
            for falsifier in self.falsifiers
        ):
            issues.append("missing executable falsifier")
        return tuple(issues)

    @property
    def is_executable(self) -> bool:
        """Whether every required executable component is present."""

        return not self.executability_issues()


def proposal_to_dict(proposal: ConceptProposal) -> dict[str, Any]:
    """Convert a proposal to a JSON-compatible dictionary."""

    return {
        "name": proposal.name,
        "domain": proposal.domain,
        "concept_type": proposal.concept_type,
        "measurement": {
            "operator": proposal.measurement.operator.value,
            "inputs": list(proposal.measurement.inputs),
            "output_domain": proposal.measurement.output_domain,
            "description": proposal.measurement.description,
            "encoder_complexity": proposal.measurement.encoder_complexity,
            "lookup_table": dict(proposal.measurement.lookup_table),
        },
        "uncertainty": {
            "kind": proposal.uncertainty.kind.value,
            "calibration_statement": proposal.uncertainty.calibration_statement,
            "parameters": dict(proposal.uncertainty.parameters),
        },
        "causal_role": {
            "variable": proposal.causal_role.variable,
            "role": proposal.causal_role.role.value,
            "parents": list(proposal.causal_role.parents),
            "children": list(proposal.causal_role.children),
            "structural_equation": proposal.causal_role.structural_equation,
        },
        "outcome_model": {
            "target": proposal.outcome_model.target,
            "intercept": proposal.outcome_model.intercept,
            "coefficients": dict(proposal.outcome_model.coefficients),
        },
        "intervention_map": [
            {
                "target": item.target,
                "values": list(item.values),
                "predicted_change": item.predicted_change,
                "concept_is_directly_manipulable": item.concept_is_directly_manipulable,
            }
            for item in proposal.intervention_map
        ],
        "scope": {
            "environments": list(proposal.scope.environments),
            "population": proposal.scope.population,
            "assumptions": list(proposal.scope.assumptions),
        },
        "predictions": [
            {
                "case_id": item.case_id,
                "target": item.target,
                "predicted_value": item.predicted_value,
                "tolerance": item.tolerance,
                "rationale": item.rationale,
            }
            for item in proposal.predictions
        ],
        "falsifiers": [
            {
                "statement": item.statement,
                "case_ids": list(item.case_ids),
                "rejection_rule": item.rejection_rule,
            }
            for item in proposal.falsifiers
        ],
    }


def _mapping_field(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a JSON object")
    return value


def _list_field(data: Mapping[str, Any], key: str) -> list[Any] | tuple[Any, ...]:
    value = data.get(key, [])
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} must be a JSON list")
    return value


def _text(data: Mapping[str, Any], key: str, default: str = "") -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _number(data: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    return _finite_float(data.get(key, default), key)


def _boolean(data: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _objects(data: Mapping[str, Any], key: str) -> tuple[Mapping[str, Any], ...]:
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(_list_field(data, key)):
        if not isinstance(item, Mapping):
            raise ValueError(f"{key}[{index}] must be a JSON object")
        result.append(item)
    return tuple(result)


def proposal_from_dict(data: Mapping[str, Any]) -> ConceptProposal:
    """Strictly parse a proposal while leaving omitted fields visibly incomplete."""

    if not isinstance(data, Mapping):
        raise ValueError("proposal must be a JSON object")
    measurement = _mapping_field(data, "measurement")
    uncertainty = _mapping_field(data, "uncertainty")
    causal_role = _mapping_field(data, "causal_role")
    outcome_model = _mapping_field(data, "outcome_model")
    scope = _mapping_field(data, "scope")
    concept_type_key = "concept_type" if "concept_type" in data else "type"
    return ConceptProposal(
        name=_text(data, "name"),
        domain=_text(data, "domain"),
        concept_type=_text(data, concept_type_key),
        measurement=MeasurementSpec(
            operator=measurement.get("operator", MeasurementOperator.CUSTOM.value),
            inputs=_string_tuple(measurement.get("inputs", []), "measurement.inputs"),
            output_domain=_text(measurement, "output_domain"),
            description=_text(measurement, "description"),
            encoder_complexity=_number(measurement, "encoder_complexity"),
            lookup_table=_numeric_mapping(
                measurement.get("lookup_table", {}), "measurement.lookup_table"
            ),
        ),
        uncertainty=UncertaintySpec(
            kind=uncertainty.get("kind", UncertaintyKind.DETERMINISTIC.value),
            calibration_statement=_text(uncertainty, "calibration_statement"),
            parameters=_numeric_mapping(
                uncertainty.get("parameters", {}), "uncertainty.parameters"
            ),
        ),
        causal_role=CausalRoleSpec(
            variable=_text(causal_role, "variable"),
            role=causal_role.get("role", CausalRole.UNKNOWN.value),
            parents=_string_tuple(causal_role.get("parents", []), "causal_role.parents"),
            children=_string_tuple(
                causal_role.get("children", []), "causal_role.children"
            ),
            structural_equation=_text(causal_role, "structural_equation"),
        ),
        outcome_model=OutcomeModelSpec(
            target=_text(outcome_model, "target", "outcome"),
            intercept=_number(outcome_model, "intercept"),
            coefficients=_numeric_mapping(
                outcome_model.get("coefficients", {}), "outcome_model.coefficients"
            ),
        ),
        intervention_map=tuple(
            InterventionSpec(
                target=_text(item, "target"),
                values=tuple(
                    _finite_float(value, "intervention value")
                    for value in _list_field(item, "values")
                ),
                predicted_change=_text(item, "predicted_change"),
                concept_is_directly_manipulable=_boolean(
                    item, "concept_is_directly_manipulable"
                ),
            )
            for item in _objects(data, "intervention_map")
        ),
        scope=ScopeSpec(
            environments=_string_tuple(scope.get("environments", []), "scope.environments"),
            population=_text(scope, "population"),
            assumptions=_string_tuple(scope.get("assumptions", []), "scope.assumptions"),
        ),
        predictions=tuple(
            PredictionSpec(
                case_id=_text(item, "case_id"),
                target=_text(item, "target", "outcome"),
                predicted_value=_number(item, "predicted_value"),
                tolerance=_number(item, "tolerance"),
                rationale=_text(item, "rationale"),
            )
            for item in _objects(data, "predictions")
        ),
        falsifiers=tuple(
            FalsifierSpec(
                statement=_text(item, "statement"),
                case_ids=_string_tuple(item.get("case_ids", []), "falsifier.case_ids"),
                rejection_rule=_text(item, "rejection_rule"),
            )
            for item in _objects(data, "falsifiers")
        ),
    )

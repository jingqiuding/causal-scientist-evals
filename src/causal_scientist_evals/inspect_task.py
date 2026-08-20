"""Optional Inspect AI adapter for reproducible model evaluation runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from importlib import resources
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, json_dataset
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import Solver, TaskState, chain, generate, solver, system_message

from .prompts import SYSTEM_PROMPT
from .schema import ConceptProposal, proposal_from_dict
from .scoring import benchmark_validation_issues, score_proposal
from .simulator import CausalBenchmark, build_default_benchmark


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Map a JSONL record to a sample while keeping sealed labels out of metadata."""

    required = {"id", "input", "sealed_target", "domain", "difficulty"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"dataset record is missing fields: {sorted(missing)}")
    return Sample(
        id=str(record["id"]),
        input=str(record["input"]),
        target=json.dumps(record["sealed_target"], sort_keys=True),
        metadata={
            "domain": str(record["domain"]),
            "difficulty": str(record["difficulty"]),
            "schema_version": str(record.get("schema_version", "unknown")),
        },
    )


def load_cases(dataset_path: str | Path | None = None) -> Dataset:
    """Load an external JSONL file or the bundled deterministic development case."""

    if dataset_path is not None:
        return json_dataset(str(dataset_path), sample_fields=record_to_sample)
    resource = resources.files("causal_scientist_evals").joinpath("data/dev.jsonl")
    with resources.as_file(resource) as path:
        return json_dataset(str(path), sample_fields=record_to_sample)


def _benchmark_from_target(target_data: Mapping[str, Any]) -> CausalBenchmark:
    """Bind scorer-only expected values to the versioned benchmark structure."""

    benchmark = build_default_benchmark()
    if target_data.get("benchmark_id") != "treatment-gate-v1":
        raise ValueError("unsupported benchmark_id in sealed target")
    raw_cases = target_data.get("scoring_cases")
    if not isinstance(raw_cases, list):
        raise ValueError("sealed target must contain a scoring_cases list")

    by_id: dict[str, Mapping[str, Any]] = {}
    for item in raw_cases:
        if not isinstance(item, dict) or not isinstance(item.get("case_id"), str):
            raise ValueError("every scoring case must be an object with a case_id")
        if item["case_id"] in by_id:
            raise ValueError(f"duplicate scoring case: {item['case_id']}")
        by_id[item["case_id"]] = item

    expected_ids = {case.case_id for case in benchmark.cases}
    if set(by_id) != expected_ids:
        missing = sorted(expected_ids - set(by_id))
        extra = sorted(set(by_id) - expected_ids)
        raise ValueError(
            f"sealed case IDs do not match benchmark; missing={missing}, extra={extra}"
        )

    cases = []
    for case in benchmark.cases:
        item = by_id[case.case_id]
        if item.get("split") != case.split.value or item.get("environment") != case.environment:
            raise ValueError(f"sealed metadata mismatch for {case.case_id}")
        cases.append(
            replace(
                case,
                expected_value=float(item["expected_value"]),
                tolerance=float(item.get("tolerance", case.tolerance)),
            )
        )
    return replace(benchmark, cases=tuple(cases))


def _zero_values() -> dict[str, float]:
    """Return a stable zero vector matching every registered component."""

    empty = score_proposal(ConceptProposal(), build_default_benchmark()).to_dict()
    values = {key: 0.0 for key in empty}
    values["schema_valid"] = 0.0
    return values


def parse_and_score(completion: str, target_text: str) -> tuple[dict[str, float], str]:
    """Pure JSON parsing and deterministic scoring used by the Inspect adapter."""

    try:
        def reject_constant(value: str) -> None:
            raise ValueError(f"non-standard JSON constant is not allowed: {value}")

        proposal_data = json.loads(completion, parse_constant=reject_constant)
        if not isinstance(proposal_data, dict):
            raise ValueError("completion must be one JSON object")
        proposal = proposal_from_dict(proposal_data)

        target_data = json.loads(target_text, parse_constant=reject_constant)
        if not isinstance(target_data, dict):
            raise ValueError("sealed target must be one JSON object")
        benchmark = _benchmark_from_target(target_data)
        report = score_proposal(proposal, benchmark)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return _zero_values(), f"Invalid proposal or target: {exc}"

    values = {
        key: float(value) for key, value in report.to_dict().items()
    }
    issues = benchmark_validation_issues(proposal, benchmark)
    values["schema_valid"] = float(report.executability_gate)
    if report.executability_gate:
        explanation = "Executable concept card scored on sealed causal cases."
    else:
        explanation = "Invalid or incomplete concept card: " + "; ".join(issues)
    return values, explanation


@solver
def direct_proposal() -> Solver:
    """Generate a single structured proposal with no tools or target feedback."""

    return chain(system_message(SYSTEM_PROMPT), generate())


@scorer(metrics={"*": [mean(), stderr()]})
def causal_utility() -> Scorer:
    """Score parse validity, executability, sealed prediction, and anti-gaming terms."""

    async def score(state: TaskState, target: Target) -> Score:
        values, explanation = parse_and_score(state.output.completion, target.text)
        return Score(
            value=values,
            answer=state.output.completion,
            explanation=explanation,
        )

    return score


@task
def sealed_interventions(dataset_path: str | None = None) -> Task:
    """Evaluate an agent's executable concept on scorer-only causal outcomes."""

    return Task(
        dataset=load_cases(dataset_path),
        solver=direct_proposal(),
        scorer=causal_utility(),
        version="0.1.0",
    )


# Backward-compatible file-task name used by early project notes.
causal_concept_discovery = sealed_interventions

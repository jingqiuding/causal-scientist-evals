"""Leakage-resistant JSONL records for the causal concept benchmark."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .prompts import OUTPUT_CONTRACT
from .simulator import CausalBenchmark, Split, build_default_benchmark

SCHEMA_VERSION = "0.1.0"
BENCHMARK_ID = "treatment-gate-v1"


def _number(value: float) -> int | float:
    """Use compact integers in generated artifacts when values are integral."""

    return int(value) if float(value).is_integer() else float(value)


def _format_table(rows: Iterable[Iterable[object]], header: Iterable[str]) -> str:
    values = [tuple(str(cell) for cell in header)]
    values.extend(tuple(str(cell) for cell in row) for row in rows)
    widths = [max(len(row[index]) for row in values) for index in range(len(values[0]))]
    return "\n".join(
        " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        for row in values
    )


def _visible_prompt(benchmark: CausalBenchmark) -> str:
    visible = (
        benchmark.cases_for(Split.OBSERVATIONAL)
        + benchmark.cases_for(Split.VISIBLE_INTERVENTION)
    )
    prospective = tuple(
        case
        for case in benchmark.cases
        if case.split not in {Split.OBSERVATIONAL, Split.VISIBLE_INTERVENTION}
    )

    observed_table = _format_table(
        (
            (
                case.case_id,
                case.split.value,
                case.environment,
                _number(case.inputs["context"]),
                _number(case.inputs["gate"]),
                _number(case.inputs["treatment"]),
                json.dumps(case.intervention, sort_keys=True, separators=(",", ":")),
                _number(case.expected_value),
            )
            for case in visible
        ),
        (
            "case_id",
            "split",
            "environment",
            "context",
            "gate",
            "treatment",
            "intervention",
            "outcome",
        ),
    )
    prospective_table = _format_table(
        (
            (
                case.case_id,
                case.split.value,
                case.environment,
                _number(case.inputs["context"]),
                _number(case.inputs["gate"]),
                _number(case.inputs["treatment"]),
                json.dumps(case.intervention, sort_keys=True, separators=(",", ":")),
            )
            for case in prospective
        ),
        (
            "case_id",
            "split",
            "environment",
            "context",
            "gate",
            "treatment",
            "intervention",
        ),
    )
    all_case_ids = ", ".join(case.case_id for case in benchmark.cases)
    return f"""Scientific setting
------------------
Each unit has a measured context, a binary gate, a binary administered treatment,
and a numeric outcome. In observational rows from the aligned-lab policy, gate and
treatment are perfectly correlated, so multiple accounts fit those rows. Disclosed
calibration interventions break that alias. Propose a compact derived concept and
commit to prospective predictions.

Visible observations and calibration interventions
-------------------------------------------------
{observed_table}

Prospective cases (outcomes sealed)
-----------------------------------
{prospective_table}

Predict the outcome for every case, including visible cases. Required case IDs:
{all_case_ids}

{OUTPUT_CONTRACT}"""


def build_record(benchmark: CausalBenchmark | None = None) -> dict[str, Any]:
    """Build one Inspect-compatible record without target leakage in input/metadata."""

    benchmark = benchmark or build_default_benchmark()
    scoring_cases = [
        {
            "case_id": case.case_id,
            "split": case.split.value,
            "environment": case.environment,
            "expected_value": _number(case.expected_value),
            "tolerance": case.tolerance,
        }
        for case in benchmark.cases
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "id": BENCHMARK_ID,
        "domain": "synthetic structural causal model",
        "difficulty": "prototype",
        "input": _visible_prompt(benchmark),
        # This field is serialized beside the sample for auditability, but the
        # Inspect adapter places it only in Sample.target, never in the prompt or metadata.
        "sealed_target": {
            "benchmark_id": BENCHMARK_ID,
            "scoring_cases": scoring_cases,
        },
    }


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    """Write deterministic UTF-8 JSONL, creating parent directories as needed."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    return output


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load records with an actionable line number on malformed input."""

    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            records.append(record)
    return records


def generate_default(path: str | Path) -> Path:
    """Generate the versioned default benchmark artifact."""

    return write_jsonl(path, [build_record()])

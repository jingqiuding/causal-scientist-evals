"""Deterministic reference fixtures and auditable summary generation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .schema import ConceptProposal
from .scoring import score_proposal
from .simulator import (
    CausalBenchmark,
    build_default_benchmark,
    empty_proposal,
    lookup_proposal,
    oracle_proposal,
    renaming_proposal,
)

ProposalFactory = Callable[[CausalBenchmark | None], ConceptProposal]

BASELINES: dict[str, ProposalFactory] = {
    "empty": lambda benchmark=None: empty_proposal(),
    "treatment_renaming": renaming_proposal,
    "lookup_table": lookup_proposal,
    "compact_causal_oracle": oracle_proposal,
}


def evaluate_baselines(benchmark: CausalBenchmark | None = None) -> dict[str, Any]:
    """Return component scores for declared reference proposals in stable order."""

    benchmark = benchmark or build_default_benchmark()
    results: dict[str, Any] = {}
    for name, factory in BASELINES.items():
        proposal = factory(benchmark)
        results[name] = {
            "is_executable": proposal.is_executable,
            "score": score_proposal(proposal, benchmark).to_dict(),
        }
    return {
        "benchmark_id": "treatment-gate-v1",
        "schema_version": "0.1.0",
        "result_kind": "deterministic_reference_fixtures",
        "disclaimer": (
            "Scorer fixtures only; not matched baselines or measured model results."
        ),
        "baselines": results,
    }

"""Deterministic standard-library tests for the core benchmark."""

import unittest
from dataclasses import replace

from causal_scientist_evals.schema import (
    ConceptProposal,
    MeasurementOperator,
    proposal_from_dict,
    proposal_to_dict,
)
from causal_scientist_evals.scoring import score_proposal
from causal_scientist_evals.simulator import (
    Split,
    StructuralCausalEnvironment,
    build_default_benchmark,
    empty_proposal,
    lookup_proposal,
    oracle_proposal,
    renaming_proposal,
)


class CoreBenchmarkTests(unittest.TestCase):
    """Exercise evidence separation, serialization, and anti-gaming scores."""

    def test_scm_is_deterministic_and_interventions_break_alias(self) -> None:
        environment = StructuralCausalEnvironment(name="test", policy="aligned")
        self.assertEqual(environment.simulate(1), environment.simulate(1))

        # Under the observational policy treatment == gate, so treatment and
        # their product coincide.  A crossed intervention reveals the alias.
        observational = environment.simulate(1)
        self.assertEqual(
            observational["treatment"], observational["activated_signal"]
        )
        crossed = environment.simulate(1, {"treatment": 1.0, "gate": 0.0})
        self.assertEqual(crossed["treatment"], 1.0)
        self.assertEqual(crossed["activated_signal"], 0.0)

    def test_default_benchmark_has_all_evaluation_splits_without_leakage(self) -> None:
        first = build_default_benchmark()
        second = build_default_benchmark()
        self.assertEqual(first, second)
        self.assertEqual(len(first.cases_for(Split.OBSERVATIONAL)), 8)
        self.assertEqual(len(first.cases_for(Split.VISIBLE_INTERVENTION)), 2)
        self.assertEqual(len(first.cases_for(Split.SEALED_INTERVENTION)), 4)
        self.assertEqual(len(first.cases_for(Split.COUNTERFACTUAL)), 4)
        self.assertEqual(len(first.cases_for(Split.TRANSPORT)), 8)
        self.assertEqual(
            {
                (case.inputs["treatment"], case.inputs["gate"])
                for case in first.cases_for(Split.SEALED_INTERVENTION)
            },
            {(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)},
        )
        self.assertTrue(
            all("activated_signal" not in case.inputs for case in first.cases)
        )

    def test_oracle_schema_is_executable_and_round_trips(self) -> None:
        proposal = oracle_proposal(build_default_benchmark())
        self.assertTrue(proposal.is_executable)
        self.assertIs(proposal.measurement.operator, MeasurementOperator.PRODUCT)
        restored = proposal_from_dict(proposal_to_dict(proposal))
        self.assertEqual(restored, proposal)

    def test_score_orders_causal_oracle_above_gaming_baselines(self) -> None:
        benchmark = build_default_benchmark()
        oracle = score_proposal(oracle_proposal(benchmark), benchmark)
        renamed = score_proposal(renaming_proposal(benchmark), benchmark)
        lookup = score_proposal(lookup_proposal(benchmark), benchmark)
        empty = score_proposal(empty_proposal(), benchmark)

        self.assertAlmostEqual(oracle.total, 1.0)
        self.assertAlmostEqual(oracle.interventional_prediction, 1.0)
        self.assertAlmostEqual(oracle.counterfactual_prediction, 1.0)
        self.assertAlmostEqual(oracle.transport_prediction, 1.0)
        self.assertGreater(oracle.total, renamed.total)
        self.assertGreater(renamed.total, empty.total)
        self.assertGreater(oracle.total, lookup.total)
        self.assertGreater(lookup.total, empty.total)
        self.assertAlmostEqual(renamed.renaming_penalty, 1.0)
        self.assertAlmostEqual(lookup.lookup_penalty, 1.0)
        self.assertGreater(lookup.complexity_penalty, 0.5)
        lookup_measurement = lookup_proposal(benchmark).measurement
        self.assertEqual(
            lookup_measurement.evaluate({}, case_id=benchmark.cases[0].case_id),
            0.0,
        )
        self.assertLess(renamed.interventional_prediction, 1.0)
        self.assertLess(renamed.counterfactual_prediction, 1.0)
        self.assertLess(renamed.transport_prediction, 1.0)

    def test_missing_sealed_declarations_fail_the_headline_gate(self) -> None:
        benchmark = build_default_benchmark()
        oracle = oracle_proposal(benchmark)
        observational_only = replace(
            oracle,
            predictions=tuple(
                prediction
                for prediction in oracle.predictions
                if prediction.case_id.startswith(Split.OBSERVATIONAL.value)
            ),
        )
        score = score_proposal(observational_only, benchmark)
        self.assertGreater(score.visible_fit_prediction, 0.0)
        # Component scores execute the model, while the headline gate enforces
        # complete prospective commitments.
        self.assertAlmostEqual(score.interventional_prediction, 1.0)
        self.assertFalse(score.executability_gate)
        self.assertAlmostEqual(score.total, 0.0)

    def test_empty_proposal_exposes_actionable_executability_issues(self) -> None:
        proposal = ConceptProposal()
        issues = proposal.executability_issues()
        self.assertFalse(proposal.is_executable)
        self.assertIn("missing concept name", issues)
        self.assertIn("missing intervention map", issues)
        self.assertIn("missing executable falsifier", issues)

    def test_executability_gate_zeros_incomplete_headline_score(self) -> None:
        benchmark = build_default_benchmark()
        incomplete = replace(oracle_proposal(benchmark), falsifiers=())
        score = score_proposal(incomplete, benchmark)
        self.assertFalse(score.executability_gate)
        self.assertGreater(score.interventional_prediction, 0.0)
        self.assertAlmostEqual(score.total, 0.0)


if __name__ == "__main__":
    unittest.main()

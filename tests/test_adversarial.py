"""Regression tests for scorer-gaming and malformed-output failures."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace

from causal_scientist_evals.dataset import build_record
from causal_scientist_evals.inspect_task import parse_and_score
from causal_scientist_evals.schema import (
    CausalRole,
    OutcomeModelSpec,
    PredictionSpec,
    proposal_to_dict,
)
from causal_scientist_evals.scoring import benchmark_validation_issues, score_proposal
from causal_scientist_evals.simulator import build_default_benchmark, oracle_proposal


class AdversarialScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.benchmark = build_default_benchmark()
        self.oracle = oracle_proposal(self.benchmark)
        self.target = json.dumps(build_record(self.benchmark)["sealed_target"])

    def test_candidate_tolerance_cannot_relax_evaluator_tolerance(self) -> None:
        wrong_declarations = replace(
            self.oracle,
            predictions=tuple(
                replace(
                    prediction,
                    predicted_value=prediction.predicted_value + 1000.0,
                    tolerance=1e9,
                )
                for prediction in self.oracle.predictions
            ),
        )
        score = score_proposal(wrong_declarations, self.benchmark)
        self.assertFalse(score.executability_gate)
        self.assertEqual(score.total, 0.0)
        self.assertTrue(
            any(
                "disagree with executable model" in issue
                for issue in benchmark_validation_issues(
                    wrong_declarations, self.benchmark
                )
            )
        )

        zero_model = replace(
            self.oracle,
            outcome_model=OutcomeModelSpec(
                target="outcome",
                coefficients={"activated_signal": 0.0, "context": 0.0},
            ),
            predictions=tuple(
                replace(prediction, predicted_value=0.0, tolerance=1e9)
                for prediction in self.oracle.predictions
            ),
        )
        zero_score = score_proposal(zero_model, self.benchmark)
        self.assertTrue(zero_score.executability_gate)
        self.assertLess(zero_score.interventional_prediction, 1.0)
        self.assertLess(zero_score.total, 1.0)

    def test_non_finite_numbers_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            PredictionSpec(predicted_value=float("nan"))

        data = proposal_to_dict(self.oracle)
        data["predictions"][0]["predicted_value"] = float("nan")
        values, explanation = parse_and_score(json.dumps(data), self.target)
        self.assertEqual(values["total"], 0.0)
        self.assertIn("non-standard JSON constant", explanation)

        data = proposal_to_dict(self.oracle)
        data["outcome_model"]["intercept"] = float("inf")
        values, _ = parse_and_score(json.dumps(data), self.target)
        self.assertEqual(values["schema_valid"], 0.0)

    def test_wrong_nested_types_return_zero_instead_of_crashing(self) -> None:
        for malformed in (
            {"measurement": None},
            {"scope": None},
            {"predictions": [1]},
            {"name": None},
        ):
            with self.subTest(malformed=malformed):
                values, explanation = parse_and_score(
                    json.dumps(malformed), self.target
                )
                self.assertEqual(values["schema_valid"], 0.0)
                self.assertEqual(values["total"], 0.0)
                self.assertIn("Invalid", explanation)

    def test_one_validator_controls_schema_and_headline_gate(self) -> None:
        unknown_role = replace(
            self.oracle,
            causal_role=replace(self.oracle.causal_role, role=CausalRole.UNKNOWN),
        )
        self.assertFalse(unknown_role.is_executable)
        score = score_proposal(unknown_role, self.benchmark)
        self.assertFalse(score.executability_gate)
        self.assertEqual(score.total, 0.0)

        data = proposal_to_dict(unknown_role)
        values, _ = parse_and_score(json.dumps(data), self.target)
        self.assertEqual(values["schema_valid"], 0.0)
        self.assertEqual(values["total"], 0.0)

    def test_required_prediction_coverage_is_enforced(self) -> None:
        single = replace(
            self.oracle,
            predictions=(
                replace(self.oracle.predictions[0], case_id="not-a-benchmark-case"),
            ),
        )
        score = score_proposal(single, self.benchmark)
        self.assertFalse(score.executability_gate)
        self.assertEqual(score.total, 0.0)


if __name__ == "__main__":
    unittest.main()

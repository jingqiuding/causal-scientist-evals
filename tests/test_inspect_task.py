"""Tests for the optional Inspect dataset, scorer, and registered task."""

from __future__ import annotations

import json
import os
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

try:
    from inspect_ai import eval as inspect_eval

    INSPECT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in minimal installs
    INSPECT_AVAILABLE = False

from causal_scientist_evals.dataset import build_record
from causal_scientist_evals.schema import proposal_to_dict
from causal_scientist_evals.simulator import build_default_benchmark, oracle_proposal


@unittest.skipUnless(INSPECT_AVAILABLE, "install the 'inspect' optional dependency")
class InspectAdapterTests(unittest.TestCase):
    def test_sample_keeps_sealed_outcomes_out_of_metadata(self) -> None:
        from causal_scientist_evals.inspect_task import record_to_sample

        record = build_record()
        sample = record_to_sample(record)
        self.assertNotIn("sealed_target", sample.metadata)
        self.assertNotIn("expected_value", str(sample.input))
        self.assertIn("scoring_cases", sample.target)

    def test_pure_scorer_accepts_oracle_and_rejects_invalid_json(self) -> None:
        from causal_scientist_evals.inspect_task import parse_and_score

        record = build_record()
        completion = json.dumps(
            proposal_to_dict(oracle_proposal(build_default_benchmark()))
        )
        target = json.dumps(record["sealed_target"])
        values, explanation = parse_and_score(completion, target)
        self.assertEqual(values["schema_valid"], 1.0)
        self.assertEqual(values["total"], 1.0)
        self.assertIn("Executable", explanation)

        invalid, invalid_explanation = parse_and_score("not json", target)
        self.assertEqual(invalid["schema_valid"], 0.0)
        self.assertEqual(invalid["total"], 0.0)
        self.assertIn("Invalid", invalid_explanation)

    def test_mock_model_smoke_run_succeeds(self) -> None:
        from causal_scientist_evals.inspect_task import sealed_interventions

        with TemporaryDirectory() as directory:
            root = Path(directory)
            xdg_dirs = {
                "XDG_DATA_HOME": str(root / "data"),
                "XDG_CACHE_HOME": str(root / "cache"),
                "XDG_CONFIG_HOME": str(root / "config"),
            }
            with patch.dict(os.environ, xdg_dirs), warnings.catch_warnings():
                # Inspect 0.3.259/anyio emits this during mock-model teardown.
                warnings.filterwarnings(
                    "ignore",
                    message=r"Unclosed <MemoryObjectReceiveStream.*",
                    category=ResourceWarning,
                )
                logs = inspect_eval(
                    sealed_interventions(),
                    model="mockllm/model",
                    limit=1,
                    display="none",
                    log_dir=str(root / "logs"),
                )
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0].status, "success")


if __name__ == "__main__":
    unittest.main()

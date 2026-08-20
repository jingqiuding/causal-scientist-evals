"""Tests for deterministic generation, evidence separation, and the CLI."""

from __future__ import annotations

import json
import unittest
from importlib import resources
from pathlib import Path
from tempfile import TemporaryDirectory

from causal_scientist_evals.cli import main
from causal_scientist_evals.dataset import build_record, read_jsonl, write_jsonl


class DatasetAndCliTests(unittest.TestCase):
    def test_record_is_deterministic_and_does_not_name_hidden_concept(self) -> None:
        first = build_record()
        self.assertEqual(first, build_record())
        self.assertNotIn("activated_signal", first["input"])
        self.assertNotIn("expected_value", first["input"])
        self.assertIn("sealed_target", first)
        self.assertEqual(first["id"], "treatment-gate-v1")

    def test_jsonl_round_trip(self) -> None:
        with TemporaryDirectory() as directory:
            output = write_jsonl(Path(directory) / "nested" / "cases.jsonl", [build_record()])
            self.assertEqual(read_jsonl(output), [build_record()])

    def test_bundled_development_record_matches_generator(self) -> None:
        resource = resources.files("causal_scientist_evals").joinpath("data/dev.jsonl")
        with resources.as_file(resource) as path:
            self.assertEqual(read_jsonl(path), [build_record()])

    def test_cli_generates_dataset_and_reference_summary(self) -> None:
        with TemporaryDirectory() as directory:
            data_path = Path(directory) / "data.jsonl"
            result_path = Path(directory) / "baselines.json"
            self.assertEqual(main(["generate", "--output", str(data_path)]), 0)
            self.assertEqual(main(["baselines", "--output", str(result_path)]), 0)
            self.assertEqual(len(read_jsonl(data_path)), 1)
            summary = json.loads(result_path.read_text(encoding="utf-8"))
            totals = {
                name: result["score"]["total"]
                for name, result in summary["baselines"].items()
            }
            self.assertEqual(totals["compact_causal_oracle"], 1.0)
            self.assertGreater(totals["compact_causal_oracle"], totals["lookup_table"])
            self.assertGreater(totals["lookup_table"], totals["empty"])


if __name__ == "__main__":
    unittest.main()

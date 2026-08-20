"""Command-line interface for artifact generation and reference baselines."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .baselines import evaluate_baselines
from .dataset import generate_default


def _write_json(path: str | Path, value: object) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser separately for deterministic tests."""

    parser = argparse.ArgumentParser(
        prog="causal-eval",
        description="Build and inspect the causal scientist evaluation prototype.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="generate the benchmark JSONL")
    generate.add_argument("--output", default="data/benchmark.jsonl")

    baselines = commands.add_parser(
        "baselines", help="score deterministic reference proposals"
    )
    baselines.add_argument("--output", default="results/baseline_summary.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a CLI command and return a process status."""

    args = build_parser().parse_args(argv)
    if args.command == "generate":
        output = generate_default(args.output)
    else:
        output = _write_json(args.output, evaluate_baselines())
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

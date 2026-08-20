"""Inspect AI registry entry point."""

# Import decorated components for their registration side effects.
from .inspect_task import causal_utility, direct_proposal, sealed_interventions

__all__ = ["causal_utility", "direct_proposal", "sealed_interventions"]

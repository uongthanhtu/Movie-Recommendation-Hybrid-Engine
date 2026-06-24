"""
Dataset Factory -- Factory Method entry point for the Grand Unified Benchmark Arena's
dataset loading system.

DatasetFactory.create(name) checks the explicit-trust registry first, then the
implicit-trust (Mode B / ablation study) registry, and returns a configured loader of
the matching type. Adding a new dataset means registering a new config in the
appropriate registry -- not modifying this file or writing a new loader class, for
either Mode A or Mode B (Open for extension, Closed for modification). Registry
membership alone is the dispatch signal; neither config type carries a "mode" field
for this purpose.
"""
from __future__ import annotations

from pipeline.data_loaders.base_loader import BaseDatasetLoader
from pipeline.data_loaders.dataset_configs import DATASET_REGISTRY, IMPLICIT_DATASET_REGISTRY
from pipeline.data_loaders.explicit_trust_loader import ExplicitTrustLoader
from pipeline.data_loaders.implicit_trust_loader import ImplicitTrustLoader


class DatasetFactory:
    """Factory Method for constructing dataset loaders by name."""

    @staticmethod
    def create(name: str) -> BaseDatasetLoader:
        """
        Look up `name` (case-insensitive) in the explicit-trust registry first, then
        the implicit-trust (Mode B) registry, and return a configured loader of the
        matching type, ready to call .load() on.

        Raises:
            ValueError: if `name` is not registered in either registry.
        """
        key = name.lower()

        if key in DATASET_REGISTRY:
            return ExplicitTrustLoader(DATASET_REGISTRY[key])

        if key in IMPLICIT_DATASET_REGISTRY:
            return ImplicitTrustLoader(IMPLICIT_DATASET_REGISTRY[key])

        available = ", ".join(sorted(list(DATASET_REGISTRY.keys()) + list(IMPLICIT_DATASET_REGISTRY.keys())))
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")

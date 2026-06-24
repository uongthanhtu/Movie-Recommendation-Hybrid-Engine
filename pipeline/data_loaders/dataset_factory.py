"""
Dataset Factory -- Factory Method entry point for the Grand Unified Benchmark Arena's
dataset loading system.

DatasetFactory.create(name) looks up a DatasetConfig in dataset_configs.DATASET_REGISTRY
and returns a configured BaseDatasetLoader. Adding a new dataset means registering a
new DatasetConfig, not modifying this file or writing a new loader class (Open for
extension, Closed for modification).
"""
from __future__ import annotations

from pipeline.data_loaders.base_loader import BaseDatasetLoader
from pipeline.data_loaders.dataset_configs import DATASET_REGISTRY
from pipeline.data_loaders.explicit_trust_loader import ExplicitTrustLoader


class DatasetFactory:
    """Factory Method for constructing dataset loaders by name."""

    @staticmethod
    def create(name: str) -> BaseDatasetLoader:
        """
        Look up `name` (case-insensitive) in the dataset registry and return a
        configured loader ready to call .load() on.

        Raises:
            ValueError: if `name` is not a registered dataset.
        """
        key = name.lower()
        if key not in DATASET_REGISTRY:
            available = ", ".join(sorted(DATASET_REGISTRY.keys()))
            raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")

        config = DATASET_REGISTRY[key]
        return ExplicitTrustLoader(config)

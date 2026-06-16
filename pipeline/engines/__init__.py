"""
Multi-Engine Recommendation System.

Engines:
  - FunkSVDEngine:  Baseline collaborative filtering (Surprise library)
  - TrustSVDEngine: Social-aware matrix factorization (NumPy SGD)
  - LightGCNEngine: Graph Convolutional Network (PyTorch)
  - SASRecEngine:   Sequential Self-Attention (PyTorch Transformer)
"""
from pipeline.engines.base_engine import BaseRecommenderEngine

__all__ = ["BaseRecommenderEngine"]

"""Compatibility shim for legacy imports.

The test suite references `src.models.inference_enhanced` while the current
implementation lives in `src.models.inference`. This module re-exports the
public classes so existing tests (and any external code) continue to work
without modification.
"""
from .inference import SkinLesionPredictor, ModelMetrics  # noqa: F401

__all__ = [
    "SkinLesionPredictor",
    "ModelMetrics",
]

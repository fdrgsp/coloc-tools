"""
Coloc Tools - Tools for colocalization analysis with auto-thresholding algorithms.

This package provides implementations of auto-thresholding algorithms for
colocalization analysis, including the Costes method and bisection method.
"""

__version__ = "0.1.0"

from .costes_auto_threshold import (
    Implementation,
    ThresholdMode,
    fiji_bisection_auto_threshold,
    fiji_costes_auto_threshold,
)

__all__ = [
    "Implementation",
    "ThresholdMode",
    "fiji_bisection_auto_threshold",
    "fiji_costes_auto_threshold",
]

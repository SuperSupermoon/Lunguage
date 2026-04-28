"""
LunguageScore - A comprehensive evaluation toolkit for medical report generation

This package provides tools for:
1. Structuring medical reports from raw text (optional)
2. Evaluating structured reports using LunguageScore metric

Main components:
- LunguageScorer: Main class for report evaluation
- ReportStructurer: Class for structuring raw reports (optional)
- MetricEvaluator: Class for calculating LunguageScore
- Config: Configuration management
"""

from .scorer import LunguageScorer
from .structurer import ReportStructurer
from .metric_evaluator import LunguageMetricEvaluator
from .config import Config, StructuringConfig, MetricConfig, SingleSRConfig, SequentialSRConfig
from .lunguagescore import set_seed

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

__author__ = "Jonghak Moon"

__all__ = [
    "LunguageScorer",
    "ReportStructurer",
    "LunguageMetricEvaluator",
    "Config",
    "StructuringConfig",
    "MetricConfig",
    "SingleSRConfig",
    "SequentialSRConfig",
    "set_seed",
]

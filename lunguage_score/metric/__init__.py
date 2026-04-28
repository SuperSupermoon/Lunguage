"""
Metric calculation subpackage for LunguageScore
"""

import logging

_logger = logging.getLogger(__name__)

try:
    from .sota_metrics import (
        calculate_rate_score,
        calculate_green_score,
        calculate_fineradscore,
        calculate_bleu,
        calculate_bertscore,
        calculate_radgraphF1,
    )
    _SOTA_AVAILABLE = True
except (ImportError, ValueError, Exception) as e:
    _SOTA_AVAILABLE = False
    _logger.warning("Some sota_metrics features are unavailable: %s", e)

__all__ = [
    "calculate_rate_score",
    "calculate_green_score",
    "calculate_fineradscore",
    "calculate_bleu",
    "calculate_bertscore",
    "calculate_radgraphF1",
]

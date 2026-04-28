"""
Local copy of the GREEN metric to avoid the upstream package's hard
Python==3.12.1 requirement. This module is vendored from
https://github.com/Stanford-AIMI/GREEN (commit 09e1de7) and keeps the
same public API so that `from green_score import GREEN` works inside the
existing metrics pipeline.
"""

from .green import GREEN

__all__ = ["GREEN"]






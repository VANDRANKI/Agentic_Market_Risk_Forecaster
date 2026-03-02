"""
pytest configuration for Agentic Market Risk Forecaster tests.

On Windows, scipy sub-modules (stats, linalg) must be loaded in a specific
order to avoid DLL import failures caused by the paging file being too small
when multiple C extensions are loaded simultaneously.

Pre-loading numpy and scipy here, before any test module is imported, ensures
the DLL load order is correct and avoids the race condition.
"""

import numpy  # noqa: F401
import scipy.linalg  # noqa: F401
import scipy.stats  # noqa: F401

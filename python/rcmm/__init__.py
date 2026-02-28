# ============================================================================
# risk-constrained-mm :: python/rcmm/__init__.py
# ============================================================================
"""RCMM — Risk-Constrained Market-Making environment."""

from __future__ import annotations

import os
import sys

# On Windows with MinGW/UCRT64-built extensions, Python >= 3.8 requires
# explicit DLL search directories for the GCC runtime (libstdc++, libgcc).
if sys.platform == "win32":
    _ucrt64_bin = r"C:\msys64\ucrt64\bin"
    if os.path.isdir(_ucrt64_bin):
        os.add_dll_directory(_ucrt64_bin)

from rcmm.env import LimitOrderBookEnv
from rcmm._rcmm_core import EnvConfig, HawkesParams, MarkConfig
from rcmm.regime_wrapper import (
    RegimeRandomizationWrapper,
    RegimeSpec,
    NORMAL_REGIME_SPEC,
    FLASH_CRASH_REGIME_SPEC,
)
from rcmm.baselines import AvellanedaStoikovAgent, AvellanedaStoikovConfig
from rcmm.stats import diebold_mariano

__all__ = [
    "LimitOrderBookEnv",
    "EnvConfig",
    "HawkesParams",
    "MarkConfig",
    "RegimeRandomizationWrapper",
    "RegimeSpec",
    "NORMAL_REGIME_SPEC",
    "FLASH_CRASH_REGIME_SPEC",
    "AvellanedaStoikovAgent",
    "AvellanedaStoikovConfig",
    "diebold_mariano",
]

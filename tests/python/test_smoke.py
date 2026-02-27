"""
risk-constrained-mm :: tests/python/test_smoke.py
=================================================
Phase 1 smoke test: verifies that pytest discovers and runs tests correctly.
"""


def test_pytest_is_operational() -> None:
    """Trivial assertion to confirm the test framework works."""
    assert 1 + 1 == 2


def test_numpy_importable() -> None:
    """Verify that NumPy is available (core dependency for the Gym environment)."""
    import numpy as np

    arr = np.zeros(5, dtype=np.float64)
    assert arr.shape == (5,)
    assert arr.sum() == 0.0


def test_gymnasium_importable() -> None:
    """Verify that Gymnasium is available (we'll wrap the C++ engine as a Gym env)."""
    import gymnasium

    assert hasattr(gymnasium, "Env")

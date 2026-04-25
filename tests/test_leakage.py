"""
Point-in-time leakage detection tests — thin wrapper to run via pytest.
Full test suite lives in data/features/leakage_tests.py.
"""
import subprocess
import sys
from pathlib import Path


def test_leakage_tests_pass():
    """Run the standalone leakage test suite and assert zero failures."""
    result = subprocess.run(
        [sys.executable, "data/features/leakage_tests.py"],
        cwd=str(Path(__file__).parent.parent),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Leakage tests FAILED:\n{result.stdout}\n{result.stderr}"
    )

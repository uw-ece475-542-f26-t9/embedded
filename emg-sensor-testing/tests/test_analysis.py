#!/usr/bin/env python3
"""End-to-end check of analyze.py on simulated recordings with known answers.

Each case runs one analyze.py command and checks it reaches the right verdict: good data must PASS,
data built to fail must FAIL for the right reason. Plots go to a temporary folder, not results/.

Usage: python tests/test_analysis.py        (or: pytest tests/)
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM = Path(tempfile.mkdtemp(prefix="emg_sim_"))
ENV = {**os.environ, "EMG_RESULTS_DIR": str(SIM / "results")}
_made = False


def run(*args):
    """Run analyze.py with recording names resolved inside the simulation folder; return its output."""
    global _made
    if not _made:
        subprocess.run([sys.executable, str(ROOT / "tests" / "make_sim.py"), str(SIM)], check=True,
                       capture_output=True)
        _made = True
    resolved = [str(SIM / a) if a.startswith(("good_", "bad_", "drop_")) else a for a in args]
    done = subprocess.run([sys.executable, str(ROOT / "analyze.py"), *resolved], env=ENV,
                          capture_output=True, text=True, check=True)
    return done.stdout


def verdict(out):
    return next(line for line in out.splitlines() if line.startswith("Result:")).split()[1]


def test_signal_good_passes():
    assert verdict(run("signal", "--rest", "good_rest", "--squeeze", "good_squeeze")) == "PASS"


def test_alias_good_passes():
    assert verdict(run("alias", "--rest", "good_rest", "--rest-alt", "good_rest_10500",
                       "--squeeze", "good_squeeze")) == "PASS"


def test_alias_catches_moving_spike():
    out = run("alias", "--rest", "bad_rest", "--rest-alt", "bad_rest_10500", "--squeeze", "good_squeeze")
    assert verdict(out) == "FAIL" and "moved: 300 Hz" in out


def test_hold_good_passes():
    assert verdict(run("hold", "--light", "good_light", "--heavy", "good_heavy1", "good_heavy2",
                       "good_heavy3")) == "PASS"


def test_hold_without_fatigue_fails():
    assert verdict(run("hold", "--heavy", "bad_heavy1", "bad_heavy2")) == "FAIL"


def test_hold_stops_where_arm_gives_out():
    out = run("hold", "--heavy", "drop_heavy1", "good_heavy2")
    assert "held 40 s" in out and verdict(out) == "PASS"


def test_control_that_drifts_fails():
    out = run("hold", "--light", "good_heavy1", "--heavy", "good_heavy2")
    assert "FAIL  Light hold: no real trend" in out


def test_curls_pass():
    assert verdict(run("curls", "--light", "good_clight", "--sets", "good_set1", "good_set2")) == "PASS"


def test_curls_cadence_fallback_passes():
    assert verdict(run("curls", "--light", "good_clight", "--sets", "good_set1", "good_set2",
                       "--cadence", "2.0")) == "PASS"


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError:
            failed += 1
            print(f"FAIL  {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

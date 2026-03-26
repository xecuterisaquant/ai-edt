#!/usr/bin/env python3
"""AI-EDT entry point.

Usage:
    python run.py           Run the sieve stress-test suite.
    python run.py watch     Start the live RSS feed watcher.
"""

from __future__ import annotations

import sys

from ai_edt import watcher
from ai_edt.logger import get_logger

logger = get_logger("run")


def watch() -> None:
    """Start the live RSS feed watcher."""
    watcher.start()


def test() -> None:
    """Run the stress-test suite (delegates to tests/test_stress.py)."""
    from tests.test_stress import run_stress_test

    run_stress_test()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        watch()
    else:
        test()

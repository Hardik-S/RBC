from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intraday_execution_research_lab.execution import vwap_schedule  # noqa: E402


class ExecutionScheduleTests(unittest.TestCase):
    def test_vwap_schedule_ignores_non_finite_volume_weights(self) -> None:
        schedule = vwap_schedule(
            volume_profile=np.array([100.0, np.nan, np.inf, -np.inf, 200.0]),
            total_qty=10,
        )

        np.testing.assert_array_equal(schedule, np.array([3, 0, 0, 0, 7]))
        self.assertEqual(int(schedule.sum()), 10)


if __name__ == "__main__":
    unittest.main()

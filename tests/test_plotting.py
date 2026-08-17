from __future__ import annotations

import unittest

from tabtester.plotting import configure_matplotlib_font


class PlottingTest(unittest.TestCase):
    def test_japanese_font_configuration_is_disabled_by_default(self):
        self.assertIsNone(configure_matplotlib_font())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt

from tabtester.plotting import configure_matplotlib_font


class PlottingTest(unittest.TestCase):
    def test_japanese_font_configuration_is_disabled_by_default(self):
        self.assertIsNone(configure_matplotlib_font())

    def test_japanese_font_is_recovered_when_matplotlib_cache_is_stale(self):
        original_family = plt.rcParams["font.family"]
        original_sans = list(plt.rcParams["font.sans-serif"])
        original_minus = plt.rcParams["axes.unicode_minus"]
        try:
            with (
                patch("tabtester.plotting.font_manager.fontManager.ttflist", []),
                patch(
                    "tabtester.plotting.font_manager.findSystemFonts",
                    return_value=["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"],
                ),
                patch("tabtester.plotting.font_manager.fontManager.addfont") as addfont,
                patch("tabtester.plotting.font_manager.FontProperties") as font_properties,
            ):
                font_properties.return_value.get_name.return_value = "Noto Sans CJK JP"
                selected = configure_matplotlib_font(True)

            self.assertEqual(selected, "Noto Sans CJK JP")
            addfont.assert_called_once_with(
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
            )
            self.assertEqual(plt.rcParams["font.family"], ["sans-serif"])
            self.assertEqual(plt.rcParams["font.sans-serif"][0], "Noto Sans CJK JP")
            self.assertFalse(plt.rcParams["axes.unicode_minus"])
        finally:
            plt.rcParams["font.family"] = original_family
            plt.rcParams["font.sans-serif"] = original_sans
            plt.rcParams["axes.unicode_minus"] = original_minus


if __name__ == "__main__":
    unittest.main()

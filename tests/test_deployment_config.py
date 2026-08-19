from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DeploymentConfigTest(unittest.TestCase):
    def test_env_file_is_not_part_of_repository(self):
        self.assertFalse((ROOT / ".env.example").exists())

    def test_compose_has_no_variable_interpolation(self):
        compose_text = (ROOT / "compose.yaml").read_text(encoding="utf-8")
        self.assertNotIn("${", compose_text)
        self.assertIn('PREFETCH_FOUNDATION_MODELS: "tabicl"', compose_text)
        self.assertIn('ACCEPT_TABFM_LICENSE: "0"', compose_text)
        self.assertIn('- "8501:8501"', compose_text)

    def test_dockerfile_has_explicit_build_capability(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("FROM nvcr.io/nvidia/pytorch:26.05-py3", dockerfile)
        self.assertNotIn("ARG BASE_IMAGE", dockerfile)
        self.assertIn("ARG ENABLE_JAPANESE_SUPPORT=0", dockerfile)
        self.assertIn('if [ "${ENABLE_JAPANESE_SUPPORT}" = "1" ]', dockerfile)
        self.assertIn("fonts-noto-cjk", dockerfile)
        self.assertIn("font.sans-serif: Noto Sans CJK JP, DejaVu Sans", dockerfile)
        self.assertNotIn("build-config.json", dockerfile)

    def test_no_python_configuration_layer(self):
        self.assertFalse((ROOT / "tabtester" / "config.py").exists())
        app_text = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("build-config.json", app_text)
        self.assertNotIn("load_build_capabilities", app_text)
        self.assertNotIn("ENABLE_JAPANESE_SUPPORT", app_text)
        self.assertNotIn("configure_matplotlib_font", app_text)
        plotting_text = (ROOT / "tabtester" / "plotting.py").read_text(encoding="utf-8")
        self.assertNotIn("font_manager", plotting_text)


if __name__ == "__main__":
    unittest.main()

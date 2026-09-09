from __future__ import annotations

import unittest

from tabtester.backends.base import BackendConfig
from tabtester.backends.registry import MODEL_SPECS, model_supports_task, registered_model_names


class RegistryTest(unittest.TestCase):
    def test_model_names_are_unique(self):
        names = [spec.name for spec in MODEL_SPECS]
        self.assertEqual(len(names), len(set(names)))

    def test_backend_config_defaults(self):
        config = BackendConfig(task="Regression")
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.tabicl_batch_size, 4)
        self.assertEqual(config.gp_max_iter, 150)

    def test_registered_model_names_preserve_registry_order(self):
        self.assertEqual(registered_model_names(), [spec.name for spec in MODEL_SPECS])

    def test_generic_gp_is_registered_for_regression_only(self):
        self.assertTrue(model_supports_task("GP (Generic Matern)", "Regression"))
        self.assertFalse(model_supports_task("GP (Generic Matern)", "Classification"))


if __name__ == "__main__":
    unittest.main()

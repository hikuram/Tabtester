from __future__ import annotations

import unittest

from tabtester.backends.base import BackendConfig
from tabtester.backends.registry import MODEL_SPECS


class RegistryTest(unittest.TestCase):
    def test_model_names_are_unique(self):
        names = [spec.name for spec in MODEL_SPECS]
        self.assertEqual(len(names), len(set(names)))

    def test_backend_config_defaults(self):
        config = BackendConfig(task="Regression")
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.tabicl_batch_size, 4)


if __name__ == "__main__":
    unittest.main()

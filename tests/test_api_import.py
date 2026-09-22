from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("fastapi") is not None, "fastapi not installed")
class ApiImportTest(unittest.TestCase):
    def test_api_app_imports(self) -> None:
        from landslide_ai.api.app import create_app

        app = create_app(".")
        self.assertIsNotNone(app)
        self.assertTrue(any(route.path == "/config/status" for route in app.routes))
        self.assertTrue(any(route.path == "/graph/infer" for route in app.routes))


if __name__ == "__main__":
    unittest.main()

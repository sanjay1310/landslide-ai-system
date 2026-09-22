from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
import subprocess
import sys


@lru_cache(maxsize=None)
def module_importable(module_name: str) -> bool:
    """Return True only when a dependency can be imported in a clean subprocess."""
    if find_spec(module_name) is None:
        return False

    probe = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        check=False,
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def torch_available() -> bool:
    return module_importable("torch")


def openai_available() -> bool:
    return module_importable("openai")


def xgboost_available() -> bool:
    return module_importable("xgboost")


def lightgbm_available() -> bool:
    return module_importable("lightgbm")


def shap_available() -> bool:
    return module_importable("shap")


def imblearn_available() -> bool:
    return module_importable("imblearn")

"""Allow direct script execution without installing the project as a package."""

from importlib import import_module
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_main(module_name):
    return import_module(module_name).main

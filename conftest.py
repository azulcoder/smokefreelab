"""Ensure the ``src/`` layout is importable during pytest collection.

Python's ``.pth`` resolver can drop paths that contain spaces under some
``uv``/``pip`` editable-install configurations (the project directory here is
``Apply Job/…``). Injecting ``src`` explicitly into ``sys.path`` keeps test
collection hermetic regardless of how the venv was created.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

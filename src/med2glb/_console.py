"""Shared Rich console singletons for med2glb CLI modules."""

from __future__ import annotations

import sys

from rich.console import Console

# Reconfigure stdout/stderr to UTF-8 to avoid Windows charmap encoding errors
# with Rich's Unicode spinners.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

console = Console()
err_console = Console(stderr=True)

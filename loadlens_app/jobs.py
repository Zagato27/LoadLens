"""Shared in-memory job state for background report generation."""

import threading
from typing import Dict

JOBS: Dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

__all__ = ["JOBS", "JOBS_LOCK"]



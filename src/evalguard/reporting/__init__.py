"""Structured JSON reporting (Phase 3): reshapes a real, already-persisted
Comparison + its PolicyConfig into a frontend-friendly ExperimentReport. See
build.py's and models.py's docstrings for exactly what is and isn't computed here.
"""

from __future__ import annotations

from .build import build_experiment_report, write_report
from .models import ExperimentReport

__all__ = ["ExperimentReport", "build_experiment_report", "write_report"]

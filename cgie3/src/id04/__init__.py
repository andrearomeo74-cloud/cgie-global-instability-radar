"""
CGIE3-ID-04 package.

Relational Identity Continuity Audit.

This package implements the frozen CGIE3-ID-04 protocol.
"""

from .loader import (
    ID04ExperimentContext,
    ID04LoaderError,
    load_experiment,
)

__all__ = [
    "ID04ExperimentContext",
    "ID04LoaderError",
    "load_experiment",
]

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

from .snapshots import (
    ID04SnapshotError,
    build_snapshots,
)

from .continuity import (
    ID04ContinuityError,
    compute_continuity,
)

from .null_controls import (
    ID04NullControlError,
    run_null_controls,
)

from .robustness import (
    ID04RobustnessError,
    run_robustness,
)

__all__ = [
    "ID04ExperimentContext",
    "ID04LoaderError",
    "load_experiment",
    "ID04SnapshotError",
    "build_snapshots",
    "ID04ContinuityError",
    "compute_continuity",
    "ID04NullControlError",
    "run_null_controls",
    "ID04RobustnessError",
    "run_robustness",
]

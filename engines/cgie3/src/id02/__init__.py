"""CGIE3-ID-02 modular experiment pipeline."""

from .loader import (
    ExperimentLoaderError,
    load_experiment,
)

from .preprocessing import (
    PreprocessingError,
    preprocess,
)

__all__ = [
    "ExperimentLoaderError",
    "PreprocessingError",
    "load_experiment",
    "preprocess",
    "DiscoveryStageError",
    "discover",
    "PersistenceStageError",
    "evaluate_persistence",
    "BootstrapStageError",
    "evaluate_bootstrap",
    "MissingnessStageError",
    "evaluate_missingness",
]

from .discovery import (
    DiscoveryStageError,
    discover,
)

from .persistence import (
    PersistenceStageError,
    evaluate_persistence,
)

from .bootstrap import (
    BootstrapStageError,
    evaluate_bootstrap,
)

from .missingness import (
    MissingnessStageError,
    evaluate_missingness,
)

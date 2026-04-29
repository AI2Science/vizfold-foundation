"""
VizFold backends: pluggable inference and trace extraction.

Each backend (openfold, esmfold) implements the base interface
so visualization and analysis can consume traces uniformly.
"""
from vizfold.backends.base import BackendBase

__all__ = ["BackendBase"]

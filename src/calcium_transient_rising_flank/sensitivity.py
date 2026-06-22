"""Boundaries for latent-confounding-aware sensitivity estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .preprocessing import validate_traces


@dataclass(frozen=True)
class PartialAncestralGraph:
    """Minimal common representation for PAG-producing external estimators."""

    endpoint_marks: np.ndarray
    method: str
    assumptions: tuple[str, ...] = ()


class PAGEstimator(Protocol):
    """Protocol implemented by LPCMCI/SVAR-FCI integrations when configured."""

    def fit(self, traces: np.ndarray) -> PartialAncestralGraph: ...


def run_latent_confounding_sensitivity(
    traces: np.ndarray, estimator: PAGEstimator
) -> PartialAncestralGraph:
    """Execute an explicitly supplied PAG estimator on validated traces.

    LPCMCI and SVAR-FCI are deliberately adapters rather than reimplemented
    here: causal assumptions, conditional-independence tests, and external
    dependencies must be selected and reported with the empirical analysis.
    """

    return estimator.fit(validate_traces(traces))

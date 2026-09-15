"""Prior-Fisher drift-budget structural control."""

from .args import FisherDriftArgs
from .control import FisherDrift

STEERING_METHOD = {
    "category": "structural_control",
    "name": "fisher_drift",
    "control": FisherDrift,
    "args": FisherDriftArgs,
}

__all__ = ["FisherDrift", "FisherDriftArgs"]

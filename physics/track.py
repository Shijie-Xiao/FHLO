"""Track provider interface for the FAST ODE.

Implementations supply the storm translational velocity at any time; the
Fast ODE uses it both for the ocean-coupling alpha term and to advect the
storm center when the position is part of the state. The ensemble pipeline
defines its own lightweight provider (BTTrack in
ensemble/run_ode_from_chis.py) implementing this interface.
"""
from typing import Protocol

import numpy as np


class Track(Protocol):
    """track provider interface"""

    def get_velocity(self, t: float, lon: float, lat: float) -> np.ndarray:
        """get the velocity at time t, lon, lat"""
        ...

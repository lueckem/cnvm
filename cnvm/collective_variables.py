from typing import Protocol
import numpy as np
from numba import njit, int64


class CollectiveVariable(Protocol):
    dimension: int

    def __call__(self, x_traj: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x_traj : np.ndarray
            trajectory of CNVM, shape = (?, num_agents).

        Returns
        -------
        np.ndarray
            trajectory projected down via the collective variable
        """


class OpinionShares:
    def __init__(self, num_opinions, normalize: bool = False):
        """
        Calculate the opinion counts/ percentages.

        Parameters
        ----------
        num_opinions : int
        normalize : bool, optional
            If true return percentages, else counts.
        """
        self.dimension = num_opinions
        self.normalize = normalize

    def __call__(self, x_traj: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x_traj : np.ndarray
            trajectory of CNVM, shape = (?, num_agents).

        Returns
        -------
        np.ndarray
            trajectory projected down via the collective variable
        """
        num_agents = x_traj.shape[1]

        x_agg = _opinion_shares_numba(x_traj.astype(int), self.dimension)
        if self.normalize:
            x_agg /= num_agents
        return x_agg


@njit
def _opinion_shares_numba(x_traj, num_opinions):
    x_agg = np.zeros((x_traj.shape[0], num_opinions))
    for i in range(x_traj.shape[0]):
        this_x_agg = np.bincount(x_traj[i, :])
        x_agg[i, :] = np.concatenate((this_x_agg, np.zeros(num_opinions - this_x_agg.shape[0], dtype=int64)))
    return x_agg

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
            trajectory projected down via the collective variable, shape = (?, self.dimension)
        """


class OpinionShares:
    def __init__(self, num_opinions, normalize: bool = False, weights: np.ndarray = None):
        """
        Calculate the opinion counts/ percentages.

        Parameters
        ----------
        num_opinions : int
        normalize : bool, optional
            If true return percentages, else counts.
        weights: np.ndarray, optional
            weight for each agent's opinion, shape=(num_agents,). Default: Each agent has weight 1.
        """
        self.dimension = num_opinions
        self.normalize = normalize
        self.weights = weights

    def __call__(self, x_traj: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x_traj : np.ndarray
            trajectory of CNVM, shape = (?, num_agents).

        Returns
        -------
        np.ndarray
            trajectory projected down via the collective variable, shape = (?, self.dimension)
        """
        num_agents = x_traj.shape[1]
        x_agg = _opinion_shares_numba(x_traj.astype(int), self.dimension, self.weights)

        if self.normalize:
            if self.weights is None:
                x_agg /= num_agents
            else:
                x_agg /= np.sum(self.weights)
        return x_agg


@njit
def _opinion_shares_numba(x_traj, num_opinions, weights):
    x_agg = np.zeros((x_traj.shape[0], num_opinions))
    for i in range(x_traj.shape[0]):
        x_agg[i, :] = np.bincount(x_traj[i, :], weights=weights, minlength=num_opinions)
    return x_agg

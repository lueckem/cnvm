from typing import Protocol, Union

import networkx as nx
import numpy as np
from numba import njit

from cnvm.parameters import Parameters


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
    def __init__(self, num_opinions,
                 normalize: bool = False,
                 weights: np.ndarray = None,
                 idx_to_return: Union[int, np.ndarray] = None):
        """
        Calculate the opinion counts/ percentages.

        Parameters
        ----------
        num_opinions : int
        normalize : bool, optional
            If true return percentages, else counts.
        weights : np.ndarray, optional
            Weight for each agent's opinion, shape=(num_agents,). Default: Each agent has weight 1.
            Negative weights are allowed.
        idx_to_return : Union[int, np.ndarray], optional
            Shares of which opinions to return. Default: all opinions.
        """
        self.num_opinions = num_opinions
        self.normalize = normalize
        self.weights = weights

        if idx_to_return is None:
            self.idx_to_return = np.arange(num_opinions)
        elif isinstance(idx_to_return, int):
            self.idx_to_return = np.array([idx_to_return])
        else:
            self.idx_to_return = idx_to_return

        self.dimension = len(self.idx_to_return)

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
        x_agg = _opinion_shares_numba(x_traj.astype(int), self.num_opinions, self.weights)
        x_agg = x_agg[:, self.idx_to_return]

        if self.normalize:
            if self.weights is None:
                x_agg /= num_agents
            else:
                x_agg /= np.sum(np.abs(self.weights))
        return x_agg


class Interfaces:
    def __init__(self, network: nx.Graph, normalize: bool = False):
        self.dimension = 1
        self.normalize = normalize
        self.network = network

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
        out = np.zeros((x_traj.shape[0], 1))

        for i in range(x_traj.shape[0]):
            for u, v in self.network.edges:
                if x_traj[i, u] != x_traj[i, v]:
                    out[i, 0] += 1

        if self.normalize:
            out /= self.network.number_of_edges()

        return out


class Propensities:
    def __init__(self, params: Parameters, weighted=False):
        """ Only for 2 opinions. """
        self.dimension = 2
        self.params = params
        self.weighted = weighted

    def __call__(self, x_traj: np.ndarray) -> np.ndarray:
        out = np.zeros((x_traj.shape[0], 2))
        for j in range(self.params.num_agents):
            neighbors = list(self.params.network.neighbors(j))
            for i in range(x_traj.shape[0]):
                share_opinion_0 = np.sum(x_traj[i, neighbors] == 0) / len(neighbors)
                share_opinion_1 = 1 - share_opinion_0
                if x_traj[i, j] == 0:
                    prop_01 = self.params.r_imit * self.params.prob_imit[0, 1] * share_opinion_1 + \
                              self.params.r_noise * self.params.prob_noise[0, 1] / self.params.num_opinions
                    if self.weighted:
                        prop_01 *= len(neighbors)
                    out[i, 0] += prop_01
                else:
                    prop_10 = self.params.r_imit * self.params.prob_imit[1, 0] * share_opinion_0 + \
                              self.params.r_noise * self.params.prob_noise[1, 0] / self.params.num_opinions
                    if self.weighted:
                        prop_10 *= len(neighbors)
                    out[i, 1] += prop_10

        return out


@njit
def _opinion_shares_numba(x_traj, num_opinions, weights):
    x_agg = np.zeros((x_traj.shape[0], num_opinions))
    for i in range(x_traj.shape[0]):
        x_agg[i, :] = np.bincount(x_traj[i, :], weights=weights, minlength=num_opinions)
    return x_agg

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


class OpinionSharesByDegree:
    def __init__(self,
                 num_opinions: int,
                 network: nx.Graph,
                 normalize: bool = False,
                 idx_to_return: Union[int, np.ndarray] = None):
        """
        Calculate the count of each opinion by degree.

        The output has dimension idx_to_return * number of different degrees.
        For example, the first idx_to_return entries will represent the counts for nodes with the smallest degree.

        Parameters
        ----------
        num_opinions : int
        network : nx.Graph
        normalize : bool, optional
            If true return percentages, else counts.
            The normalization is done within each group of nodes with the same degree.
        idx_to_return : Union[int, np.ndarray], optional
            Shares of which opinions to return. Default: all opinions.
        """
        self.degrees_of_nodes = np.array([d for _, d in network.degree()])
        self.degrees = np.sort(np.unique(self.degrees_of_nodes))
        self.num_opinions = num_opinions
        self.normalize = normalize

        if idx_to_return is None:
            self.idx_to_return = np.arange(num_opinions)
        elif isinstance(idx_to_return, int):
            self.idx_to_return = np.array([idx_to_return])
        else:
            self.idx_to_return = idx_to_return

        self.dimension = len(self.idx_to_return) * self.degrees.shape[0]

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
        cv = np.zeros((x_traj.shape[0], self.dimension))
        num_agents = x_traj.shape[1]
        x_traj_int = x_traj.astype(int)

        for i, deg in enumerate(self.degrees):
            weights = np.zeros(num_agents)
            weights[np.nonzero(self.degrees_of_nodes == deg)] = 1
            x_agg = _opinion_shares_numba(x_traj_int, self.num_opinions, weights)
            x_agg = x_agg[:, self.idx_to_return]
            if self.normalize:
                x_agg /= np.sum(weights)
            cv[:, i * len(self.idx_to_return): (i + 1) * len(self.idx_to_return)] = np.copy(x_agg)

        return cv


class CompositeCollectiveVariable:
    def __init__(self, collective_variables: list[CollectiveVariable]):
        """
        Concatenate multiple collective variables.

        Typical use-case: CV1 measures the share of opinion 1 in one part of the network,
        CV2 in a different part of the network (both built via OpinionShares class with weights).
        CompositeCollectiveVariable([CV1, CV2]) concatenates the output of the two.

        Parameters
        ----------
        collective_variables : list
        """
        self.collective_variables = collective_variables
        self.dimension = sum([cv.dimension for cv in collective_variables])

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
        return np.concatenate([cv(x_traj) for cv in self.collective_variables], axis=1)


class Interfaces:
    def __init__(self, network: nx.Graph, normalize: bool = False):
        """
        Count the number of interfaces between opinion 0 and 1.

        Can not be used when there are more than these two opinions.

        Parameters
        ----------
        network : nx.Graph
        normalize : bool, optional
            Normalize by dividing by the number of edges in the network.
        """
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
        if np.max(x_traj) > 1:
            raise ValueError("Interfaces can only be used for 2 opinions.")
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

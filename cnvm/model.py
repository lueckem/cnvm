from __future__ import annotations
import networkx as nx
import numpy as np
from numba import njit
from numba.typed import List
from typing import Optional

from cnvm.parameters import Parameters


class CNVM:
    def __init__(self, params: Parameters):
        """
        Continuous-time Noisy Voter Model.
        
        Parameters
        ----------
        params : Parameters
        """
        self.params = params

        self.neighbor_list = List()  # self.neighbor_list[i] = array of neighbors of node i
        if self.params.network is not None:  # not needed for complete network
            for i in range(self.params.num_agents):
                self.neighbor_list.append(np.array(list(self.params.network.neighbors(i)), dtype=int))

        self.next_event_rate = 1 / (self.params.num_agents * (self.params.r_imit + self.params.r_noise))
        self.noise_probability = self.params.r_noise / (self.params.r_imit + self.params.r_noise)

    def update_network(self, network: Optional[nx.Graph] = None):
        """
        If no network is provided, generation from a NetworkGenerator in params is attempted.
        """
        if network is None:
            if self.params.network_generator is not None:
                network = self.params.network_generator()
            else:
                raise ValueError("Either provide a network or a NetworkGenerator.")

        self.params.set_network(network)

        self.neighbor_list = List()
        for i in range(self.params.num_agents):
            self.neighbor_list.append(np.array(list(self.params.network.neighbors(i)), dtype=int))

        self.next_event_rate = 1 / (self.params.num_agents * (self.params.r_imit + self.params.r_noise))

    def update_rates(self, r_imit: float = None,
                     r_noise: float = None,
                     prob_imit: float | np.ndarray = None,
                     prob_noise: float | np.ndarray = None):
        if r_imit is not None:
            self.params.r_imit = r_imit
        if r_noise is not None:
            self.params.r_noise = r_noise
        if prob_imit is not None:
            if isinstance(prob_imit, (float, int)):
                self.params.prob_imit = prob_imit * np.ones((self.params.num_opinions, self.params.num_opinions))
            else:
                self.params.prob_imit = prob_imit
        if prob_noise is not None:
            if isinstance(prob_noise, (float, int)):
                self.params.prob_noise = prob_noise * np.ones((self.params.num_opinions, self.params.num_opinions))
            else:
                self.params.prob_noise = prob_noise

        self.next_event_rate = 1 / (self.params.num_agents * (self.params.r_imit + self.params.r_noise))
        self.noise_probability = self.params.r_noise / (self.params.r_imit + self.params.r_noise)

    def simulate(self,
                 t_max: float,
                 x_init: np.ndarray = None,
                 len_output: int = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the model from t=0 to t=t_max.

        Parameters
        ----------
        t_max : float
        x_init : np.ndarray, optional
            shape=(num_agents,)
        len_output : int, optional
            number of snapshots to output, as equidistantly spaced as possible between 0 and t_max

        Returns
        -------
        tuple[np.ndarray]
            t_traj (shape=(?,)), x_traj (shape=(?,num_agents))
        """
        if self.params.network is None and self.params.network_generator is not None:
            self.update_network()

        if x_init is None:
            x_init = np.random.choice(np.arange(self.params.num_opinions), size=self.params.num_agents)
        x = np.copy(x_init).astype(int)

        t_delta = 0 if len_output is None else t_max / len_output

        if self.params.network is not None:
            t_traj, x_traj = _simulate_numba(x, t_delta, self.next_event_rate, self.noise_probability, t_max,
                                             self.params.num_agents, self.params.num_opinions, self.params.prob_imit,
                                             self.params.prob_noise, self.neighbor_list)
        else:
            t_traj, x_traj = _simulate_numba_complete_network(x, t_delta, self.next_event_rate, self.noise_probability,
                                                              t_max, self.params.num_agents, self.params.num_opinions,
                                                              self.params.prob_imit, self.params.prob_noise)

        return np.array(t_traj), np.array(x_traj, dtype=int)


@njit
def _simulate_numba(x, t_delta, next_event_rate, noise_probability, t_max, num_agents, num_opinions,
                    prob_imit, prob_noise, neighbor_list):
    x_traj = [np.copy(x)]
    t = 0
    t_traj = [0]

    t_store = t_delta
    while t < t_max:
        t += np.random.exponential(next_event_rate)  # time of next event
        agent = np.random.randint(0, num_agents)  # agent of next event
        noise = True if np.random.random() < noise_probability else False

        if noise:
            new_opinion = np.random.randint(0, num_opinions)
            if np.random.random() < prob_noise[x[agent], new_opinion]:
                x[agent] = new_opinion
        else:
            neighbors = neighbor_list[agent]
            if len(neighbors) == 0:
                continue
            new_opinion = x[np.random.choice(neighbors)]
            if np.random.random() < prob_imit[x[agent], new_opinion]:
                x[agent] = new_opinion

        if t >= t_store:
            t_store += t_delta
            x_traj.append(x.copy())
            t_traj.append(t)

    return t_traj, x_traj


@njit
def _simulate_numba_complete_network(x, t_delta, next_event_rate, noise_probability, t_max, num_agents, num_opinions,
                                     prob_imit, prob_noise):
    x_traj = [np.copy(x)]
    t = 0
    t_traj = [0]

    t_store = t_delta
    while t < t_max:
        t += np.random.exponential(next_event_rate)  # time of next event
        agent = np.random.randint(0, num_agents)  # agent of next event
        noise = True if np.random.random() < noise_probability else False

        if noise:
            new_opinion = np.random.randint(0, num_opinions)
            if np.random.random() < prob_noise[x[agent], new_opinion]:
                x[agent] = new_opinion
        else:
            neighbor = np.random.randint(0, num_agents)
            while neighbor == agent:
                neighbor = np.random.randint(0, num_agents)
            new_opinion = x[neighbor]
            if np.random.random() < prob_imit[x[agent], new_opinion]:
                x[agent] = new_opinion

        if t >= t_store:
            t_store += t_delta
            x_traj.append(x.copy())
            t_traj.append(t)

    return t_traj, x_traj

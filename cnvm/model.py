from __future__ import annotations
import numpy as np
from numba import njit
from numba.typed import List

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
        # self.neighbor_list[i] = array of neighbors of node i
        self.neighbor_list = List()
        self.degree_alpha = None  # array containing d(i)^(1 - alpha)
        self.next_event_rate = None
        self.noise_probability = None

        self.calculate_neighbor_list()
        self.calculate_rates()

    def calculate_neighbor_list(self):
        """
        Calculate and set self.neighbor_list.
        """
        self.neighbor_list = List()
        if self.params.network is not None:  # not needed for complete network
            for i in range(self.params.num_agents):
                self.neighbor_list.append(
                    np.array(list(self.params.network.neighbors(i)), dtype=int)
                )

    def calculate_rates(self):
        """
        Calculate and set self.degree_alpha, self.next_event_rate, and self.noise_probability.
        """
        if self.params.network is not None:
            self.degree_alpha = np.array(
                [d ** (1 - self.params.alpha) for _, d in self.params.network.degree()]
            )
        else:  # fully connected
            self.degree_alpha = np.ones(self.params.num_agents) * (
                self.params.num_agents - 1
            ) ** (1 - self.params.alpha)

        total_rate_noise = self.params.r_noise * self.params.num_agents
        total_rate_imit = self.params.r_imit * np.sum(self.degree_alpha)
        self.next_event_rate = 1 / (total_rate_noise + total_rate_imit)
        self.noise_probability = total_rate_noise / (total_rate_imit + total_rate_noise)

    def update_network(self):
        """
        Update network from NetworkGenerator in params.
        """
        self.params.update_network_by_generator()
        self.calculate_neighbor_list()
        self.calculate_rates()

    def update_rates(
        self, r: float | np.ndarray = None, r_tilde: float | np.ndarray = None
    ):
        self.params.change_rates(r, r_tilde)
        self.calculate_rates()

    def simulate(
        self, t_max: float, x_init: np.ndarray = None, len_output: int = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the model from t=0 to t=t_max.

        Parameters
        ----------
        t_max : float
        x_init : np.ndarray, optional
            Initial state, shape=(num_agents,). If no x_init is given, a random one is generated.
        len_output : int, optional
            Number of snapshots to output, as equidistantly spaced as possible between 0 and t_max.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            t_traj (shape=(?,)), x_traj (shape=(?,num_agents))
        """
        if self.params.network_generator is not None:
            self.update_network()

        if x_init is None:
            x_init = np.random.choice(
                np.arange(self.params.num_opinions), size=self.params.num_agents
            )
        x = np.copy(x_init).astype(int)

        t_delta = 0 if len_output is None else t_max / len_output

        # for complete networks, we have a faster implementation
        if self.params.network is None:
            t_traj, x_traj = _simulate_numba_complete_network(
                x,
                t_delta,
                self.next_event_rate,
                self.noise_probability,
                t_max,
                self.params.num_agents,
                self.params.num_opinions,
                self.params.prob_imit,
                self.params.prob_noise,
            )
        # for alpha = 1, we have a faster implementation
        elif self.params.alpha == 1:
            t_traj, x_traj = _simulate_numba(
                x,
                t_delta,
                self.next_event_rate,
                self.noise_probability,
                t_max,
                self.params.num_agents,
                self.params.num_opinions,
                self.params.prob_imit,
                self.params.prob_noise,
                self.neighbor_list,
            )
        else:
            prob_cumsum = self.degree_alpha / np.sum(self.degree_alpha)
            prob_cumsum = np.cumsum(prob_cumsum)
            t_traj, x_traj = _simulate_numba_alpha(
                x,
                t_delta,
                self.next_event_rate,
                self.noise_probability,
                t_max,
                self.params.num_agents,
                self.params.num_opinions,
                self.params.prob_imit,
                self.params.prob_noise,
                self.neighbor_list,
                prob_cumsum,
            )

        return np.array(t_traj), np.array(x_traj, dtype=int)


@njit
def _simulate_numba(
    x,
    t_delta,
    next_event_rate,
    noise_probability,
    t_max,
    num_agents,
    num_opinions,
    prob_imit,
    prob_noise,
    neighbor_list,
):
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
def _simulate_numba_complete_network(
    x,
    t_delta,
    next_event_rate,
    noise_probability,
    t_max,
    num_agents,
    num_opinions,
    prob_imit,
    prob_noise,
):
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


@njit
def rand_index_numba(prob_cumsum) -> int:
    """
    Sample random index 0 <= i < len(prob_cumsum) according to probability distribution.

    Parameters
    ----------
    prob_cumsum : np.ndarray
        1D array containing the cumulative probabilities, i.e., the first entry is the probability of choosing index 0,
        the second entry the probability of choosing index 0 or 1, and so on. The last entry is 1.
    Returns
    -------
    int
    """
    return np.searchsorted(prob_cumsum, np.random.random(), side="right")


@njit
def _simulate_numba_alpha(
    x,
    t_delta,
    next_event_rate,
    noise_probability,
    t_max,
    num_agents,
    num_opinions,
    prob_imit,
    prob_noise,
    neighbor_list,
    prob_cumsum,
):
    x_traj = [np.copy(x)]
    t = 0
    t_traj = [0]

    t_store = t_delta
    while t < t_max:
        t += np.random.exponential(next_event_rate)  # time of next event
        noise = True if np.random.random() < noise_probability else False

        if noise:
            agent = np.random.randint(0, num_agents)  # agent of next event
            new_opinion = np.random.randint(0, num_opinions)
            if np.random.random() < prob_noise[x[agent], new_opinion]:
                x[agent] = new_opinion
        else:
            agent = rand_index_numba(prob_cumsum)
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

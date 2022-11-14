from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import networkx as nx
import numpy as np

from cnvm.network_generator import NetworkGenerator


@dataclass()
class Parameters:
    """
    Container for all the simulation Parameters of the Continuous-time Noisy Voter Model (CNVM).

    An agent i transitions from his current opinion m to a different opinion n at rate
    r_imit * q(i,n) * prob_imit[m, n] + r_noise * (1/num_opinions) * prob_noise[m, n]
    where q(i,n) is the share of opinion n in the neighborhood of agent i.

    Either a network has to specified, or a NetworkGenerator,
    or num_agents, in which case a complete network is used.

    If prob_imit and prob_noise are given as ndarray, the shape has to be (num_opinions, num_opinions).
    """
    num_opinions: int
    r_imit: float
    r_noise: float
    num_agents: Optional[int] = None
    network: Optional[nx.Graph] = field(default=None, repr=False)
    network_generator: Optional[NetworkGenerator] = None
    prob_imit: float | np.ndarray = 1.0
    prob_noise: float | np.ndarray = 1.0

    def __post_init__(self):
        onemat = np.ones((self.num_opinions, self.num_opinions))
        if isinstance(self.prob_imit, (float, int)):
            self.prob_imit = self.prob_imit * onemat
        if isinstance(self.prob_noise, (float, int)):
            self.prob_noise = self.prob_noise * onemat

        if self.network_generator is not None:
            self.num_agents = self.network_generator.num_agents
            self.network = None
        elif self.network is not None:
            self.num_agents = len(self.network.nodes)
        elif self.num_agents is None:
            raise ValueError("Either a network or a NetworkGenerator or num_agents has to be specified.")

    def get_network(self) -> nx.Graph:
        if self.network_generator is not None:
            return self.network_generator()
        elif self.network is not None:
            return self.network
        else:
            return nx.complete_graph(self.num_agents)

    def set_network(self, network: nx.Graph):
        self.network = network
        self.num_agents = len(network.nodes)


def convert_rate_to_cnvm(r: np.ndarray, r_tilde: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Convert the rates r and r_tilde to the parameters used in the CNVM, i.e., r_imit, r_noise, prob_imit, prob_noise.

    The rates r and r_tilde are defined as:
    An agent i transitions from his current opinion m to a different opinion n at rate
    r[m, n] * q(i, n) + r_tilde[m, n]
    where q(i,n) is the share of opinion n in the neighborhood of agent i.
    """
    num_opinions = r.shape[0]

    # r[m,n] = r_imit * prob_imit[m,n]
    r_imit = np.max(r)
    if r_imit > 0:
        prob_imit = r / r_imit
    else:
        prob_imit = np.zeros((num_opinions, num_opinions))

    # r_tilde[m, n] = r_noise / num_opinions * prob_noise[m, n]
    r_noise = np.max(r_tilde) * num_opinions
    if r_noise > 0:
        prob_noise = r_tilde * num_opinions / r_noise
    else:
        prob_noise = np.zeros((num_opinions, num_opinions))

    return r_imit, r_noise, prob_imit, prob_noise


def convert_rate_from_cnvm(params: Parameters) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the rates used in the CNVM to r and r_tilde.

    The rates r and r_tilde are defined as:
    An agent i transitions from his current opinion m to a different opinion n at rate
    r[m, n] * q(i, n) + r_tilde[m, n]
    where q(i,n) is the share of opinion n in the neighborhood of agent i.
    """
    r = params.r_imit * params.prob_imit
    r_tilde = params.r_noise * params.prob_noise / params.num_opinions
    return r, r_tilde

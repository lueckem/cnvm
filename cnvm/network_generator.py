from typing import Protocol
import networkx as nx
import numpy as np


class NetworkGenerator(Protocol):
    num_agents: int

    def __call__(self) -> nx.Graph:
        """ Generate a network. """

    def __repr__(self) -> str:
        """ Return string representation of the generator. """

    def abrv(self) -> str:
        """ Return short description for file names. """


class ErdosRenyiGenerator:
    def __init__(self, num_agents: int, p: float):
        self.num_agents = num_agents
        self.p = p

    def __call__(self) -> nx.Graph:
        if self.p > 0.2:
            return nx.erdos_renyi_graph(self.num_agents, self.p)
        return nx.fast_gnp_random_graph(self.num_agents, self.p)

    def __repr__(self) -> str:
        return f"Erdos-Renyi random graph with p={self.p} on {self.num_agents} nodes"

    def abrv(self) -> str:
        return f"ER_p{int(self.p * 100)}_N{self.num_agents}"


class RandomRegularGenerator:
    def __init__(self, num_agents: int, d: int):
        self.num_agents = num_agents
        self.d = d

    def __call__(self) -> nx.Graph:
        return nx.random_regular_graph(self.d, self.num_agents)

    def __repr__(self) -> str:
        return f"Uniform random regular graph with d={self.d} on {self.num_agents} nodes"

    def abrv(self) -> str:
        return f"regular_d{self.d}_N{self.num_agents}"


class BarabasiAlbertGenerator:
    def __init__(self, num_agents: int, m: int):
        self.num_agents = num_agents
        self.m = m

    def __call__(self) -> nx.Graph:
        return nx.barabasi_albert_graph(self.num_agents, self.m)

    def __repr__(self) -> str:
        return f"Barabasi-Albert random graph on {self.num_agents} nodes"

    def abrv(self):
        return f"barabasi_m{self.m}_N{self.num_agents}"


class WattsStrogatzGenerator:
    def __init__(self, num_agents: int, num_neighbors: int, p: float):
        self.num_agents = num_agents
        self.num_neighbors = num_neighbors
        self.p = p

    def __call__(self) -> nx.Graph:
        return nx.watts_strogatz_graph(self.num_agents, self.num_neighbors, self.p)

    def __repr__(self) -> str:
        return f"Watts-Strogatz random graph on {self.num_agents} nodes"

    def abrv(self):
        return f"watts_k{self.num_neighbors}_p{int(self.p * 100)}_N{self.num_agents}"


class StochasticBlockGenerator:
    def __init__(self, num_agents: int, p_matrix: np.ndarray):
        """
        p_matrix is a (n x n)-matrix.
        Creates n stochastic blocks, block i is randomly connected to block j with edge density p_matrix[i, j].
        """
        self.p_matrix = p_matrix

        self.num_blocks = p_matrix.shape[0]
        self.block_size = int(num_agents / self.num_blocks)
        self.num_agents = self.block_size * self.num_blocks

    def __call__(self) -> nx.Graph:
        adj_matrix = np.zeros((self.num_agents, self.num_agents))
        for i in range(self.num_blocks):
            for j in range(i + 1):
                this_block = (np.random.random((self.block_size, self.block_size)) <= self.p_matrix[i, j])
                adj_matrix[i * self.block_size: (i + 1) * self.block_size,
                           j * self.block_size: (j + 1) * self.block_size] = this_block

        # only keep the lower triangle and symmetrize
        adj_matrix = np.tril(adj_matrix, -1)
        adj_matrix = adj_matrix + np.tril(adj_matrix).T

        return nx.from_numpy_array(adj_matrix)

    def __repr__(self) -> str:
        return f"stochastic block model with {self.num_blocks} blocks and {self.num_agents} nodes."

    def abrv(self):
        return f"sbm_b{self.num_blocks}_N{self.num_agents}"


class GridGenerator:
    def __init__(self, num_agents: int, periodic: bool = False):
        self.num_agents = num_agents
        self.dim = 2  # only implemented for 2D grids for now
        self.periodic = periodic

        # find shape as square as possible
        shape0 = round(num_agents ** 0.5)
        while num_agents % shape0 != 0:
            shape0 += 1
        shape1 = int(num_agents / shape0)
        self.shape = shape0, shape1

    def __call__(self) -> nx.Graph:
        g = nx.grid_graph(self.shape, periodic=self.periodic)

        relabel_dict = {}
        for i, val in enumerate(g.nodes):
            relabel_dict[val] = i

        return nx.relabel_nodes(g, relabel_dict)

    def __repr__(self) -> str:
        return f"{self.shape} lattice graph graph"

    def abrv(self):
        if self.periodic:
            return f"lattice_{self.dim}D_periodic_N{self.num_agents}"
        else:
            return f"lattice_{self.dim}D_N{self.num_agents}"


class BinomialWattsStrogatzGenerator:
    def __init__(self, num_agents: int, num_neighbors: int, p_rewire: float):
        """
        Creates a ring where each node is connected to the num_neighbors nearest neighbors.
        (num_neighbors needs to be even!)
        Then iterate through each edge and rip it out with probability p_rewire.
        Then iterate through all the possible edges that are not present and insert with such a probability,
        that in expectation the resulting graph has the same number of edges again.
        For p=1, this yields the binomial Erdös-Renyi graph G(N, K/N).
        """
        self.num_agents = num_agents
        self.num_neighbors = num_neighbors
        self.p_rewire = p_rewire

    def __call__(self) -> nx.Graph:
        network = nx.watts_strogatz_graph(self.num_agents, self.num_neighbors, 0)

        # remove edges
        edges = np.array(network.edges)
        idx_to_keep = np.random.random(edges.shape[0]) > self.p_rewire
        edges = edges[idx_to_keep, :]

        # insert edges
        p_insert = self.p_rewire * self.num_neighbors / (self.num_agents - 1 - (1 - self.p_rewire) * self.num_neighbors)
        if p_insert > 0.2:
            network = nx.erdos_renyi_graph(self.num_agents, p_insert)
        else:
            network = nx.fast_gnp_random_graph(self.num_agents, p_insert)
        network.add_edges_from(edges)

        return network

    def __repr__(self) -> str:
        return f"Binomial Watts-Strogatz graph on {self.num_agents} nodes with p_rewire={self.p_rewire}"

    def abrv(self):
        return f"binWS_k{self.num_neighbors}_p{int(self.p_rewire * 100)}_N{self.num_agents}"

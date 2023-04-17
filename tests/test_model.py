from unittest import TestCase
import numpy as np
import networkx as nx

from cnvm.parameters import Parameters
from cnvm.model import CNVM
import cnvm.network_generator as ng


class TestModel(TestCase):
    def setUp(self):
        self.num_opinions = 3
        self.num_agents = 100
        self.r = np.array([[0, 1, 2], [1, 0, 1], [2, 0, 0]])
        self.r_tilde = np.array([[0, 0.2, 0.1], [0, 0, 0.1], [0.1, 0.2, 0]])

        self.params_complete = Parameters(
            num_opinions=self.num_opinions,
            num_agents=self.num_agents,
            r=self.r,
            r_tilde=self.r_tilde,
        )

        self.params_network = Parameters(
            num_opinions=self.num_opinions,
            network=nx.barabasi_albert_graph(self.num_agents, 2),
            r=self.r,
            r_tilde=self.r_tilde,
            alpha=0,
        )

        self.params_generator = Parameters(
            num_opinions=self.num_opinions,
            network_generator=ng.BarabasiAlbertGenerator(self.num_agents, 2),
            r=self.r,
            r_tilde=self.r_tilde,
        )

    def test_output(self):
        model = CNVM(self.params_complete)
        t_max = 100
        t, x = model.simulate(t_max, len_output=10)
        self.assertEqual(t.shape, (10,))
        self.assertEqual(x.shape, (10, self.num_agents))
        self.assertEqual(t[0], 0)
        self.assertTrue(t[-1] >= t_max)

        model = CNVM(self.params_network)
        t, x = model.simulate(t_max, len_output=10)
        self.assertEqual(t.shape, (10,))
        self.assertEqual(x.shape, (10, self.num_agents))
        self.assertEqual(t[0], 0)
        self.assertTrue(t[-1] >= t_max)

        x_init = np.ones(self.num_agents)
        model = CNVM(self.params_generator)
        t, x = model.simulate(t_max, x_init, len_output=10)
        self.assertEqual(t.shape, (10,))
        self.assertEqual(x.shape, (10, self.num_agents))
        self.assertEqual(t[0], 0)
        self.assertTrue(t[-1] >= t_max)
        self.assertTrue(np.allclose(x[0], x_init))

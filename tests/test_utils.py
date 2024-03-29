from unittest import TestCase
import numpy as np

import cnvm.utils as cu
from cnvm.parameters import Parameters
from cnvm.collective_variables import OpinionShares


class TestArgmatch(TestCase):
    def test_1(self):
        t = np.array([1, 2, 3, 4, 5, 6, 7])
        t_ref = np.array([1.8, 2, 4.4, 6.7, 6.9])
        ind = cu.argmatch(t_ref, t)
        self.assertTrue(np.allclose(ind, [1, 1, 3, 6, 6]))

    def test_2(self):
        t = np.array([1, 2, 3, 4, 5])
        t_ref = np.array([0.1, 1.8, 2, 2.5, 4.4, 6.7, 7.7])
        ind = cu.argmatch(t_ref, t)
        self.assertTrue(np.allclose(ind, [0, 1, 1, 2, 3, 4, 4]))


class TestSampleManyRuns(TestCase):
    def setUp(self):
        self.num_opinions = 3
        self.num_agents = 100
        self.r = np.array([[0, 1, 2], [1, 0, 1], [2, 0, 0]])
        self.r_tilde = np.array([[0, 0.2, 0.1], [0, 0, 0.1], [0.1, 0.2, 0]])

        self.params = Parameters(
            num_opinions=self.num_opinions,
            num_agents=self.num_agents,
            r=self.r,
            r_tilde=self.r_tilde,
        )

        self.t_max = 100
        self.num_timesteps = 10
        self.cv = OpinionShares(self.num_opinions)
        self.num_initial_states = 5
        self.initial_states = np.random.randint(
            2,
            size=(self.num_initial_states, self.num_agents),
        )

    def test_split_runs(self):
        num_runs, num_chunks = 20, 6
        chunks = cu._split_runs(num_runs, num_chunks)
        self.assertTrue(np.allclose(chunks, [4, 4, 3, 3, 3, 3]))

    def test_parallelization_runs(self):
        t, x = cu.sample_many_runs(
            params=self.params,
            initial_states=self.initial_states,
            t_max=self.t_max,
            num_timesteps=self.num_timesteps,
            num_runs=15,
            n_jobs=2,
        )
        self.assertEqual(t.shape, (self.num_timesteps,))
        self.assertEqual(
            x.shape,
            (
                self.num_initial_states,
                15,
                self.num_timesteps,
                self.num_agents,
            ),
        )

    def test_parallelization_initial_states(self):
        t, x = cu.sample_many_runs(
            params=self.params,
            initial_states=self.initial_states,
            t_max=self.t_max,
            num_timesteps=self.num_timesteps,
            num_runs=3,
            n_jobs=2,
        )
        self.assertEqual(t.shape, (self.num_timesteps,))
        self.assertEqual(
            x.shape,
            (
                self.num_initial_states,
                3,
                self.num_timesteps,
                self.num_agents,
            ),
        )

    def test_no_parallelization(self):
        t, x = cu.sample_many_runs(
            params=self.params,
            initial_states=self.initial_states,
            t_max=self.t_max,
            num_timesteps=self.num_timesteps,
            num_runs=3,
            n_jobs=None,
        )
        self.assertEqual(t.shape, (self.num_timesteps,))
        self.assertEqual(
            x.shape,
            (
                self.num_initial_states,
                3,
                self.num_timesteps,
                self.num_agents,
            ),
        )

    def test_parallelization_with_cv(self):
        t, x = cu.sample_many_runs(
            params=self.params,
            initial_states=self.initial_states,
            t_max=self.t_max,
            num_timesteps=self.num_timesteps,
            num_runs=15,
            n_jobs=2,
            collective_variable=self.cv,
        )
        self.assertEqual(t.shape, (self.num_timesteps,))
        self.assertEqual(
            x.shape,
            (
                self.num_initial_states,
                15,
                self.num_timesteps,
                self.num_opinions,
            ),
        )

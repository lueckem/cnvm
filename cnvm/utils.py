import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
import multiprocessing as mp

from cnvm.parameters import Parameters
from cnvm.model import CNVM
from cnvm.collective_variables import CollectiveVariable


def sample_many_runs(params: Parameters,
                     x_init: np.ndarray,
                     t_max: float,
                     num_timesteps: int,
                     num_runs: int,
                     n_jobs: int = None,
                     collective_variable: CollectiveVariable = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample multiple runs of the CNVM specified by params.

    Parameters
    ----------
    params : Parameters
        If a NetworkGenerator is used, a new network will be sampled for every run.
    x_init : np.ndarray
        Initial state.
    t_max : float
        End time.
    num_timesteps : int
        Trajectory will be saved at equidistant time points np.linspace(0, t_max, num_timesteps).
    num_runs : int
        Number of samples.
    n_jobs : int, optional
        If "None", no multiprocessing is applied. If "-1", all available CPUs will be used.
    collective_variable : CollectiveVariable, optional
        If collective variable is specified, the projected trajectory will be returned
        instead of the full trajectory.

    Returns
    -------
    t_out : np.ndarray
        time points, shape = (num_timesteps,)
    x_out : np.ndarray
        trajectories, shape = (num_runs, num_timesteps, num_agents)

    """
    t_out = np.linspace(0, t_max, num_timesteps)

    if n_jobs is None:
        x_out = _sample_many_runs_subprocess(params, x_init, t_max, num_timesteps, num_runs, collective_variable)
        return t_out, x_out

    else:
        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        num_runs_per_process = int(np.ceil(num_runs / n_jobs))
        with mp.Pool(n_jobs) as pool:
            x_out = pool.starmap(_sample_many_runs_subprocess,
                                 [(params, x_init, t_max, num_timesteps, num_runs_per_process,
                                   collective_variable)] * n_jobs)
            x_out = np.concatenate(x_out, axis=0)
            return t_out, x_out


def _sample_many_runs_subprocess(params: Parameters,
                                 x_init: np.ndarray,
                                 t_max: float,
                                 num_timesteps: int,
                                 num_runs: int,
                                 collective_variable: CollectiveVariable = None) -> np.ndarray:
    t_out = np.linspace(0, t_max, num_timesteps)
    model = CNVM(params)
    if collective_variable is None:
        x_out = np.zeros((num_runs, num_timesteps, model.params.num_agents))
    else:
        x_out = np.zeros((num_runs, num_timesteps, collective_variable.dimension))

    for i in range(num_runs):
        if model.params.network_generator is not None:
            model.update_network()
        t, x = model.simulate(t_max, len_output=4 * num_timesteps, x_init=x_init)
        t_ind = argmatch(t_out, t)
        if collective_variable is None:
            x_out[i, :, :] = x[t_ind, :]
        else:
            x_out[i, :, :] = collective_variable(x[t_ind, :])
    return x_out


def calc_rre_traj(params: Parameters,
                  c_0: np.ndarray,
                  t_max: float,
                  t_eval=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the RRE given parameters, starting from c_0, up to time t_max.

    Outputs timepoints (shape=(?,)) and c (shape=(?, num_opinions)).
    """

    def rhs(t, c):
        out = np.zeros_like(c)
        for m in range(params.num_opinions):
            for n in range(params.num_opinions):
                if n == m:
                    continue

                state_change = np.zeros_like(c)
                state_change[m] = -1
                state_change[n] = 1

                prop = c[m] * (params.r_imit * params.prob_imit[m, n] * c[n] +
                               params.r_noise * params.prob_noise[m, n] / params.num_opinions)

                out += prop * state_change
        return out

    sol = solve_ivp(rhs, (0, t_max), c_0, rtol=1e-8, atol=1e-8, t_eval=t_eval)
    return sol.t, sol.y.T


@njit
def argmatch(x_ref, x):
    """
    Find indices such that |x[indices] - x_ref| = min!

    Parameters
    ----------
    x_ref : np.ndarray
        1D, sorted
    x : np.ndarray
        1D, sorted
    Returns
    -------
    np.ndarray
    """
    size = np.shape(x_ref)[0]
    out = np.zeros(size, dtype=np.int64)
    ref_ind = 0
    ind = 0
    while x_ref[ref_ind] < x[ind]:
        ref_ind += 1

    while ref_ind < size and ind < x.shape[0] - 1:
        if x[ind] <= x_ref[ref_ind] <= x[ind + 1]:
            if np.abs(x[ind] - x_ref[ref_ind]) < np.abs(x[ind + 1] - x_ref[ref_ind]):  # smaller is nearer
                out[ref_ind] = ind
            else:  # bigger is nearer
                out[ref_ind] = ind + 1
            ref_ind += 1
        else:
            ind += 1

    while ref_ind < size:
        out[ref_ind] = x.shape[0] - 1
        ref_ind += 1

    return out

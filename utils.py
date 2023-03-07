"""Utility functions and config."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sp
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist, pdist
from tqdm.auto import tqdm

from dm import diffusion_maps
from int.lorenz import integrate

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True


config = {}
config["EXAMPLE"] = "brusselator"
# config["EXAMPLE"] = 'lorenz'
config["PATH"] = "data/"

config["DATA"] = {}
config["DATA"]["n_train"] = 5
config["DATA"]["n_val"] = 1
config["DATA"]["n_test"] = 1
config["DATA"]["l_trajectories"] = 2000
config["DATA"]["l_trajectories_test"] = 500
config["DATA"]["lenght_chunks"] = 10
config["DATA"]["shift_betw_chunks"] = 4
config["DATA"]["initial_set_off"] = 20
config["DATA"]["max_n_transients"] = 200
config["DATA"]["max_warmup"] = 50
config["DATA"]["integration_steps"] = 250
config["DATA"]["gh_lenght_chunks"] = 5

config["MODEL"] = {}
# Number of variables to use when using the lorenz system
config["MODEL"]["input_size"] = 3
config["MODEL"]["reservoir_size"] = 2000
config["MODEL"]["scale_rec"] = 0.9
config["MODEL"]["scale_in"] = 0.02
config["MODEL"]["quadratic"] = False

config["TRAINING"] = {}
config["TRAINING"]["epochs"] = 500
config["TRAINING"]["batch_size"] = 400
config["TRAINING"]["learning_rate"] = 5e-1
config["TRAINING"]["ridge"] = True
config["TRAINING"]["dtype"] = torch.float64
config["TRAINING"]["gh_num_eigenpairs"] = 100

config["PLOTS"] = {}
config["PLOTS"]["textwidth_pts"] = 505
config["PLOTS"]["textwidth_inch"] = config["PLOTS"]["textwidth_pts"] / 72.27


class LorenzDataset(torch.utils.data.Dataset):
    """Dataset of transients obtained from a Brusselator."""

    def __init__(self, n_train, l_trajectories, input_size, verbose=False):
        self.ids = np.arange(n_train)

        self.x = []
        self.y = []
        self.y_data = []
        for i in tqdm(range(n_train), leave=True, position=0):
            data_dict = integrate(T=l_trajectories, ic="random")
            self.x.append(data_dict["data"][:-1, :input_size])
            self.y.append(data_dict["data"][1:, :input_size])
        self.tt = data_dict["tt"]
        if verbose:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(
                data_dict["data"][:, 0],
                data_dict["data"][:, 1],
                data_dict["data"][:, 2],
            )
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_zlabel(r"$z$")
            plt.savefig("fig/lorenz_trajectory.pdf")
            plt.show()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x = self.x[self.ids[index]]
        y = self.y[self.ids[index]]
        return torch.tensor(x, dtype=torch.get_default_dtype()), torch.tensor(y, dtype=torch.get_default_dtype())


def f_brusselator(t, y, arg_a, arg_b):
    """Temporal evolution of the Brusselator."""
    dx_dt = arg_a + np.real(y) ** 2 * np.imag(y) - (arg_b + 1) * np.real(y)
    dy_dt = arg_b * np.real(y) - np.real(y) ** 2 * np.imag(y)
    return dx_dt + 1.0j * dy_dt


def integrate_brusselator(
    initial_condition,
    l_trajectories,
    tmin=0,
    delta_t=1e-3,
    parameters={"a": 1.0, "b": 2.1},
):
    """Integrate Brusselator using initial condition y0."""
    initial_time = 0.0
    tmax = l_trajectories / 5
    tt_arr = np.linspace(tmin, tmax, l_trajectories + 1)
    data = []
    runner = sp.ode(f_brusselator).set_integrator("zvode", method="Adams")
    runner.set_initial_value(initial_condition, initial_time).set_f_params(parameters["a"], parameters["b"])
    data.append(runner.y)

    i = 0
    while runner.successful() and np.abs(runner.t) < np.abs(tmax):
        i = i + 1
        runner.integrate(runner.t + delta_t)
        if i % int(float(tmax) / float(l_trajectories * delta_t)) == 0:
            data.append(runner.y)
    return tt_arr, np.array(data, dtype="complex")


class BrusselatorDataset(torch.utils.data.Dataset):
    """Dataset of transients obtained from a Brusselator."""

    def __init__(self, n_train, l_trajectories, verbose=False):
        self.ids = np.arange(n_train)

        self.x = []
        self.y = []
        self.y_data = []
        for i in tqdm(range(n_train), leave=True, position=0):
            y0 = 2.0 * np.random.random() + 1.0j * 3.0 * np.random.random()
            tt_arr, trajectory = integrate_brusselator(y0, l_trajectories)
            self.x.append(trajectory[:-1].real)
            self.y.append(trajectory[1:].real)
            self.y_data.append(trajectory[:-1].imag)
        self.tt = tt_arr
        if verbose:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(trajectory[:-1].real, trajectory[:-1].imag)
            ax.set_xlabel(r"$u$")
            ax.set_ylabel(r"$v$")
            plt.savefig("fig/brusselator_trajectory.pdf")
            plt.show()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x = self.x[self.ids[index]]
        y = self.y[self.ids[index]]
        return torch.tensor(x, dtype=torch.get_default_dtype()), torch.tensor(y, dtype=torch.get_default_dtype())

    def save_data(self, path, filename):
        np.savez(
            path + filename,
            x=self.x,
            y=self.y,
            y_data=self.y_data,
            tt=self.tt,
            ids=self.ids,
        )


class LoadBrusselatorDataset(torch.utils.data.Dataset):
    """Dataset of transients obtained from a Brusselator."""

    def __init__(self, path, filename, verbose=False):
        data = np.load(path + filename)
        self.ids = data["ids"]
        self.x = data["x"]
        self.y = data["y"]
        self.y_data = data["y_data"]
        self.tt = data["tt"]
        if verbose:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.x[0], self.y_data[0])
            ax.set_xlabel(r"$u$")
            ax.set_ylabel(r"$v$")
            plt.savefig("fig/brusselator_trajectory.pdf")
            plt.show()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x = self.x[self.ids[index]]
        y = self.y[self.ids[index]]
        return torch.tensor(x, dtype=torch.get_default_dtype()), torch.tensor(y, dtype=torch.get_default_dtype())


def nystrom(Xnew, EigVec, Data, EigenVal, eps):
    epsq = (eps) * 2
    [Nsamp, dsmall] = EigVec.shape
    phi = np.zeros((dsmall, 1))
    dist = cdist(Xnew, Data, metric="sqeuclidean")
    dist = np.array(dist)
    w = np.exp((-dist) / epsq)
    Wtotal = np.sum(w)
    for j in range(0, dsmall):
        phi[j] = (np.sum((w[0, :] / (Wtotal) * EigVec[:, j]))) * (1 / EigenVal[j])
    return phi


def create_chunks(data, max_n_transients, length_chunks, shift_betw_chunks):
    chunks = []
    for i in range(int(min(len(data), max_n_transients))):
        for j in range(int((len(data[i]) - length_chunks) / shift_betw_chunks) + 1):
            chunks.append(data[i][int(j * shift_betw_chunks) : int(j * shift_betw_chunks + length_chunks)])
    chunks = np.squeeze(np.array(chunks))
    return chunks


def dmaps(data, eps=None, return_eps=False):
    """Do diffusion maps on data."""
    pw_dists = pdist(data, "euclidean")
    if eps is None:
        eps = np.median(pw_dists**2)  # scale parameter
    dmap = diffusion_maps.SparseDiffusionMaps(
        points=data,
        epsilon=eps,
        num_eigenpairs=12,
        cut_off=np.inf,
        renormalization=0,
        normalize_kernel=True,
        use_cuda=False,
    )
    V = dmap.eigenvectors.T
    D = dmap.eigenvalues.T
    if return_eps:
        return D, V, eps
    return D, V

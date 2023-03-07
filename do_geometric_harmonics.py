"""
This module contains functions to do geometric harmonics on Brusselator time series.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import cdist, pdist
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# import dm as diffusion_maps
from dm import diffusion_maps, geometric_harmonics

# from datafold import dynfold as diffusion_maps
from utils import ESN, ESNModel, LoadBrusselatorDataset, config, create_chunks, dmaps

torch.set_default_dtype(config["TRAINING"]["dtype"])

# Colors for plotting
# plt.style.use('tableau-colorblind10')
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

CMAP = "plasma"


def get_hidden_states(dataset, model):
    """Get the cts corresponding to the xts."""
    hidden_states = (
        model.net.forward(torch.tensor(dataset.x, dtype=torch.get_default_dtype()), return_states=True)[0]
        .detach()
        .cpu()
        .numpy()
    )
    if model.net.quadratic:
        hidden_states = hidden_states[:, :, : model.net.reservoir_size]
    hidden_states = np.concatenate((np.zeros_like(hidden_states[:, :1]), hidden_states), axis=1)[:, :-1]
    return hidden_states


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


def plot_integration_w_gh(dataset_test, path="data/"):

    initial_condition = torch.tensor(dataset_test.x[0], dtype=torch.float)

    trajectory_w_warmup = np.load(path + "gh_trajectory_w_warmup.npy")
    trajectory = np.load(path + "gh_trajectory.npy")

    warmup_length = config["DATA"]["gh_lenght_chunks"]

    Ttotal = 400
    tmax = dataset_test.tt[-1]
    tis = len(dataset_test.tt)
    dt = tmax / tis
    ttnow = np.linspace(0, dt * Ttotal, Ttotal)
    tt_init = np.linspace(-dt * len(initial_condition), -dt, len(initial_condition))
    start_point = config["DATA"]["max_warmup"]

    fig = plt.figure(figsize=(config["PLOTS"]["textwidth_inch"], 2.5))
    ax = fig.add_subplot(111)
    ax.plot(
        np.append(tt_init[-warmup_length:], ttnow),
        trajectory_w_warmup,
        ".-",
        label="with warmup",
        color=COLORS[2],
    )
    ax.plot(
        ttnow[: min(len(initial_condition[start_point:]), Ttotal)],
        initial_condition[start_point : min(Ttotal + start_point, len(initial_condition))],
        ".-",
        label="true",
        color=COLORS[0],
        markersize=5,
    )
    ax.plot(
        ttnow[: len(trajectory) - warmup_length],
        trajectory[warmup_length:],
        ".-",
        label="with geometric harmonics",
        color=COLORS[7],
        markersize=5,
    )
    ax.plot(
        tt_init[-warmup_length:],
        initial_condition[start_point - warmup_length : start_point],
        ".-",
        label="warmup",
        color=COLORS[1],
        markersize=5,
    )
    ax.axvline(x=0, color="k", zorder=-1, alpha=0.5)
    ax.legend(loc=4, fontsize=10)
    ax.set_ylabel("$u$")
    ax.set_xlim((-dt * warmup_length, dt * Ttotal))
    ax.set_xlabel("$t$")
    ax.set_xlim((-2, 20))
    plt.tight_layout()
    plt.savefig(config["PATH"] + "figure_7.eps")
    plt.savefig(config["PATH"] + "figure_7.pdf")
    plt.close()


def main():
    if config["EXAMPLE"] == "brusselator":
        config["MODEL"]["input_size"] = 1

    dataset_train = LoadBrusselatorDataset(path=config["PATH"], filename="training_data.npz")
    dataset_val = LoadBrusselatorDataset(path=config["PATH"], filename="validation_data.npz")
    dataset_test = LoadBrusselatorDataset(path=config["PATH"], filename="test_data.npz")

    # Create PyTorch dataloaders for train and test data
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config["TRAINING"]["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config["TRAINING"]["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Create the network architecture
    network = ESN(
        config["MODEL"]["input_size"],
        config["MODEL"]["reservoir_size"],
        config["MODEL"]["input_size"],
        config["MODEL"]["scale_rec"],
        config["MODEL"]["scale_in"],
        quadratic=config["MODEL"]["quadratic"],
    )
    print(network)

    # Create model wrapper around architecture
    # Contains train and validation functions
    model = ESNModel(
        dataloader_train,
        dataloader_val,
        network,
        learning_rate=config["TRAINING"]["learning_rate"],
    )
    model.load_network(config["PATH"] + "model_")

    c_train = get_hidden_states(dataset_train, model)
    c_test = get_hidden_states(dataset_test, model)

    x_data_train = dataset_train.x[:, config["DATA"]["initial_set_off"] :]
    c_data_train = c_train[:, config["DATA"]["initial_set_off"] :]

    x_data_test = dataset_test.x[:, config["DATA"]["initial_set_off"] :]
    c_data_test = c_test[:, config["DATA"]["initial_set_off"] :]

    x_chunks_train = create_chunks(
        x_data_train,
        config["DATA"]["max_n_transients"],
        config["DATA"]["gh_lenght_chunks"],
        config["DATA"]["shift_betw_chunks"],
    )
    c_chunks_train = create_chunks(
        c_data_train,
        config["DATA"]["max_n_transients"],
        config["DATA"]["gh_lenght_chunks"],
        config["DATA"]["shift_betw_chunks"],
    )
    c_chunks_train = c_chunks_train[:, 0, :]

    x_chunks_test = create_chunks(
        x_data_test,
        config["DATA"]["max_n_transients"],
        config["DATA"]["gh_lenght_chunks"],
        int(5 * config["DATA"]["shift_betw_chunks"]),
    )
    c_chunks_test = create_chunks(
        c_data_test,
        config["DATA"]["max_n_transients"],
        config["DATA"]["gh_lenght_chunks"],
        int(5 * config["DATA"]["shift_betw_chunks"]),
    )
    c_chunks_test = c_chunks_test[:, 0, :]

    # Diffusion maps on input data
    D, V, eps = dmaps(x_chunks_train, return_eps=True)

    for i in range(2, 5):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(V[:, 1], V[:, i], c=c_chunks_train[:, 0], cmap=CMAP)
        ax.set_xlabel("")
        ax.set_ylabel("")
        # plt.savefig('')
        plt.show()
    V = V[:, [1, 3]]
    D = D[[1, 3]]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    scat1 = ax.scatter(V[:, 0], V[:, 1], c=c_chunks_train[:, 0], cmap=CMAP)
    ax.set_xlabel(r"$\phi_1$")
    ax.set_ylabel(r"$\phi_2$")
    cbar = plt.colorbar(scat1)
    cbar.set_label("$h_t^{(1)}$")
    plt.savefig(config["PATH"] + "dmaps_on_input_data_2d.png", dpi=300)
    plt.savefig(config["PATH"] + "dmaps_on_input_data_2d.pdf")
    plt.close()

    print("Creating geometric harmonics.")
    V_train, V_test, c_chunks_train_train, c_chunks_train_test = train_test_split(
        V, c_chunks_train, random_state=np.random.seed(10), train_size=8 / 10
    )

    pw = pdist(V_train, "euclidean")
    eps_GH = np.median(pw**2) * 0.05

    GH = diffusion_maps.SparseDiffusionMaps(
        points=V_train,
        epsilon=eps_GH,
        num_eigenpairs=config["TRAINING"]["gh_num_eigenpairs"],
        cut_off=np.inf,
        renormalization=0,
        normalize_kernel=False,
    )
    print("Interpolation function.")
    interp_c = geometric_harmonics.GeometricHarmonicsInterpolator(
        points=V_train, epsilon=None, values=c_chunks_train_train, diffusion_maps=GH
    )

    print("Interpolating.")
    # predictions_train = np.c_[interp_c1(V_train), interp_c2(V_train), interp_c3(V_train),
    #                           interp_c4(V_train)]
    predictions_train = interp_c(V_train)

    fig = plt.figure(figsize=(10, 10))
    axs = []
    for i in range(1, 5):
        ax = fig.add_subplot(2, 2, i, aspect="equal")
        axs.append(ax)
        ax.scatter(c_chunks_train_train[::10, i - 1], predictions_train[::10, i - 1], s=7)
        ax.plot(
            np.linspace(
                np.min(c_chunks_train_train[:, i - 1]),
                np.max(c_chunks_train_train[:, i - 1]),
                10,
            ),
            np.linspace(
                np.min(c_chunks_train_train[:, i - 1]),
                np.max(c_chunks_train_train[:, i - 1]),
                10,
            ),
            "k",
        )
    axs[0].set_xlabel("True $h^{(1)}$")
    axs[0].set_ylabel("Predicted $h^{(1)}$")
    axs[1].set_xlabel("True $h^{(2)}$")
    axs[1].set_ylabel("Predicted $h^{(2)}$")
    axs[2].set_xlabel("True $h^{(3)}$")
    axs[2].set_ylabel("Predicted $h^{(3)}$")
    axs[3].set_xlabel("True $h^{(4)}$")
    axs[3].set_ylabel("Predicted $h^{(4)}$")
    plt.tight_layout()
    plt.savefig(
        config["PATH"] + "geometric_harmonics_h_" + str(config["DATA"]["gh_lenght_chunks"]) + ".png",
        dpi=300,
    )
    plt.savefig(config["PATH"] + "geometric_harmonics_h_" + str(config["DATA"]["gh_lenght_chunks"]) + ".pdf")
    plt.close()

    predictions_test = interp_c(V_test)

    fig = plt.figure(figsize=(10, 10))
    axs = []
    for i in range(1, 5):
        ax = fig.add_subplot(2, 2, i, aspect="equal")
        axs.append(ax)
        ax.scatter(c_chunks_train_test[:, i - 1], predictions_test[:, i - 1], s=7)
        ax.plot(
            np.linspace(
                np.min(c_chunks_train_test[:, i - 1]),
                np.max(c_chunks_train_test[:, i - 1]),
                10,
            ),
            np.linspace(
                np.min(c_chunks_train_test[:, i - 1]),
                np.max(c_chunks_train_test[:, i - 1]),
                10,
            ),
            "k",
        )
    axs[0].set_xlabel("True $h^{(1)}$")
    axs[0].set_ylabel("Predicted $h^{(1)}$")
    axs[1].set_xlabel("True $h^{(2)}$")
    axs[1].set_ylabel("Predicted $h^{(2)}$")
    axs[2].set_xlabel("True $h^{(3)}$")
    axs[2].set_ylabel("Predicted $h^{(3)}$")
    axs[3].set_xlabel("True $h^{(4)}$")
    axs[3].set_ylabel("Predicted $h^{(4)}$")
    plt.tight_layout()
    plt.savefig(
        config["PATH"] + "geometric_harmonics_test_h_" + str(config["DATA"]["gh_lenght_chunks"]) + ".png",
        dpi=300,
    )
    plt.savefig(config["PATH"] + "geometric_harmonics_test_h_" + str(config["DATA"]["gh_lenght_chunks"]) + ".pdf")
    plt.close()

    nystrom_data = np.empty((x_chunks_test.shape[0], 2))

    for i in tqdm(range(x_chunks_test.shape[0])):
        nystrom_data[i, :] = nystrom(x_chunks_test[i, :].reshape(1, -1), V, x_chunks_train, D, eps).T

    fig = plt.figure()  # figsize=(7, 7))
    # ax = fig.add_subplot(111)
    plt.scatter(V[:, 0], V[:, 1], c="k", s=5)
    plt.scatter(
        nystrom_data[:, 0],
        nystrom_data[:, 1],
        c=COLORS[1],
        marker="x",
        s=2,
        label="Restricted with Nyström",
    )
    plt.legend()
    plt.xlabel(r"$\phi_1$")
    plt.ylabel(r"$\phi_2$")
    plt.savefig(
        config["PATH"] + "nystrom_test_" + str(config["DATA"]["gh_lenght_chunks"]) + ".png",
        dpi=300,
    )
    plt.savefig(config["PATH"] + "nystrom_test_" + str(config["DATA"]["gh_lenght_chunks"]) + ".pdf")
    plt.close()

    predictions_validation = interp_c(nystrom_data[:, :])

    np.save(config["PATH"] + "plotting_c_chunks_test.npy", c_chunks_test)
    np.save(config["PATH"] + "plotting_predictions_validation.npy", predictions_validation)

    # Test integration
    idx = 0
    # config["DATA"]["max_warmup"] = 100
    initial_test_chunk = dataset_test.x[
        idx,
        config["DATA"]["max_warmup"] - config["DATA"]["gh_lenght_chunks"] : config["DATA"]["max_warmup"],
    ]
    nystrom_test_chunk = nystrom(initial_test_chunk.reshape(1, -1), V, x_chunks_train, D, eps).T
    h_pred = interp_c(nystrom_test_chunk)
    print(
        c_data_test[
            idx,
            config["DATA"]["max_warmup"] - config["DATA"]["initial_set_off"] - config["DATA"]["gh_lenght_chunks"],
        ]
        - h_pred
    )

    Ttotal = 400
    trajectory, _ = model.integrate(
        torch.tensor(initial_test_chunk[:1], dtype=torch.get_default_dtype()).to(model.device),
        T=Ttotal - 1,
        h0=torch.tensor(h_pred, dtype=torch.get_default_dtype()).to(model.device).unsqueeze(0),
    )

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(dataset_test.x[idx, config["DATA"]["max_warmup"]-config["DATA"]
    #                        ["gh_lenght_chunks"]:config["DATA"]["max_warmup"]+Ttotal])
    # ax.plot(trajectory)
    # ax.set_xlabel('')
    # ax.set_ylabel('')
    # # plt.savefig('')
    # plt.show()

    warmup_length_list = [config["DATA"]["gh_lenght_chunks"]]
    initial_condition = torch.tensor(dataset_test.x[idx], dtype=torch.get_default_dtype())

    start_point = config["DATA"]["max_warmup"]

    warmup_length = warmup_length_list[0]
    trajectory_w_warmup, _ = model.integrate(initial_condition[start_point - warmup_length : start_point], Ttotal - 1)

    np.save(config["PATH"] + "gh_trajectory_w_warmup.npy", trajectory_w_warmup)
    np.save(config["PATH"] + "gh_trajectory.npy", trajectory)

    plot_integration_w_gh(dataset_test, path="data/")


if __name__ == "__main__":
    main()

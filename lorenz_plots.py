import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import torch

from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist, pdist

from dm import diffusion_maps, geometric_harmonics

from dmaps_utils import create_geometric_harmonics_lorenz, nystrom, get_hidden_states, create_chunks, dmaps
from lorenz.config import config
from lorenz.datasets import LorenzParallelDataset
from plot_utils import load_model

torch.set_default_dtype(config["TRAINING"]["dtype"])

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
CMAP = "plasma"


# Load data
dataset_train = LorenzParallelDataset(
    config["DATA"]["n_train"], config["DATA"]["l_trajectories"], config["DATA"]["parameters"], True
)
dataset_val = LorenzParallelDataset(
    config["DATA"]["n_val"], config["DATA"]["l_trajectories"], config["DATA"]["parameters"], True
)
dataset_test = LorenzParallelDataset(
    config["DATA"]["n_test"], config["DATA"]["l_trajectories_test"], config["DATA"]["parameters"], True
)

# Load model
model = load_model(dataset_train, dataset_val, config)


# Do geometric harmonics
c_train = get_hidden_states(dataset_train, model)
c_test = get_hidden_states(dataset_test, model)

x_data_train = dataset_train.input_data[:, config["GH"]["initial_set_off"] :]
c_data_train = c_train[:, config["GH"]["initial_set_off"] :]

x_data_test = dataset_test.input_data[:, config["GH"]["initial_set_off"] :]
c_data_test = c_test[:, config["GH"]["initial_set_off"] :]

x_chunks_train = create_chunks(
    x_data_train,
    config["GH"]["max_n_transients"],
    config["GH"]["gh_lenght_chunks"],
    config["GH"]["shift_betw_chunks"],
)
c_chunks_train = create_chunks(
    c_data_train,
    config["GH"]["max_n_transients"],
    config["GH"]["gh_lenght_chunks"],
    config["GH"]["shift_betw_chunks"],
)
c_chunks_train = c_chunks_train[:, 0, :]

x_chunks_test = create_chunks(
    x_data_test,
    config["GH"]["max_n_transients"],
    config["GH"]["gh_lenght_chunks"],
    int(config["GH"]["shift_betw_chunks"]),
)
c_chunks_test = create_chunks(
    c_data_test,
    config["GH"]["max_n_transients"],
    config["GH"]["gh_lenght_chunks"],
    int(config["GH"]["shift_betw_chunks"]),
)
c_chunks_test = c_chunks_test[:, 0, :]

# Diffusion maps on input data
D, V, eps = dmaps(x_chunks_train, return_eps=True)

for i in range(2, 6):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(V[:, 1], V[:, i])
    ax.set_xlabel('')
    ax.set_ylabel('')
    # plt.savefig('')
    plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(V[:, 1], V[:, 4], V[:, 5])
ax.set_xlabel('')
ax.set_ylabel('')
# plt.savefig('')
plt.show()

V = V[:, [1, 4, 5]]
D = D[[1, 4, 5]]

print("Creating geometric harmonics.")
V_train, V_test, c_chunks_train_train, c_chunks_train_test = train_test_split(
    V, c_chunks_train, random_state=np.random.seed(10), train_size=8 / 10
)

pw = pdist(V_train, "euclidean")
eps_GH = np.median(pw**2) * 0.05

config["GH"]["gh_num_eigenpairs"] = 200
GH = diffusion_maps.SparseDiffusionMaps(
    points=V_train,
    epsilon=eps_GH,
    num_eigenpairs=config["GH"]["gh_num_eigenpairs"],
    cut_off=np.inf,
    renormalization=0,
    normalize_kernel=False,
)
print("Interpolation function.")
interp_c = geometric_harmonics.GeometricHarmonicsInterpolator(
    points=V_train, epsilon=None, values=c_chunks_train_train, diffusion_maps=GH
)

def plot_predictions():
    predictions_train = interp_c(V_train)

    fig = plt.figure(figsize=(10, 10))
    axs = []
    for i in range(1, 5):
        ax = fig.add_subplot(2, 2, i, aspect='equal')
        axs.append(ax)
        ax.scatter(c_chunks_train_train[:, i-1], predictions_train[:, i-1], s=7)
        ax.plot(np.linspace(np.min(c_chunks_train_train[:, i-1]),
                            np.max(c_chunks_train_train[:, i-1]), 10),
                np.linspace(np.min(c_chunks_train_train[:, i-1]),
                            np.max(c_chunks_train_train[:, i-1]), 10), 'k')
    axs[0].set_xlabel('True $h^{(1)}$')
    axs[0].set_ylabel('Predicted $h^{(1)}$')
    axs[1].set_xlabel('True $h^{(2)}$')
    axs[1].set_ylabel('Predicted $h^{(2)}$')
    axs[2].set_xlabel('True $h^{(3)}$')
    axs[2].set_ylabel('Predicted $h^{(3)}$')
    axs[3].set_xlabel('True $h^{(4)}$')
    axs[3].set_ylabel('Predicted $h^{(4)}$')
    plt.tight_layout()
    plt.savefig(config["PATH"]+'geometric_harmonics_h_' +
                str(config["GH"]["gh_lenght_chunks"])+'.png', dpi=300)
    plt.savefig(config["PATH"]+'geometric_harmonics_h_' +
                str(config["GH"]["gh_lenght_chunks"])+'.pdf')
    plt.show()

    predictions_test = interp_c(V_test)

    fig = plt.figure(figsize=(10, 10))
    axs = []
    for i in range(1, 5):
        ax = fig.add_subplot(2, 2, i, aspect='equal')
        axs.append(ax)
        ax.scatter(c_chunks_train_test[:, i-1], predictions_test[:, i-1], s=7)
        ax.plot(np.linspace(np.min(c_chunks_train_test[:, i-1]),
                            np.max(c_chunks_train_test[:, i-1]), 10),
                np.linspace(np.min(c_chunks_train_test[:, i-1]),
                            np.max(c_chunks_train_test[:, i-1]), 10), 'k')
    axs[0].set_xlabel('True $h^{(1)}$')
    axs[0].set_ylabel('Predicted $h^{(1)}$')
    axs[1].set_xlabel('True $h^{(2)}$')
    axs[1].set_ylabel('Predicted $h^{(2)}$')
    axs[2].set_xlabel('True $h^{(3)}$')
    axs[2].set_ylabel('Predicted $h^{(3)}$')
    axs[3].set_xlabel('True $h^{(4)}$')
    axs[3].set_ylabel('Predicted $h^{(4)}$')
    plt.tight_layout()
    plt.savefig(config["PATH"]+'geometric_harmonics_test_h_' +
                str(config["GH"]["gh_lenght_chunks"])+'.png', dpi=300)
    plt.savefig(config["PATH"]+'geometric_harmonics_test_h_' +
                str(config["GH"]["gh_lenght_chunks"])+'.pdf')
    plt.show()

plot_predictions()



# # Create geometric harmonics
# V, D, eps, x_chunks_train, c_chunks_train, interp_c = create_geometric_harmonics_lorenz(
#     dataset_train, dataset_test, config, model
# )

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dataset_test.tt[:-1], dataset_test.input_data[0], "-")
ax.plot(dataset_test.tt[:-1], dataset_test.input_data[1], "-")
ax.plot(dataset_test.tt[:-1], dataset_test.input_data[2], "-")
ax.set_xlabel("")
ax.set_ylabel("")
# plt.savefig('')
plt.show()

idx = 0

start_point = config["DATA"]["max_warmup"]
warmup_length_list = [config["DATA"]["max_warmup"]]
warmup_length = warmup_length_list[0]

initial_condition = torch.tensor(dataset_test.input_data[idx], dtype=torch.get_default_dtype())
trajectory_w_warmup_initial, _ = model.integrate(
    initial_condition[start_point - warmup_length : start_point], len(dataset_test.tt) - start_point - 1
)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dataset_test.tt[:-1], dataset_test.input_data[idx], "-", label="Actual", color="red")
ax.plot(dataset_test.tt, trajectory_w_warmup_initial, "-", label="prediction")
ax.axvline(x=dataset_test.tt[start_point], color="k")
ax.set_xlabel(r"$t$")
# plt.savefig('')
plt.show()

initial_test_chunk = dataset_test.input_data[
    idx, config["DATA"]["max_warmup"] - config["GH"]["gh_lenght_chunks"] : config["DATA"]["max_warmup"]
]

nystrom_test_chunk = nystrom(initial_test_chunk.reshape(1, -1), V, x_chunks_train, D, eps).T

h_pred = interp_c(nystrom_test_chunk)
trajectory_gh, _ = model.integrate(
    torch.tensor(initial_test_chunk[:1], dtype=torch.get_default_dtype()).to(model.device),
    T=len(dataset_test.tt) - config["DATA"]["max_warmup"] + config["GH"]["gh_lenght_chunks"] - 2,
    h0=torch.tensor(h_pred, dtype=torch.get_default_dtype()).to(model.device).unsqueeze(0),
)

# warmup_length_list = [config["DATA"]["gh_lenght_chunks"]]
# initial_condition = torch.tensor(dataset_test.input_data[idx], dtype=torch.get_default_dtype())
# start_point = config["DATA"]["max_warmup"]
# warmup_length = warmup_length_list[0]
# trajectory_w_warmup, _ = model.integrate(initial_condition[start_point - warmup_length : start_point], len(dataset_test.tt) - warmup_length - 1)

trajectory_w_warmup, _ = model.integrate(
    torch.tensor(initial_test_chunk), len(dataset_test.tt) - config["DATA"]["max_warmup"] - 1
)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dataset_test.tt[:-1], dataset_test.input_data[idx], label="Actual", color="red")
# ax.plot(dataset_test.tt[, trajectory_w_warmup, label='prediction with warmup')
ax.plot(
    dataset_test.tt[config["DATA"]["max_warmup"] - config["GH"]["gh_lenght_chunks"] :],
    trajectory_w_warmup,
    label="prediction with warmup",
)
ax.plot(
    dataset_test.tt[config["DATA"]["max_warmup"] - config["GH"]["gh_lenght_chunks"] :],
    trajectory_gh,
    label="prediction with GH",
)
ax.set_xlabel("")
ax.set_ylabel("")
# plt.savefig('')
plt.legend()
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(V[:, 0], V[:, 1], c=c_chunks_train[:, 0], cmap=CMAP)
ax.set_xlabel("")
ax.set_ylabel("")
# plt.savefig('')
plt.show()

t_series = [0, 1, 2]
cm_to_inch = 0.39370079
fig = plt.figure(figsize=(15.8 * cm_to_inch, 15.8 * cm_to_inch))
ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=1)
for series in t_series[1:]:
    ax1.plot(dataset_train.input_data[series], dataset_train.output_data[series], "-")
ax1.set_xlabel(r"$u$", labelpad=0)
ax1.set_ylabel(r"$v$", labelpad=0)
ax2 = plt.subplot2grid((4, 2), (0, 1), colspan=1)
ax2.scatter(V[:, 0], V[:, 1], c=c_chunks_train[:, 0], cmap=CMAP, s=3)
ax2.set_xlabel(r"$\phi_1$", labelpad=-3)
ax2.set_ylabel(r"$\phi_2$", labelpad=-3)
ax3 = plt.subplot2grid((4, 2), (1, 0), colspan=2)
for series in t_series[1:]:
    ax3.plot(dataset_train.tt[:-1], dataset_train.input_data[series], "-x", markersize=3)
ax3.set_xlabel("$t$", labelpad=0)
ax3.set_ylabel("$u$")
ax3.set_xlim((dataset_train.tt[0], dataset_train.tt[-1]))
ax4 = plt.subplot2grid((4, 2), (2, 0), colspan=2)
ax4.plot(dataset_test.tt[:-1], dataset_test.input_data[idx], "-x", markersize=3, label="Actual", color="red")
ax4.plot(dataset_test.tt, trajectory_w_warmup_initial, "-x", markersize=3, label="Autonomous", color="blue")
ax4.plot(
    dataset_test.tt[: config["DATA"]["max_warmup"]],
    dataset_test.input_data[idx, : config["DATA"]["max_warmup"]],
    "-x",
    markersize=3,
    label="Driven",
    color="green",
)
ax4.axvline(x=dataset_test.tt[config["DATA"]["max_warmup"]], color="k")
ax4.set_xlabel("$t$", labelpad=0)
ax4.set_ylabel("$u$")
ax4.set_xlim((dataset_test.tt[0], dataset_test.tt[-1]))
plt.legend(fontsize=8)
ax5 = plt.subplot2grid((4, 2), (3, 0), colspan=2)
ax5.plot(
    dataset_test.tt[config["DATA"]["max_warmup"] : -1],
    dataset_test.input_data[idx, config["DATA"]["max_warmup"] :],
    "-x",
    markersize=3,
    label="Actual",
    color="red",
)
ax5.plot(
    dataset_test.tt[config["DATA"]["max_warmup"] - config["GH"]["gh_lenght_chunks"] :],
    trajectory_w_warmup,
    "-x",
    markersize=3,
    label="Autonomous with 5 steps warmup",
    color="blue",
)
ax5.plot(
    dataset_test.tt[config["DATA"]["max_warmup"] - config["GH"]["gh_lenght_chunks"] : config["DATA"]["max_warmup"]],
    trajectory_w_warmup[: config["GH"]["gh_lenght_chunks"]],
    "-x",
    markersize=3,
    label="Driven",
    color="green",
)
ax5.plot(
    dataset_test.tt[config["DATA"]["max_warmup"] - config["GH"]["gh_lenght_chunks"] :],
    trajectory_gh,
    "-x",
    markersize=3,
    label="Autonomous with cold start",
    color="orange",
)
ax5.set_xlabel("$t$", labelpad=0)
ax5.set_ylabel("$u$")
ax5.axvline(x=dataset_test.tt[config["DATA"]["max_warmup"]], color="k")
ax5.set_xlim((dataset_test.tt[0], dataset_test.tt[-1]))
plt.legend(fontsize=8)
plt.subplots_adjust(bottom=0.08, left=0.09, top=0.97, right=0.97, hspace=0.35, wspace=0.3)
ax1.text(-0.23, 1.0, "a", transform=ax1.transAxes, size=12, weight="bold")
ax2.text(-0.23, 1.0, "b", transform=ax2.transAxes, size=12, weight="bold")
ax3.text(-0.1, 1.1, "c", transform=ax3.transAxes, size=12, weight="bold")
ax4.text(-0.1, 1.1, "d", transform=ax4.transAxes, size=12, weight="bold")
ax5.text(-0.1, 1.1, "e", transform=ax5.transAxes, size=12, weight="bold")
plt.savefig("fig/figure_lorenz.pdf")
plt.savefig("fig/figure_lorenz.png", dpi=400)
plt.show()

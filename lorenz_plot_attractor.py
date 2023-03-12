import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import torch

from pydszoo import lorenz  # pylint: disable=E0401

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
    config["DATA"]["n_val"], config["DATA"]["l_trajectories"], config["DATA"]["parameters"], False
)
dataset_test = LorenzParallelDataset(
    config["DATA"]["n_test"], config["DATA"]["l_trajectories_test"], config["DATA"]["parameters"], False
)

def create_initial_condition() -> np.ndarray:
    """Create initial conditions for Lorenz system.

    Returns:
        Numpy array with values [u, v, w]
    """
    return np.array((np.random.randn(), np.random.randn(), np.random.randn()))


initial_condition = create_initial_condition()
length = 2000
trajectory = lorenz.integrate(initial_condition, np.linspace(0, length/25, length), config["DATA"]["parameters"])


# Load model
model = load_model(dataset_train, dataset_val, config)

prediction = model.integrate(
    torch.tensor([[np.random.randn()]]), length
)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.linspace(0, length/25, length), trajectory[:, 0], label='lorenz')
ax.plot(np.linspace(0, length/25, length), prediction[0][1:-1], label='reservoir')
ax.set_xlabel('t')
ax.set_ylabel('u')
plt.legend()
plt.xlim((0, length/25))
plt.savefig("fig/figure_lorenz_trajectories.pdf")
plt.savefig("fig/figure_lorenz_trajectories.png", dpi=400)
plt.show()

delay = 3
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(trajectory[delay:, 0], trajectory[:-delay, 0])
ax1.plot(trajectory[50:][delay:, 0], trajectory[50:][:-delay, 0])
ax1.set_title('Lorenz')
ax1.set_xlabel(r'$x_{t}$')
ax1.set_ylabel(r'$x_{t-3}$')
ax2 = fig.add_subplot(122)
pred_tmp = prediction[0][1:-1]
ax2.plot(pred_tmp[delay:], pred_tmp[:-delay])
ax2.plot(pred_tmp[50:][delay:], pred_tmp[50:][:-delay])
ax2.set_title('Reservoir')
ax2.set_xlabel(r'$x_{t}$')
ax2.set_ylabel(r'$x_{t-3}$')
ax1.set_xlim((-35, 35))
ax2.set_xlim((-35, 35))
ax1.set_ylim((-35, 35))
ax2.set_ylim((-35, 35))
plt.subplots_adjust(hspace=0.35, wspace=0.3)
plt.savefig("fig/figure_lorenz_attractor.pdf")
plt.savefig("fig/figure_lorenz_attractor.png", dpi=400)
plt.show()

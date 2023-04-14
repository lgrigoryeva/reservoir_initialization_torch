"""Calculate Lyapunov exponents for Lorenz system and reservoir."""
import numpy as np
import matplotlib.pyplot as plt
from pydszoo import lorenz  # pylint: disable=E0401
import nolds


def create_initial_condition() -> np.ndarray:
    """Create initial conditions for Lorenz system.

    Returns:
        Numpy array with values [u, v, w]
    """
    return np.array((np.random.randn(), np.random.randn(), np.random.randn()))


parameters = {}
parameters["sigma"] = 10.0
parameters["rho"] = 28.0
parameters["beta"] = 8.0 / 3.0

expected_dim = 3 - 2 * (parameters["sigma"] + parameters["beta"] + 1) / (parameters["sigma"] + 1 + np.sqrt((parameters["sigma"]-1) ** 2 + 4 * parameters["sigma"] * parameters["rho"]))
print(f"Expected dimension: {expected_dim}")

trajectory = lorenz.integrate(create_initial_condition(), np.linspace(0, 50, 5001), parameters)


def lorenz_step(variables, dt, parameters):
    return lorenz.integrate(variables, np.linspace(0, dt, 2), parameters)[-1, :]


def max_lyap(pars,
             ic,
             stepper,
             d0=1e-8,
             nstep=1e4,
             dTarr=np.linspace(0.04, 0.16, 4),
             plot_flag=False):
    """Calculate the maximal lyapunov exponent."""
    # Set initial conditons (must be on the attractor)
    y = ic

    # Dimension of the system:
    N = len(y)

    # Initial perturbation
    deltax_0 = np.random.randn(N)
    deltax_0 = deltax_0 / np.linalg.norm(deltax_0) * d0
    deltax_t = np.zeros((int(nstep), N))
    deltax_t[0] = deltax_0

    yper = y + deltax_0

    # Arrays in which the mean and variance of the log |deltax_t|/|deltax_0| are saved
    lyapspec = []
    lyapspecvar = []
    # The time ranges which are going to be calculated
    for dT in dTarr:
        print(str(dT))
        for idx in np.arange(1, int(nstep)):
            # Run trajectory with perturbation Ainit as initial condition
            yper = stepper(yper, dT, pars)
            y = stepper(y, dT, pars)
            # Calculate the distance
            deltax_t[idx] = np.linalg.norm(yper - y)
            # Calculate the direction of new perturbation along the direction of maximal divergence
            deltax_0 = d0 * (yper - y) / np.linalg.norm(yper - y)
            # Set the new initial perturbed initial condition
            yper = y + deltax_0
        # Add mean and variance of the trajectory distances
        lyapspec.append(
            np.mean(np.log(np.abs(np.linalg.norm(deltax_t[int(nstep/2):], axis=1) / np.linalg.norm(deltax_t[0])))))
        lyapspecvar.append(
            np.var(np.log(np.abs(np.linalg.norm(deltax_t[int(nstep/2):], axis=1) / np.linalg.norm(deltax_t[0])))))

    m, b = np.polyfit(dTarr, lyapspec, 1)
    if plot_flag:
        plt.errorbar(dTarr, lyapspec, yerr=lyapspecvar, fmt='o')
        plt.plot(dTarr, m * dTarr + b)
        plt.xlabel(r"$dT$")
        plt.ylabel(r"$\log |\delta x(dT) | / |\delta x(o) |$")
        plt.show()
    return m

lyapunov_exponent = max_lyap(parameters, trajectory[-1], lorenz_step, plot_flag=True, nstep=2e4)
print(lyapunov_exponent)




len_trajectories = 25000
time_array = np.linspace(0, len_trajectories / 25, len_trajectories + 1) + 20

initial_condition = create_initial_condition()
trajectory = lorenz.integrate(initial_condition, time_array, parameters)[:, 0]
# nolds.lyap_r(trajectory, emb_dim=5, tau=0.04)
nolds.lyap_r(trajectory, tau=time_array[1]-time_array[0],
             debug_plot=True, min_tsep=1000, emb_dim=5, lag=1, trajectory_len=4)
nolds.lyap_r(trajectory, tau=time_array[1]-time_array[0], debug_plot=True, trajectory_len=8)
# nolds.lyap_r(trajectory, min_tsep=1000, emb_dim=5, tau=0.01, lag=5)
# nolds.lyap_r(trajectory, min_tsep=1000, emb_dim=5, tau=0.04, lag=3)
# nolds.lyap_r(trajectory, tau=0.1, emb_dim=5)

len_trajectories = 25000
trajectory = lorenz.integrate(create_initial_condition(),
                              np.linspace(0, 0.01*len_trajectories, len_trajectories+1),
                              parameters)[:, 0]
nolds.lyap_r(trajectory, min_tsep=1000, emb_dim=5, tau=0.01, lag=5, debug_plot=True)

dt = 2e-3
len_trajectories = 20000
trajectory = lorenz.integrate(create_initial_condition(),
                              np.linspace(0, dt*len_trajectories, len_trajectories+1),
                              parameters)[:, 0]
nolds.lyap_r(trajectory, min_tsep=1000, emb_dim=5, tau=dt, lag=5, debug_plot=True, trajectory_len=40)
nolds.lyap_r(trajectory, emb_dim=5, tau=dt, debug_plot=True, trajectory_len=30, fit='poly',
             fit_offset=10)

# min_tsep=1000, emb_dim=5, tau=0.01, lag=5
# 0.8438 -0.0409 -14.5011

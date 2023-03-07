"""Functions to integrate Lorenz system."""
import numpy as np
from scipy.integrate import solve_ivp


def create_initial_conditions(ic="standard"):
    """Specify initial conditions."""
    if ic == "standard":
        x0 = 10.0
        y0 = 1.0
        z0 = 0.0
    elif ic == "random":
        # x0 = 20.0*np.random.random()
        # y0 = 2.0*np.random.random()
        # z0 = 1.0*np.random.random()
        x0 = 0.2 * np.random.random()
        y0 = 0.2 * np.random.random()
        z0 = 0.2 * np.random.random()
        x0 = 0.1
        y0 = 0.1
        z0 = 0.1
    return np.array([x0, y0, z0])


def f(t, y, arg_sigma, arg_rho, arg_beta):
    """Temporal evolution of the oscillator."""
    xprime = arg_sigma * (y[1] - y[0])
    yprime = -y[0] * y[2] + arg_rho * y[0] - y[1]
    zprime = y[0] * y[1] - arg_beta * y[2]
    return np.array([xprime, yprime, zprime])


def jac(t, y, arg_sigma, arg_rho, arg_beta):
    """Calculate Jacobian."""
    J = np.array(
        [
            [-arg_sigma, arg_sigma, 0],
            [arg_rho - y[2], -1, -y[0]],
            [y[1], y[0], -arg_beta],
        ]
    )
    return J


def integrate(
    sigma=10.0,
    rho=28.0,
    beta=8.0 / 3.0,
    tmin=50,
    dt=1e-2,
    T=20000,
    ic="standard",
    Ainit=0,
):
    """Integrate Lorenz oscillator."""
    # Write the parameters into a dictionary for future use.
    tmax = T * dt + tmin
    Adict = dict()
    Adict["sigma"] = sigma
    Adict["rho"] = rho
    Adict["beta"] = beta
    Adict["tmin"] = tmin
    Adict["tmax"] = tmax
    Adict["dt"] = dt
    Adict["T"] = T
    Adict["ic"] = ic

    if ic == "manual":
        if Ainit.shape[0] != 2:
            raise ValueError("Initial data must have two real entries.")
        y0 = Ainit
    else:
        y0 = create_initial_conditions(ic)

    Adict["init"] = y0

    tt = np.linspace(tmin, tmax, Adict["T"] + 1)

    Adict["tt"] = tt

    sol = solve_ivp(f, [0, tt[-1]], y0, t_eval=tt, args=[sigma, rho, beta])
    sol.y = sol.y.T

    Adict["data"] = np.array(sol.y)
    print("Integration done!")
    return Adict

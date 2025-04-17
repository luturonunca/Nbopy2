import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from math import pi


def compute_fE(alpha, beta, gamma, Rmin=1e-2, Rmax=100, NE=1000, NR=1000, epsrel=1e-6):
    """
    Compute the distribution function f(E) for a dimensionless Zhao profile
    with given (alpha, beta, gamma) using Eddington inversion.

    Returns a dictionary with:
        R       : radius grid
        rho     : density profile (normalized)
        psi     : relative potential
        E       : energy grid
        fE      : distribution function values on E
    """

    # Zhao profile with rho0 = 1, rs = 1
    def rho(r):
        return r**(-gamma) * (1 + r**alpha)**((gamma - beta) / alpha)

    # Normalization
    M_tot = 4 * pi * quad(lambda r: r**2 * rho(r), 0, np.inf)[0]

    # Grids
    R = np.logspace(np.log10(Rmin), np.log10(Rmax), NR)
    rho_r = rho(R) / M_tot

    # Cumulative mass profile
    def M_r(r):
        return 4 * pi * quad(lambda rp: rp**2 * rho(rp), 0, r)[0] / M_tot
    M_r_vec = np.vectorize(M_r)
    Mcum = M_r_vec(R)

    # Gravitational potential
    def Phi(r):
        I1 = quad(lambda rp: rp**2 * rho(rp), 0, r)[0]
        I2 = quad(lambda rp: rp * rho(rp), r, np.inf)[0]
        return -4 * pi * (I1 / r + I2) / M_tot
    Phi_vec = np.vectorize(Phi)
    psi = -Phi_vec(R)  # relative potential

    # Interpolate psi and rho
    psi_interp = interp1d(R, psi, bounds_error=False, fill_value=(psi[0], 0.0))
    rho_interp = interp1d(R, rho_r, bounds_error=False, fill_value=(rho_r[0], 0.0))

    # Derivatives d^2 rho / d psi^2
    dndr = np.gradient(rho_r, psi)
    d2nd2r = np.gradient(dndr, psi)

    # Interpolator for second derivative
    d2nd2p_interp = interp1d(psi, d2nd2r, kind='linear', fill_value=0, bounds_error=False)

    # Build energy grid
    maxE = psi[0]           # most bound
    minE = maxE / NE
    E = np.linspace(minE, maxE, NE)

    # Compute f(E)
    fE = []
    for eps in E:
        def integrand(p):
            if p >= eps:
                return 0.0
            return d2nd2p_interp(p) / np.sqrt(eps - p)

        integral, _ = quad(integrand, psi[-1], eps, epsrel=epsrel, limit=100)
        f_eps = (1 / (np.sqrt(8) * pi**2)) * integral
        fE.append(f_eps)

    fE = np.array(fE)

    return {
        "R": R,
        "rho": rho_r,
        "psi": psi,
        "E": E,
        "fE": fE,
        "M_r": Mcum
    }

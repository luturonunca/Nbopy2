import numpy as np
from scipy.integrate import quad
from math import pi

G = 4.30091e-6  # gravitational constant in kpc^3 / (Msun s^2)

def compute_distribution_function(rho_func, Rmin=1e-2, Rmax=100.0, NE=10000, NR=10000, epsrel=1e-6):
    R = np.logspace(np.log10(Rmin), np.log10(Rmax), NR)
    
    # Unnormalized density and mass
    rho_r = rho_func(R)
    M = 4 * pi * quad(lambda r: r**2 * rho_func(r), 0., np.inf)[0]

    Mr  = np.vectorize(lambda x: 4*pi * quad(lambda r: r**2 * rho_func(r)/M, 0., x, epsrel=epsrel)[0])
    Phi = np.vectorize(lambda x: -4*pi * (1/x * quad(lambda r: r**2 * rho_func(r)/M, 0., x, epsrel=epsrel)[0] + 
                                          quad(lambda r: r * rho_func(r)/M, x, np.inf, epsrel=epsrel)[0]))

    psi = -Phi(R)
    nu  = rho_r / M
    mcum = Mr(R)

    # Gradients
    dndp   = np.gradient(nu, psi)
    d2nd2p = np.gradient(dndp, psi)

    # Build DF(E)
    fE = []
    maxE = psi[0]
    minE = maxE / NE
    E = np.linspace(minE, maxE, NE)
    
    for e in E:
        integrand = lambda p: np.interp(p, psi[::-1], d2nd2p[::-1]) / np.sqrt(e - p)
        integral, _ = quad(integrand, 0, e, epsrel=epsrel)
        fval = 1./(np.sqrt(8) * pi**2) * integral
        fE.append(fval)
    
    fE = np.array(fE)
    
    return {
        "R": R,
        "rho_r": rho_r,
        "psi_r": psi,
        "nu": nu,
        "E": E,
        "fE": fE,
        "mcum": mcum,
        "dndp": dndp,
        "d2nd2p": d2nd2p,
        "Mtot": M
    }

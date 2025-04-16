import numpy as np

def rho(r, rho0, rs, alpha, beta, gamma):
    x = r / rs
    return rho0 / (x**gamma * (1 + x**alpha)**((beta - gamma)/alpha))

def mass(r, rho0, rs, alpha, beta, gamma):
    from scipy.integrate import quad
    integrand = lambda rp: rho(rp, rho0, rs, alpha, beta, gamma) * rp**2
    result = np.array([quad(integrand, 0, ri)[0] for ri in r])
    return 4 * np.pi * result


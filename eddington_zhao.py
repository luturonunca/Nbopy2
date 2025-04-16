import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson, cumulative_trapezoid

G = 4.30091e-6  # kpc^3 / (Msun s^2)

def compute_fE(r_grid, rho_func):
    # 1. Density
    rho_r = rho_func(r_grid)

    # 2. Enclosed mass
    mass_r = cumulative_trapezoid(4 * np.pi * rho_r * r_grid**2, r_grid, initial=0)

    # 3. Potential Phi(r)
    phi_r = np.zeros_like(r_grid)
    for i in range(len(r_grid)):
        r_slice = r_grid[i:]
        m_slice = mass_r[i:]
        phi_r[i] = -G * simpson(m_slice / r_slice**2, r_slice)

    # 4. Relative potential Ψ(r) = Φ(R_max) - Φ(r)
    psi_r = phi_r[-1] - phi_r

    # 5. Derivatives
    sorted_idx = np.argsort(psi_r)
    psi_sorted = psi_r[sorted_idx]
    rho_sorted = rho_r[sorted_idx]

    dpsi = np.gradient(psi_sorted)
    drho = np.gradient(rho_sorted, dpsi)
    d2rho = np.gradient(drho, dpsi)

    d2rho_interp = interp1d(psi_sorted, d2rho, bounds_error=False, fill_value=0)

    # 6. Compute f(E)
    eps_grid = np.linspace(psi_sorted[0], psi_sorted[-1], 200)
    f_eps = []
    for eps in eps_grid:
        psi_vals = psi_sorted[psi_sorted < eps]
        if len(psi_vals) < 2:
            f_eps.append(0.0)
            continue
        integrand_vals = d2rho_interp(psi_vals) / np.sqrt(eps - psi_vals)
        integral = simpson(integrand_vals, psi_vals)
        f_val = (1 / (np.sqrt(8) * np.pi**2)) * integral
        f_eps.append(f_val)

    return eps_grid, np.array(f_eps), psi_r, rho_r, phi_r, mass_r


import numpy as np
from scipy.special import hyp2f1
from scipy.integrate import cumulative_trapezoid
from math import pi, sqrt

def compute_fE(alpha, beta, gamma, *,
               Rmin=1e-2, Rmax=100,
               NR=1000, NE=1000, chunk=2048):
    """Fast Eddington Inversion for a Zhao (α,β,γ) halo,
       normalised to the finite mass M(<Rmax).

       Works for any β > 2 (outer density slope steeper than r^{-2})."""

    # ----------  density on a logarithmic grid  ----------
    R   = np.logspace(np.log10(Rmin), np.log10(Rmax), NR)
    rho = R**(-gamma) * (1 + R**alpha)**((gamma-beta)/alpha)

    # ----------  cumulative mass up to Rmax (vectorised trapz) ----------
    M_cum = 4*pi * cumulative_trapezoid(rho*R**2, R, initial=0)
    M_tot = M_cum[-1]          #   *finite* mass inside Rmax
    rho  /= M_tot              #   normalise
    M_cum/= M_tot

    # ----------  potential with outer cut at Rmax ----------
    # Φ(r) = - M(Rmax)/Rmax  - ∫_r^{Rmax} M(x)/x² dx
    integrand = (M_cum / R**2)[::-1]
    int_tail  = cumulative_trapezoid(integrand, R[::-1], initial=0)[::-1]
    phi       = -(M_cum[-1]/R[-1] + int_tail)

    # choose ψ = Φ(R) - Φ(Rmax)  (⇒ ψ(Rmax)=0, ψ>0 inside)
    psi = phi - phi[-1]

    # ----------  d²ρ/dψ²  ----------
    dρdψ   = np.gradient(rho, psi, edge_order=2)
    d2ρdψ2 = np.gradient(dρdψ, psi, edge_order=2)
    Δψ     = np.gradient(psi)

    # ----------  Abel inversion in memory‑friendly chunks ----------
    const = 1.0 / (sqrt(8)*pi**2)
    Emax, Emin = psi[0], psi[0]/NE
    E = np.linspace(Emin, Emax, NE)

    fE = np.empty_like(E)
    for i in range(0, NE, chunk):                # ≤800 MB → ≤160 MB
        sl  = slice(i, min(i+chunk, NE))
        Δ   = E[sl, None] - psi[None, :]
        K   = np.where(Δ > 0, d2ρdψ2 / np.sqrt(Δ), 0.0)
        #fE[sl] = const * np.sum(K * Δψ, axis=1)
        fE[sl] = const * np.sum(K * np.abs(Δψ), axis=1)
        #fE[sl] = const * np.trapz(K, psi, axis=1)

    return dict(R=R, rho=rho, psi=psi, M_r=M_cum,
                E=E, fE=fE)

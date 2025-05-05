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



def sample_particles_from_fE(df, N=10_000, *, seed=42,
                             batch=32_768):
    """
    Draw N particles from an isotropic DF produced by `compute_fE_fast`.

    The algorithm is identical to your original:
      • invert M(<r) to get radii
      • rejection–sample E in [0, ψ(r)] with envelope f(ψ)
      • isotropic angles for x,y,z and v

    ---------
    Arguments
    ---------
    df      : dict returned by compute_fE / compute_fE_fast
    N       : number of particles
    seed    : RNG seed (uses NumPy Generator)
    batch   : candidates drawn per iteration – raise for higher throughput
    """
    rng = np.random.default_rng(seed)

    # pull arrays once (no attribute look‑ups in the hot loop)
    R, psi, Mcum = df["R"], df["psi"], df["M_r"]
    Egrid, fEgrid = df["E"], df["fE"]

    # quick linear interpolators   (≈20× faster than SciPy interp1d here)
    inv_M = lambda u: np.interp(u, Mcum, R)          # M⁻¹(u) → r
    psi_of_r = lambda r: np.interp(r, R, psi)        # r → ψ
    fE_of_E = lambda e: np.interp(e, Egrid, fEgrid, left=0.0, right=0.0)

    # envelope for rejection sampling – local maximum f(ψ) is tight
    envelope = lambda psi_r: 1.05 * fE_of_E(psi_r)   # 5 % safety margin

    pos = np.empty((N, 3))
    vel = np.empty((N, 3))
    filled = 0                                       # how many accepted so far

    while filled < N:
        # ---------- draw a batch of candidate radii ----------
        u = rng.random(batch)                        # uniform [0,1]
        r = inv_M(u)                                 # inverse‑CDF
        psi_r = psi_of_r(r)

        # ---------- candidate energies and acceptance ----------
        eps  = rng.random(batch) * psi_r             # uniform in [0, ψ(r)]
        fval = fE_of_E(eps)
        acc  = rng.random(batch) < fval / envelope(psi_r)
        if not acc.any():                 # unlikely but cheap guard
            continue

        # keep only accepted ones
        r, psi_r, eps = r[acc], psi_r[acc], eps[acc]
        m = len(r)
        if filled + m > N:                 # clip the final over‑fill
            r        = r[: N-filled]
            psi_r    = psi_r[: N-filled]
            eps      = eps[: N-filled]
            m        = len(r)

        # ---------- positions ----------
        cosθ = 2.0 * rng.random(m) - 1.0
        sinθ = np.sqrt(1.0 - cosθ**2)
        φ    = 2*np.pi * rng.random(m)
        pos[filled:filled+m, 0] = r * sinθ * np.cos(φ)
        pos[filled:filled+m, 1] = r * sinθ * np.sin(φ)
        pos[filled:filled+m, 2] = r * cosθ

        # ---------- velocities ----------
        v   = np.sqrt(2.0 * (psi_r - eps))
        cosθv = 2.0 * rng.random(m) - 1.0
        sinθv = np.sqrt(1.0 - cosθv**2)
        φv    = 2*np.pi * rng.random(m)
        vel[filled:filled+m, 0] = v * sinθv * np.cos(φv)
        vel[filled:filled+m, 1] = v * sinθv * np.sin(φv)
        vel[filled:filled+m, 2] = v * cosθv

        filled += m

    return pos, vel



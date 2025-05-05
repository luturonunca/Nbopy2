# ------------------------------------------------------------
#  Rodionov iterative shrink‑wrap  (uses your existing functions)
# ------------------------------------------------------------
import numpy as np
from math import pi
import unsiotools.simulations.cfalcon as falcon
# import the two helpers exactly as you already do
from .Edd_inversion import compute_fE, sample_particles_from_fE


def rodionov_triaxial_ic(*,
        alpha=1, beta=3, gamma=1,
        axis_final=(1.3, 1.0, 0.7),      # (a,b,c)
        N=2_00_000, iters=20,
        mode="adiabatic",                # or "rescale"
        eps=0.02, dt_frac=0.08,
        seed=1):
    """
    Build an equilibrium triaxial Zhao halo by the Rodionov iterative method.

    mode="adiabatic"  – deform axes a little each iteration (robust)
    mode="rescale"    – jump straight to final axes, then rescale v each wrap
    """

    rng = np.random.default_rng(seed)
    cf  = falcon.CFalcon()

    # ---------- spherical DF → initial sample ----------
    df          = compute_fE(alpha, beta, gamma)
    pos, vel    = sample_particles_from_fE(df, N, seed=seed)
    pos32       = np.ascontiguousarray(pos, dtype=np.float32).ravel()
    mass32      = np.full(N, 1.0/N, dtype=np.float32)

    # helper: draw fresh positions on ellipsoid with given axes
    def fresh_positions(ax, ay, az):
        r_sorted = np.sort(np.linalg.norm(pos, axis=1))
        cos = 2*rng.random(N)-1
        sin = np.sqrt(1-cos**2)
        phi = 2*pi*rng.random(N)
        x = r_sorted*sin*np.cos(phi)*ax
        y = r_sorted*sin*np.sin(phi)*ay
        z = r_sorted*cos             *az
        pos[:]  = np.stack((x,y,z),1)
        pos32[:] = pos.ravel()

    # start with spherical positions
    fresh_positions(1.0, 1.0, 1.0)

    # ---------- iterate ----------
    for k in range(iters):

        # --- live evolution Δt ---
        ok, acc, _ = cf.getGravity(pos32, mass32, eps, G=1.0)
        acc = acc.reshape(N,3)
        dt  = dt_frac * np.mean(np.linalg.norm(pos,axis=1)**1.5)
        vel += 0.5*acc*dt
        pos += vel*dt
        pos32[:] = pos.ravel()
        ok, acc, phi_old = cf.getGravity(pos32, mass32, eps, G=1.0)
        vel += 0.5*acc.reshape(N,3)*dt
        phi_old = phi_old.astype(float)   # keep for rescale mode

        # --- shrink‑wrap positions ---
        if mode == "adiabatic":
            f = (k+1) / iters              # 0→1 ramp
            a, b, c = 1 + f*(np.array(axis_final)-1)
            fresh_positions(a, b, c)
        elif mode == "rescale":
            a, b, c = axis_final
            # --- stretch positions without changing particle order ---
            pos[:, 0] *= a
            pos[:, 1] *= b
            pos[:, 2] *= c
            pos32[:] = pos.ravel()

            # --- new potential at stretched positions ---
            ok, _, phi_new = cf.getGravity(pos32, mass32, eps, G=1.0)
            phi_new = phi_new.astype(float)

            # --- energy‑conserving velocity rescale (per particle) ---
            v2_old = np.sum(vel**2, axis=1)
            v2_new = v2_old + 2*(phi_old - phi_new)         # keeps E = const
            v2_new = np.maximum(v2_new, 0.0)                # numerical floor
            vel *= np.sqrt(v2_new / v2_old)[:, None] 

        else:
            raise ValueError("mode must be 'adiabatic' or 'rescale'")

        # --- diagnostics every 3 steps ---
        if k % 3 == 0 or k == iters-1:
            ok, _, phi_chk = cf.getGravity(pos32, mass32, eps, G=1.0)
            Q = np.sum(np.sum(vel**2,1)) / np.sum(-phi_chk)
            print(f"iter {k:2d}: 2T/|U| = {Q:.3f}")

    return pos.copy(), vel.copy()

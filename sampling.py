import numpy as np

def sample_particles(df_data, N=1000000, seed=667408, epsrel=1e-6):
    from scipy.interpolate import interp1d
    from scipy.integrate import quad

    np.random.seed(seed)

    R      = df_data["R"]
    psi_r  = df_data["psi_r"]
    fE     = df_data["fE"]
    Egrid  = df_data["E"]
    mcum   = df_data["mcum"]
    Mtot   = df_data["Mtot"]

    # Precompute useful interpolators
    psi_of_r  = interp1d(R, psi_r, bounds_error=False, fill_value=0)
    max_psi_r = np.max(psi_r)
    R_of_psi  = interp1d(psi_r[::-1], R[::-1], bounds_error=False, fill_value=(R[0], R[-1]))
    fE_interp = interp1d(Egrid, fE, bounds_error=False, fill_value=0)

    # Phase space volume element
    def dPdr(e, r):
        return np.sqrt(2 * (psi_of_r(r) - e)) * r**2 if e < psi_of_r(r) else 0

    def PLikelihood(e, r):
        return fE_interp(e) * dPdr(e, r)

    def max_plikelihood(r):
        e_vals = Egrid[Egrid < psi_of_r(r)]
        return 1.1 * np.max([PLikelihood(e, r) for e in e_vals]) if len(e_vals) else 0

    # Inverse transform sampling for radius
    Nin = int(N * 4 * np.pi * quad(lambda r: r**2 * df_data["rho_r"][np.argmin(abs(R - r))] / Mtot, 0., R[0])[0])
    Nout = int(N * (1 - 4 * np.pi * quad(lambda r: r**2 * df_data["rho_r"][np.argmin(abs(R - r))] / Mtot, 0., R[-1])[0]))

    randMcum = Nin / N + (1. - (Nin + Nout)/N) * np.random.rand(N)
    randR = np.interp(randMcum, mcum, R)

    # Sample energies via rejection
    psiR = psi_of_r(randR)
    randE = np.random.rand(N) * psiR
    rhoE = np.array([PLikelihood(e, r) for e, r in zip(randE, randR)])
    randY = np.random.rand(N) * np.array([max_plikelihood(r) for r in randR])

    # Rejection loop
    ok = randY <= rhoE
    while not np.all(ok):
        retry = np.where(~ok)[0]
        randE[retry] = np.random.rand(len(retry)) * psi_of_r(randR[retry])
        rhoE[retry] = [PLikelihood(e, r) for e, r in zip(randE[retry], randR[retry])]
        randY[retry] = np.random.rand(len(retry)) * np.array([max_plikelihood(r) for r in randR[retry]])
        ok = randY <= rhoE

    acceptedR = randR
    acceptedE = randE

    # Draw angles
    theta_r = np.arccos(2 * np.random.rand(N) - 1)
    phi_r = 2 * np.pi * np.random.rand(N)

    theta_v = np.arccos(2 * np.random.rand(N) - 1)
    phi_v = 2 * np.pi * np.random.rand(N)
    v = np.sqrt(2 * (psi_of_r(acceptedR) - acceptedE))

    # Positions
    x = acceptedR * np.sin(theta_r) * np.cos(phi_r)
    y = acceptedR * np.sin(theta_r) * np.sin(phi_r)
    z = acceptedR * np.cos(theta_r)

    # Velocities
    vx = v * np.sin(theta_v) * np.cos(phi_v)
    vy = v * np.sin(theta_v) * np.sin(phi_v)
    vz = v * np.cos(theta_v)

    return np.column_stack((x, y, z)), np.column_stack((vx, vy, vz))

import os

os.environ["MPLBACKEND"] = "Agg"

import matplotlib
import numpy as np
from scipy.optimize import newton

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Physical parameters and initial condition from the assignment.
gamma = 1.4
t_end = 0.14
x_range = (-1.0, 1.0)
W_L = np.array([0.445, 0.311, 8.928], dtype=float)  # [rho, m, E]
W_R = np.array([0.5, 0.0, 1.4275], dtype=float)


def primitive_to_conservative(rho, u, p, gam=gamma):
    """Convert primitive variables (rho, u, p) to conserved variables."""
    m = rho * u
    E = p / (gam - 1.0) + 0.5 * rho * u**2
    return np.array([rho, m, E], dtype=float)


def conservative_to_primitive(rho, m, E, gam=gamma):
    """Convert conserved variables (rho, m, E) to primitive variables."""
    rho = max(float(rho), 1e-12)
    u = m / rho
    p = (gam - 1.0) * (E - 0.5 * rho * u**2)
    return rho, u, max(p, 1e-12)


def sound_speed(rho, p, gam=gamma):
    """Calculate sound speed."""
    return np.sqrt(max(gam * p / rho, 1e-12))


def get_primitive(w):
    """Vectorized conserved-to-primitive conversion."""
    rho = np.maximum(w[0], 1e-12)
    u = w[1] / rho
    p = (gamma - 1.0) * (w[2] - 0.5 * rho * u**2)
    return rho, u, np.maximum(p, 1e-12)


def get_flux(w):
    rho, u, p = get_primitive(w)
    return np.array([w[1], w[1] * u + p, (w[2] + p) * u])


def pressure_function(p, rho_k, p_k, gam=gamma):
    """Toro exact solver helper f_k(p) and derivative."""
    a_k = sound_speed(rho_k, p_k, gam)
    if p > p_k:
        A_k = 2.0 / ((gam + 1.0) * rho_k)
        B_k = (gam - 1.0) / (gam + 1.0) * p_k
        sqrt_term = np.sqrt(A_k / (p + B_k))
        f_k = (p - p_k) * sqrt_term
        df_k = sqrt_term * (1.0 - 0.5 * (p - p_k) / (p + B_k))
    else:
        p_ratio = max(p / p_k, 1e-12)
        expo = (gam - 1.0) / (2.0 * gam)
        f_k = 2.0 * a_k / (gam - 1.0) * (p_ratio**expo - 1.0)
        df_k = (1.0 / (rho_k * a_k)) * p_ratio ** (-(gam + 1.0) / (2.0 * gam))
    return f_k, df_k


def find_star_state(rhoL, uL, pL, rhoR, uR, pR, gam=gamma):
    """Find star-region pressure and velocity."""
    aL = sound_speed(rhoL, pL, gam)
    aR = sound_speed(rhoR, pR, gam)
    p_pv = 0.5 * (pL + pR) - 0.125 * (uR - uL) * (rhoL + rhoR) * (aL + aR)
    p_guess = max(p_pv, 1e-8)

    def phi(p):
        fL, _ = pressure_function(p, rhoL, pL, gam)
        fR, _ = pressure_function(p, rhoR, pR, gam)
        return fL + fR + (uR - uL)

    def dphi(p):
        _, dfL = pressure_function(p, rhoL, pL, gam)
        _, dfR = pressure_function(p, rhoR, pR, gam)
        return dfL + dfR

    try:
        p_star = max(newton(phi, p_guess, fprime=dphi, tol=1e-12, maxiter=100), 1e-8)
    except RuntimeError:
        p_star = p_guess

    fL, _ = pressure_function(p_star, rhoL, pL, gam)
    fR, _ = pressure_function(p_star, rhoR, pR, gam)
    u_star = 0.5 * (uL + uR + fR - fL)
    return p_star, u_star


def sample_solution(xi, p_star, u_star, rhoL, uL, pL, rhoR, uR, pR, gam=gamma):
    """Sample the exact solution at xi = x / t."""
    aL = sound_speed(rhoL, pL, gam)
    aR = sound_speed(rhoR, pR, gam)
    g1 = (gam - 1.0) / (2.0 * gam)
    g2 = (gam - 1.0) / (gam + 1.0)

    if p_star > pL:
        rhoL_star = rhoL * (p_star / pL + g2) / (g2 * p_star / pL + 1.0)
    else:
        rhoL_star = rhoL * (p_star / pL) ** (1.0 / gam)

    if p_star > pR:
        rhoR_star = rhoR * (p_star / pR + g2) / (g2 * p_star / pR + 1.0)
    else:
        rhoR_star = rhoR * (p_star / pR) ** (1.0 / gam)

    if xi <= u_star:
        if p_star > pL:
            SL = uL - aL * np.sqrt((gam + 1.0) / (2.0 * gam) * p_star / pL + (gam - 1.0) / (2.0 * gam))
            if xi <= SL:
                return rhoL, uL, pL
            return rhoL_star, u_star, p_star

        SHL = uL - aL
        STL = u_star - aL * (p_star / pL) ** g1
        if xi <= SHL:
            return rhoL, uL, pL
        if xi >= STL:
            return rhoL_star, u_star, p_star

        u = 2.0 / (gam + 1.0) * (aL + 0.5 * (gam - 1.0) * uL + xi)
        a = 2.0 / (gam + 1.0) * (aL + 0.5 * (gam - 1.0) * (uL - xi))
        rho = rhoL * (a / aL) ** (2.0 / (gam - 1.0))
        p = pL * (a / aL) ** (2.0 * gam / (gam - 1.0))
        return rho, u, p

    if p_star > pR:
        SR = uR + aR * np.sqrt((gam + 1.0) / (2.0 * gam) * p_star / pR + (gam - 1.0) / (2.0 * gam))
        if xi >= SR:
            return rhoR, uR, pR
        return rhoR_star, u_star, p_star

    SHR = uR + aR
    STR = u_star + aR * (p_star / pR) ** g1
    if xi >= SHR:
        return rhoR, uR, pR
    if xi <= STR:
        return rhoR_star, u_star, p_star

    u = 2.0 / (gam + 1.0) * (-aR + 0.5 * (gam - 1.0) * uR + xi)
    a = 2.0 / (gam + 1.0) * (aR - 0.5 * (gam - 1.0) * (uR - xi))
    rho = rhoR * (a / aR) ** (2.0 / (gam - 1.0))
    p = pR * (a / aR) ** (2.0 * gam / (gam - 1.0))
    return rho, u, p


def exact_riemann_solution(qL, qR, x, t, gam=gamma):
    """Compute the exact solution of the Euler Riemann problem."""
    rhoL, uL, pL = conservative_to_primitive(qL[0], qL[1], qL[2], gam)
    rhoR, uR, pR = conservative_to_primitive(qR[0], qR[1], qR[2], gam)
    p_star, u_star = find_star_state(rhoL, uL, pL, rhoR, uR, pR, gam)

    rho = np.zeros_like(x)
    u = np.zeros_like(x)
    p = np.zeros_like(x)

    for i, xi in enumerate(x / t):
        rho[i], u[i], p[i] = sample_solution(xi, p_star, u_star, rhoL, uL, pL, rhoR, uR, pR, gam)

    return rho, u, p


def lax_wendroff(w, dx, dt):
    """Richtmyer two-step Lax-Wendroff scheme."""
    nx = w.shape[1]
    w_new = w.copy()
    w_half = np.zeros((3, nx - 1))

    for i in range(nx - 1):
        f_i = get_flux(w[:, i])
        f_ip1 = get_flux(w[:, i + 1])
        w_half[:, i] = 0.5 * (w[:, i] + w[:, i + 1]) - (dt / (2.0 * dx)) * (f_ip1 - f_i)

    for i in range(1, nx - 1):
        f_half_i = get_flux(w_half[:, i])
        f_half_im1 = get_flux(w_half[:, i - 1])
        w_new[:, i] = w[:, i] - (dt / dx) * (f_half_i - f_half_im1)

    return w_new


def tvd_van_leer(w, dx, dt):
    """van Leer flux-vector-splitting scheme."""
    nx = w.shape[1]
    w_new = w.copy()
    f_p = np.zeros_like(w)
    f_m = np.zeros_like(w)

    for j in range(nx):
        rho, u, p = get_primitive(w[:, j])
        c = np.sqrt(gamma * p / rho)
        ma = u / c
        if ma >= 1.0:
            f_p[:, j] = get_flux(w[:, j])
            f_m[:, j] = 0.0
        elif ma <= -1.0:
            f_p[:, j] = 0.0
            f_m[:, j] = get_flux(w[:, j])
        else:
            fac_p = rho * c / 4.0 * (ma + 1.0) ** 2
            fac_m = -rho * c / 4.0 * (ma - 1.0) ** 2
            f_p[:, j] = fac_p * np.array(
                [
                    1.0,
                    2.0 * c / gamma * (1.0 + 0.5 * (gamma - 1.0) * ma),
                    2.0 * c**2 / (gamma**2 - 1.0) * (1.0 + 0.5 * (gamma - 1.0) * ma) ** 2,
                ]
            )
            f_m[:, j] = fac_m * np.array(
                [
                    1.0,
                    2.0 * c / gamma * (-1.0 + 0.5 * (gamma - 1.0) * ma),
                    2.0 * c**2 / (gamma**2 - 1.0) * (1.0 - 0.5 * (gamma - 1.0) * ma) ** 2,
                ]
            )

    for j in range(1, nx - 1):
        w_new[:, j] = w[:, j] - (dt / dx) * (f_m[:, j + 1] - f_m[:, j] + f_p[:, j] - f_p[:, j - 1])

    return w_new


def upwind_characteristic(w, dx, dt):
    """First-order characteristic upwind scheme in primitive variables."""
    nx = w.shape[1]
    rho, u, p = get_primitive(w)
    U_old = np.vstack([rho, u, p])
    U_new = U_old.copy()

    for i in range(1, nx - 1):
        rho_i = max(U_old[0, i], 1e-12)
        u_i = U_old[1, i]
        p_i = max(U_old[2, i], 1e-12)
        a_i = np.sqrt(gamma * p_i / rho_i)

        R = np.array(
            [
                [1.0, 1.0, 1.0],
                [-a_i / rho_i, 0.0, a_i / rho_i],
                [a_i**2, 0.0, a_i**2],
            ]
        )
        L = np.linalg.inv(R)
        lam = np.array([u_i - a_i, u_i, u_i + a_i])
        A_plus = R @ np.diag(np.maximum(lam, 0.0)) @ L
        A_minus = R @ np.diag(np.minimum(lam, 0.0)) @ L

        dU_minus = U_old[:, i] - U_old[:, i - 1]
        dU_plus = U_old[:, i + 1] - U_old[:, i]
        U_new[:, i] = U_old[:, i] - (dt / dx) * (A_plus @ dU_minus + A_minus @ dU_plus)

    U_new[0] = np.maximum(U_new[0], 1e-12)
    U_new[2] = np.maximum(U_new[2], 1e-12)

    w_new = np.zeros_like(w)
    for i in range(nx):
        w_new[:, i] = primitive_to_conservative(U_new[0, i], U_new[1, i], U_new[2, i], gamma)
    return w_new


def run_simulation(nx, cfl, method, t_target=t_end):
    dx = (x_range[1] - x_range[0]) / (nx - 1)
    x = np.linspace(x_range[0], x_range[1], nx)
    w = np.zeros((3, nx))
    w[:, x < 0.0] = W_L.reshape(3, 1)
    w[:, x >= 0.0] = W_R.reshape(3, 1)

    t = 0.0
    while t < t_target:
        rho, u, p = get_primitive(w)
        c = np.sqrt(gamma * p / rho)
        dt = cfl * dx / np.max(np.abs(u) + c)
        if t + dt > t_target:
            dt = t_target - t

        if method == "lw":
            w = lax_wendroff(w, dx, dt)
        elif method == "tvd":
            w = tvd_van_leer(w, dx, dt)
        elif method == "upwind":
            w = upwind_characteristic(w, dx, dt)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Keep far-field states fixed because the waves never reach the boundaries.
        w[:, 0] = W_L
        w[:, -1] = W_R
        w[0] = np.maximum(w[0], 1e-12)
        w[2] = np.maximum(w[2], 1e-12)
        t += dt

    return x, w


def plot_results(x, w, title, save_filename=None, show_exact=True):
    rho, u, p = get_primitive(w)
    m = rho * u
    E = w[2]

    rho_init = np.where(x < 0.0, W_L[0], W_R[0])
    m_init = np.where(x < 0.0, W_L[1], W_R[1])
    E_init = np.where(x < 0.0, W_L[2], W_R[2])

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    fig.suptitle(title)

    exact_data = None
    x_exact = None
    if show_exact:
        x_exact = np.linspace(x_range[0], x_range[1], 2000)
        rho_exact, u_exact, p_exact = exact_riemann_solution(W_L, W_R, x_exact, t_end, gamma)
        exact_data = [
            rho_exact,
            rho_exact * u_exact,
            p_exact / (gamma - 1.0) + 0.5 * rho_exact * u_exact**2,
        ]

    labels = ["Density rho", "Momentum m=rho u", "Energy E"]
    initial_data = [rho_init, m_init, E_init]
    numerical_data = [rho, m, E]

    for idx in range(3):
        axes[idx].plot(x, initial_data[idx], color="lightskyblue", linewidth=1.5, label="Initial value")
        if exact_data is not None:
            axes[idx].plot(x_exact, exact_data[idx], color="tab:orange", linewidth=1.8, label="Exact")
        axes[idx].plot(
            x,
            numerical_data[idx],
            color="tab:green",
            marker="o",
            markersize=2.8,
            linewidth=1.0,
            label="Numerical",
        )
        axes[idx].set_ylabel(labels[idx])
        axes[idx].grid(True, alpha=0.4)
        axes[idx].legend(loc="best")

    axes[-1].set_xlabel("x")
    plt.tight_layout()

    if save_filename is not None:
        fig.savefig(save_filename, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_filename}")
    plt.close(fig)


def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    x1, w1 = run_simulation(300, 0.22, "lw")
    plot_results(
        x1,
        w1,
        "Fig 1: Lax-Wendroff (300 grids, CFL=0.22)",
        save_filename=os.path.join(output_dir, "fig1_lax_wendroff.png"),
    )

    x2, w2 = run_simulation(300, 0.10, "tvd")
    plot_results(
        x2,
        w2,
        "Fig 2: TVD (300 grids, CFL=0.1)",
        save_filename=os.path.join(output_dir, "fig2_tvd_300.png"),
    )

    x3, w3 = run_simulation(600, 0.10, "tvd")
    plot_results(
        x3,
        w3,
        "Fig 3: TVD (600 grids, CFL=0.1)",
        save_filename=os.path.join(output_dir, "fig3_tvd_600.png"),
    )

    x4, w4 = run_simulation(200, 0.01, "upwind")
    plot_results(
        x4,
        w4,
        "Fig 4: Upwind (200 grids, CFL=0.01)",
        save_filename=os.path.join(output_dir, "fig4_upwind.png"),
    )


if __name__ == "__main__":
    main()

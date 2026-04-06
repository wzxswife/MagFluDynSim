
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.size'] = 15
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 11

# =========================
# 1. 初值
# =========================
def initial_burgers(x):
    u = np.zeros_like(x)

    u[x < -0.8] = 1.8

    mask1 = (-0.8 <= x) & (x < -0.3)
    u[mask1] = 1.4 + 0.4 * np.cos(2.0 * np.pi * (x[mask1] + 0.8))

    mask2 = (-0.3 <= x) & (x < 0.0)
    u[mask2] = 1.0

    u[x >= 0.0] = 1.8
    return u


# =========================
# 2. minmod
# =========================
def minmod_array(a, b):
    out = np.zeros_like(a)
    mask = (a * b) > 0
    out[mask] = np.where(np.abs(a[mask]) < np.abs(b[mask]), a[mask], b[mask])
    return out


# =========================
# 3. 一阶迎风格式
#    u^{n+1}_j = u^n_j - 1/2 * dt/dx * ((u_j)^2 - (u_{j-1})^2)
# =========================
def burgers_upwind_step(u, dt, dx):
    unew = u.copy()
    C = dt / dx

    unew[1:] = u[1:] - 0.5 * C * (u[1:]**2 - u[:-1]**2)

    # 固定边界
    unew[0] = 1.8
    unew[-1] = 1.8
    return unew


def solve_burgers_upwind(u0, dt, dx, nsteps):
    u = u0.copy()
    for _ in range(nsteps):
        u = burgers_upwind_step(u, dt, dx)
    return u


# =========================
# 4.  Minmod 格式
# =========================
def burgers_minmod_step(u, dt, dx):
    unew = u.copy()
    C = dt / dx
    nx = len(u)

    sigma = np.zeros(nx)
    sigma[1:-1] = minmod_array(u[1:-1] - u[:-2], u[2:] - u[1:-1])

    term1 = 0.5 * (u[1:-1]**2 - u[:-2]**2)

    term2 = 0.25 * (u[2:]**2 - u[1:-1]**2) * (
        1.0 - 0.5 * C * (u[1:-1] + u[2:])
    ) * sigma[1:-1]

    term3 = 0.25 * (u[1:-1]**2 - u[:-2]**2) * (
        1.0 - 0.5 * C * (u[1:-1] + u[:-2])
    ) * sigma[:-2]

    unew[1:-1] = u[1:-1] - C * (term1 + term2 - term3)

    # 固定边界
    unew[0] = 1.8
    unew[-1] = 1.8
    return unew


def solve_burgers_minmod(u0, dt, dx, nsteps):
    u = u0.copy()
    for _ in range(nsteps):
        u = burgers_minmod_step(u, dt, dx)
    return u


# =========================
# 5. Burgers 熵解（作为 Exact solution）
# =========================
def primitive_u0(y):
    u = initial_burgers(y)
    U = np.zeros_like(y)

    # 累积积分，假设 y 单调递增
    dy = np.diff(y)
    U[1:] = np.cumsum(0.5 * (u[:-1] + u[1:]) * dy)

    # 让 U(0)=0
    idx0 = np.argmin(np.abs(y))
    U = U - U[idx0]
    return U


def exact_burgers_solution(x_eval, t, y_min=-3.0, y_max=3.0, ny=40000, chunk=200):
    if t == 0:
        return initial_burgers(x_eval)

    y = np.linspace(y_min, y_max, ny)
    U0 = primitive_u0(y)

    u_exact = np.zeros_like(x_eval)

    for i in range(0, len(x_eval), chunk):
        xs = x_eval[i:i + chunk]  # shape: (m,)
        # Phi shape: (ny, m)
        Phi = U0[:, None] + (xs[None, :] - y[:, None])**2 / (2.0 * t)
        idx = np.argmin(Phi, axis=0)
        y_star = y[idx]
        u_exact[i:i + chunk] = (xs - y_star) / t

    return u_exact


# =========================
# 6. 主程序
# =========================
def main():

    nx = 1000
    x = np.linspace(-1.0, 2.0, nx)
    dx = x[1] - x[0]
    u0 = initial_burgers(x)

    CFL = 0.95
    umax = np.max(np.abs(u0))
    dt_base = CFL * dx / umax

    times = [0.25, 0.5, 0.75, 1.0]

    fig, axes = plt.subplots(4, 2, figsize=(12, 15))
    axes = axes.ravel()

    x_exact = np.linspace(-1.0, 2.0, 1600)

    for i, t in enumerate(times):
        nsteps = int(np.ceil(t / dt_base))
        dt = t / nsteps

        u_upwind = solve_burgers_upwind(u0, dt, dx, nsteps)
        u_minmod = solve_burgers_minmod(u0, dt, dx, nsteps)
        u_exact = exact_burgers_solution(x_exact, t)

        ax1 = axes[2 * i]
        ax2 = axes[2 * i + 1]

        # 左列：Upwind
        ax1.plot(x, u0, ':', linewidth=1.9, color='dimgray', label='Initial value')
        ax1.plot(x_exact, u_exact, '-', linewidth=1.9, color='black', label='Exact solution')

        ax1.plot(
            x, u_upwind,
            linestyle='None',
            marker='o',
            markersize=4.9,
            markerfacecolor='none',
            markeredgewidth=0.8,
            color='royalblue',
            label='Numerical solution'
        )

        ax1.set_title(f'Upwind $C=0.95$ Time={t}', pad=6)
        ax1.set_xlim(-1.0, 2.0)
        ax1.set_ylim(1.0, 1.82)

        ax1.legend(loc='lower right', bbox_to_anchor=(0.985, 0.03), frameon=True)
        ax1.tick_params(direction='in')

        ax2.plot(x, u0, ':', linewidth=1.9, color='dimgray', label='Initial value')
        ax2.plot(x_exact, u_exact, '-', linewidth=1.9, color='black', label='Exact solution')

        ax2.plot(
            x, u_minmod,
            linestyle='None',
            marker='o',
            markersize=4.9,
            markerfacecolor='none',
            markeredgewidth=1.0,
            color='royalblue',
            label='Numerical solution'
        )

        ax2.set_title(f'Upwind minmod $C=0.95$ Time={t}', pad=6)
        ax2.set_xlim(-1.0, 2.0)
        ax2.set_ylim(1.0, 1.82)

        ax2.legend(loc='lower right', bbox_to_anchor=(0.985, 0.03), frameon=True)
        ax2.tick_params(direction='in')

        # 子图标号
        labels = [r'$(a)$', r'$(b)$', r'$(c)$', r'$(d)$', r'$(e)$', r'$(f)$', r'$(g)$', r'$(h)$']
        ax1.text(0.04, 0.78, labels[2 * i], transform=ax1.transAxes, fontsize=15)
        ax2.text(0.04, 0.78, labels[2 * i + 1], transform=ax2.transAxes, fontsize=15)

    plt.tight_layout()
    plt.savefig("Burgers.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("burgers_results.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    main()
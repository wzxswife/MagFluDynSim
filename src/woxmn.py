from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

"""
Create Your Own Constrained Transport Magnetohydrodynamics Simulation (With Python)
Philip Mocz (2023), @PMocz

Simulate the Orszag-Tang vortex MHD problem

Modified to output a series of result snapshots.
"""

import warnings
warnings.filterwarnings("ignore")

# GIF 输出（需 pip install imageio imageio-ffmpeg）
try:
    import imageio.v3 as iio

    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


# ============================================================
# 输出配置
# ============================================================

OUT_DIR = Path.cwd() / "orszagtang_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 快照时间点：覆盖 Orszag-Tang 涡旋从初始到湍流的完整演化
SNAPSHOT_TIMES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# 面板图精选时间点（从快照中挑选 6 个关键阶段）
PANEL_TIMES = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]
PANEL_LABELS = [
    r"$t=0.00\tau_A$（初始）",
    r"$t=0.10\tau_A$（涡旋变形）",
    r"$t=0.20\tau_A$（激波形成）",
    r"$t=0.30\tau_A$（激波作用）",
    r"$t=0.40\tau_A$（湍流发展）",
    r"$t=0.50\tau_A$（充分湍流）",
]

# 绘图风格
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

NAVY = "#0b2f78"
BLUE = "#2563eb"
RED = "#d7191c"
PURPLE = "#7a32b8"
TEXT = "#1f2937"
BORDER = "#b8c3d6"


# ============================================================
# 绘图辅助函数
# ============================================================

def style_ax(ax):
    """统一坐标轴样式"""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
        sp.set_linewidth(0.8)
    ax.tick_params(labelsize=8, colors=TEXT)


def save_density_frame(rho, t, out_dir):
    """
    保存单帧密度图。
    用与原始代码一致的 jet 色阶和色标范围 [0.06, 0.5]。
    """
    fig, ax = plt.subplots(figsize=(5.0, 4.8), dpi=200, facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.92, top=0.92, bottom=0.10)

    im = ax.imshow(
        rho.T,
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap="jet",
        vmin=0.06,
        vmax=0.5,
        interpolation="bilinear",
    )

    ax.set_title(rf"$t = {t:.2f}\,\tau_A$", fontsize=16, color=NAVY, fontweight="bold", pad=10)
    ax.set_xlabel(r"$x/L$", fontsize=12)
    ax.set_ylabel(r"$y/L$", fontsize=12)
    style_ax(ax)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\rho$", fontsize=12)
    cbar.ax.tick_params(labelsize=9)

    fig.savefig(out_dir / f"frame_t={t:.2f}.png", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: frame_t={t:.2f}.png")


def create_panel_figure(snapshots, out_dir):
    """
    创建多面板拼图，展示密度场的完整演化时序。
    """
    n_panels = len(PANEL_TIMES)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(3.2 * n_panels, 4.2),
        dpi=200,
        facecolor="white",
    )
    fig.subplots_adjust(left=0.04, right=0.94, top=0.76, bottom=0.12, wspace=0.08)

    fig.text(
        0.5, 0.94,
        r"Orszag-Tang 涡旋密度演化 $\rho(x,y,t)$",
        ha="center", va="center",
        fontsize=20, color=NAVY, fontweight="bold",
    )

    for i, t in enumerate(PANEL_TIMES):
        ax = axes[i]
        snap = snapshots[t]
        rho = snap["rho"]

        im = ax.imshow(
            rho.T,
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="jet",
            vmin=0.06,
            vmax=0.5,
            interpolation="bilinear",
        )

        ax.set_title(PANEL_LABELS[i], fontsize=11, color=TEXT, fontweight="bold", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        style_ax(ax)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.012)
    cbar.set_label(r"$\rho$", fontsize=13)
    cbar.ax.tick_params(labelsize=10)

    fig.savefig(out_dir / "panel_density_evolution.png", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: panel_density_evolution.png")


def save_animation_gif(out_dir, fps=4):
    """
    将已保存的帧 PNG 合成为 GIF 动图。
    需要 imageio 库：pip install imageio imageio-ffmpeg
    """
    if not HAS_IMAGEIO:
        print("  [GIF] 跳过：未安装 imageio (pip install imageio imageio-ffmpeg)")
        return

    frames = sorted(out_dir.glob("frame_t=*.png"))

    def _time_key(p):
        try:
            return float(p.stem.replace("frame_t=", ""))
        except ValueError:
            return 0.0

    frames.sort(key=_time_key)

    if len(frames) < 2:
        print("  [GIF] 跳过：帧数不足")
        return

    images = [iio.imread(f) for f in frames]
    gif_path = out_dir / "orszagtang_evolution.gif"

    iio.imwrite(gif_path, images, duration=1000 / fps, loop=0)
    print(f"  Saved: {gif_path.name} ({len(images)} frames @ {fps}FPS)")


def plot_diagnostics(t_hist, diag, out_dir):
    """
    绘制诊断量时间演化图。
    """
    t_arr = np.array(t_hist)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=200, facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.96, top=0.92, bottom=0.10, hspace=0.30, wspace=0.25)

    fig.text(
        0.5, 0.96,
        "Orszag-Tang 涡旋诊断量",
        ha="center", va="center",
        fontsize=20, color=NAVY, fontweight="bold",
    )

    # 总能量
    ax = axes[0, 0]
    ax.plot(t_arr, diag["Etotal"], color=NAVY, lw=2.0, label=r"$E_{\text{total}}$")
    ax.plot(t_arr, diag["Emag"], color=RED, lw=1.5, label=r"$E_{\text{mag}}$")
    ax.plot(t_arr, diag["Ekin"], color=BLUE, lw=1.5, label=r"$E_{\text{kin}}$")
    ax.plot(t_arr, diag["Eint"], color=PURPLE, lw=1.5, label=r"$E_{\text{int}}$")
    ax.set_xlabel(r"$t/\tau_A$", fontsize=13)
    ax.set_ylabel("Energy", fontsize=13)
    ax.set_title("能量演化", fontsize=14, color=NAVY, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)

    # 平均 |div B|
    ax = axes[0, 1]
    ax.semilogy(t_arr, diag["mean_divB"], color=RED, lw=2.0)
    ax.axhline(1e-15, color="gray", lw=0.8, ls="--", alpha=0.6, label=r"$10^{-15}$")
    ax.set_xlabel(r"$t/\tau_A$", fontsize=13)
    ax.set_ylabel(r"$\langle|\nabla\cdot\boldsymbol{B}|\rangle$", fontsize=13)
    ax.set_title(r"$\nabla\cdot\boldsymbol{B}$ 诊断", fontsize=14, color=NAVY, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)

    # 动能占比
    ax = axes[1, 0]
    ratio = np.array(diag["Ekin"]) / np.maximum(np.array(diag["Etotal"]), 1e-14)
    ax.plot(t_arr, ratio, color=BLUE, lw=2.0)
    ax.set_xlabel(r"$t/\tau_A$", fontsize=13)
    ax.set_ylabel(r"$E_{\text{kin}} / E_{\text{total}}$", fontsize=13)
    ax.set_title("动能占比", fontsize=14, color=NAVY, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)

    # 磁能占比
    ax = axes[1, 1]
    ratio = np.array(diag["Emag"]) / np.maximum(np.array(diag["Etotal"]), 1e-14)
    ax.plot(t_arr, ratio, color=RED, lw=2.0)
    ax.set_xlabel(r"$t/\tau_A$", fontsize=13)
    ax.set_ylabel(r"$E_{\text{mag}} / E_{\text{total}}$", fontsize=13)
    ax.set_title("磁能占比", fontsize=14, color=NAVY, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)

    fig.savefig(out_dir / "diagnostics.png", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: diagnostics.png")


def getCurl(Az, dx):
    """
    Calculate the discrete curl
        Az       is matrix of nodal z-component of magnetic potential
        dx       is the cell size
        bx       is matrix of cell face x-component magnetic-field
        by       is matrix of cell face y-component magnetic-field
    """
    # directions for np.roll()
    R = -1  # right/up
    L = 1  # left/down

    bx = (Az - np.roll(Az, L, axis=1)) / dx  # = d Az / d y
    by = -(Az - np.roll(Az, L, axis=0)) / dx  # =-d Az / d x

    return bx, by


def getDiv(bx, by, dx):
    """
    Calculate the discrete curl of each cell
    dx       is the cell size
        bx       is matrix of cell face x-component magnetic-field
        by       is matrix of cell face y-component magnetic-field
    """
    # directions for np.roll()
    R = -1  # right/up
    L = 1  # left/down

    divB = (bx - np.roll(bx, L, axis=0) + by - np.roll(by, L, axis=1)) / dx

    return divB


def getBavg(bx, by):
    """
    Calculate the volume-averaged magnetic field
        bx       is matrix of cell face x-component magnetic-field
        by       is matrix of cell face y-component magnetic-field
        Bx       is matrix of cell Bx
        By       is matrix of cell By
    """
    # directions for np.roll()
    R = -1  # right/up
    L = 1  # left/down

    Bx = 0.5 * (bx + np.roll(bx, L, axis=0))
    By = 0.5 * (by + np.roll(by, L, axis=1))

    return Bx, By


def getConserved(rho, vx, vy, P, Bx, By, gamma, vol):
    """
    Calculate the conserved variable from the primitive
        rho      is matrix of cell densities
        vx       is matrix of cell x-velocity
        vy       is matrix of cell y-velocity
        P        is matrix of cell Total pressures
        Bx       is matrix of cell Bx
        By       is matrix of cell By
        gamma    is ideal gas gamma
        vol      is cell volume
        Mass     is matrix of mass in cells
        Momx     is matrix of x-momentum in cells
        Momy     is matrix of y-momentum in cells
        Energy   is matrix of energy in cells
    """
    Mass = rho * vol
    Momx = rho * vx * vol
    Momy = rho * vy * vol
    Energy = (
        (P - 0.5 * (Bx**2 + By**2)) / (gamma - 1)
        + 0.5 * rho * (vx**2 + vy**2)
        + 0.5 * (Bx**2 + By**2)
    ) * vol

    return Mass, Momx, Momy, Energy


def getPrimitive(Mass, Momx, Momy, Energy, Bx, By, gamma, vol):
    """
    Calculate the primitive variable from the conservative
        Mass     is matrix of mass in cells
        Momx     is matrix of x-momentum in cells
        Momy     is matrix of y-momentum in cells
        Energy   is matrix of energy in cells
        Bx       is matrix of cell Bx
        By       is matrix of cell By
        gamma    is ideal gas gamma
        vol      is cell volume
        rho      is matrix of cell densities
        vx       is matrix of cell x-velocity
        vy       is matrix of cell y-velocity
        P        is matrix of cell Total pressures
    """
    rho = Mass / vol
    vx = Momx / rho / vol
    vy = Momy / rho / vol
    P = (Energy / vol - 0.5 * rho * (vx**2 + vy**2) - 0.5 * (Bx**2 + By**2)) * (
        gamma - 1
    ) + 0.5 * (Bx**2 + By**2)

    return rho, vx, vy, P


def getGradient(f, dx):
    """
    Calculate the gradients of a field
        f        is a matrix of the field
        dx       is the cell size
        f_dx     is a matrix of derivative of f in the x-direction
        f_dy     is a matrix of derivative of f in the y-direction
    """
    # directions for np.roll()
    R = -1  # right
    L = 1  # left

    f_dx = (np.roll(f, R, axis=0) - np.roll(f, L, axis=0)) / (2 * dx)
    f_dy = (np.roll(f, R, axis=1) - np.roll(f, L, axis=1)) / (2 * dx)

    return f_dx, f_dy


def slopeLimit(f, dx, f_dx, f_dy):
    """
    Apply slope limiter to slopes
        f        is a matrix of the field
        dx       is the cell size
        f_dx     is a matrix of derivative of f in the x-direction
        f_dy     is a matrix of derivative of f in the y-direction
    """
    # directions for np.roll()
    R = -1  # right
    L = 1  # left

    f_dx = (
        np.maximum(
            0.0,
            np.minimum(
                1.0, ((f - np.roll(f, L, axis=0)) / dx) / (f_dx + 1.0e-8 * (f_dx == 0))
            ),
        )
        * f_dx
    )
    f_dx = (
        np.maximum(
            0.0,
            np.minimum(
                1.0, (-(f - np.roll(f, R, axis=0)) / dx) / (f_dx + 1.0e-8 * (f_dx == 0))
            ),
        )
        * f_dx
    )
    f_dy = (
        np.maximum(
            0.0,
            np.minimum(
                1.0, ((f - np.roll(f, L, axis=1)) / dx) / (f_dy + 1.0e-8 * (f_dy == 0))
            ),
        )
        * f_dy
    )
    f_dy = (
        np.maximum(
            0.0,
            np.minimum(
                1.0, (-(f - np.roll(f, R, axis=1)) / dx) / (f_dy + 1.0e-8 * (f_dy == 0))
            ),
        )
        * f_dy
    )

    return f_dx, f_dy


def extrapolateInSpaceToFace(f, f_dx, f_dy, dx):
    """
    Calculate the gradients of a field
        f        is a matrix of the field
        f_dx     is a matrix of the field x-derivatives
        f_dy     is a matrix of the field y-derivatives
        dx       is the cell size
        f_XL     is a matrix of spatial-extrapolated values on `left' face along x-axis
        f_XR     is a matrix of spatial-extrapolated values on `right' face along x-axis
        f_YL     is a matrix of spatial-extrapolated values on `left' face along y-axis
        f_YR     is a matrix of spatial-extrapolated values on `right' face along y-axis
    """
    # directions for np.roll()
    R = -1  # right
    L = 1  # left

    f_XL = f - f_dx * dx / 2
    f_XL = np.roll(f_XL, R, axis=0)
    f_XR = f + f_dx * dx / 2

    f_YL = f - f_dy * dx / 2
    f_YL = np.roll(f_YL, R, axis=1)
    f_YR = f + f_dy * dx / 2

    return f_XL, f_XR, f_YL, f_YR


def applyFluxes(F, flux_F_X, flux_F_Y, dx, dt):
    """
    Apply fluxes to conserved variables
        F        is a matrix of the conserved variable field
        flux_F_X is a matrix of the x-dir fluxes
        flux_F_Y is a matrix of the y-dir fluxes
        dx       is the cell size
        dt       is the timestep
    """
    # directions for np.roll()
    R = -1  # right
    L = 1  # left

    # update solution
    F += -dt * dx * flux_F_X
    F += dt * dx * np.roll(flux_F_X, L, axis=0)
    F += -dt * dx * flux_F_Y
    F += dt * dx * np.roll(flux_F_Y, L, axis=1)

    return F


def constrainedTransport(bx, by, flux_By_X, flux_Bx_Y, dx, dt):
    """
    Apply fluxes to face-centered magnetic fields in a constrained transport manner
        bx        is matrix of cell face x-component magnetic-field
        by        is matrix of cell face y-component magnetic-field
        flux_By_X is a matrix of the x-dir fluxes of By
        flux_Bx_Y is a matrix of the y-dir fluxes of Bx
        dx        is the cell size
        dt        is the timestep
    """
    # directions for np.roll()
    R = -1  # right
    L = 1  # left

    # update solution
    # Ez at top right node of cell = avg of 4 fluxes
    Ez = 0.25 * (
        -flux_By_X
        - np.roll(flux_By_X, R, axis=1)
        + flux_Bx_Y
        + np.roll(flux_Bx_Y, R, axis=0)
    )
    dbx, dby = getCurl(-Ez, dx)

    bx += dt * dbx
    by += dt * dby

    return bx, by


def getFlux(
    rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, Bx_L, Bx_R, By_L, By_R, gamma
):
    """
    Calculate fluxed between 2 states with local Lax-Friedrichs/Rusanov rule
        rho_L        is a matrix of left-state  density
        rho_R        is a matrix of right-state density
        vx_L         is a matrix of left-state  x-velocity
        vx_R         is a matrix of right-state x-velocity
        vy_L         is a matrix of left-state  y-velocity
        vy_R         is a matrix of right-state y-velocity
        P_L          is a matrix of left-state  Total pressure
        P_R          is a matrix of right-state Total pressure
        Bx_L         is a matrix of left-state  x-magnetic-field
        Bx_R         is a matrix of right-state x-magnetic-field
        By_L         is a matrix of left-state  y-magnetic-field
        By_R         is a matrix of right-state y-magnetic-field
        gamma        is the ideal gas gamma
        flux_Mass    is the matrix of mass fluxes
        flux_Momx    is the matrix of x-momentum fluxes
        flux_Momy    is the matrix of y-momentum fluxes
        flux_Energy  is the matrix of energy fluxes
    """

    # left and right energies
    en_L = (
        (P_L - 0.5 * (Bx_L**2 + By_L**2)) / (gamma - 1)
        + 0.5 * rho_L * (vx_L**2 + vy_L**2)
        + 0.5 * (Bx_L**2 + By_L**2)
    )
    en_R = (
        (P_R - 0.5 * (Bx_R**2 + By_R**2)) / (gamma - 1)
        + 0.5 * rho_R * (vx_R**2 + vy_R**2)
        + 0.5 * (Bx_R**2 + By_R**2)
    )

    # compute star (averaged) states
    rho_star = 0.5 * (rho_L + rho_R)
    momx_star = 0.5 * (rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5 * (rho_L * vy_L + rho_R * vy_R)
    en_star = 0.5 * (en_L + en_R)
    Bx_star = 0.5 * (Bx_L + Bx_R)
    By_star = 0.5 * (By_L + By_R)

    P_star = (gamma - 1) * (
        en_star
        - 0.5 * (momx_star**2 + momy_star**2) / rho_star
        - 0.5 * (Bx_star**2 + By_star**2)
    ) + 0.5 * (Bx_star**2 + By_star**2)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass = momx_star
    flux_Momx = momx_star**2 / rho_star + P_star - Bx_star * Bx_star
    flux_Momy = momx_star * momy_star / rho_star - Bx_star * By_star
    flux_Energy = (en_star + P_star) * momx_star / rho_star - Bx_star * (
        Bx_star * momx_star + By_star * momy_star
    ) / rho_star
    flux_By = (By_star * momx_star - Bx_star * momy_star) / rho_star

    # find wavespeeds
    c0_L = np.sqrt(gamma * (P_L - 0.5 * (Bx_L**2 + By_L**2)) / rho_L)
    c0_R = np.sqrt(gamma * (P_R - 0.5 * (Bx_R**2 + By_R**2)) / rho_R)
    ca_L = np.sqrt((Bx_L**2 + By_L**2) / rho_L)
    ca_R = np.sqrt((Bx_R**2 + By_R**2) / rho_R)
    cf_L = np.sqrt(0.5 * (c0_L**2 + ca_L**2) + 0.5 * np.sqrt((c0_L**2 + ca_L**2) ** 2))
    cf_R = np.sqrt(0.5 * (c0_R**2 + ca_R**2) + 0.5 * np.sqrt((c0_R**2 + ca_R**2) ** 2))
    C_L = cf_L + np.abs(vx_L)
    C_R = cf_R + np.abs(vx_R)
    C = np.maximum(C_L, C_R)

    # add stabilizing diffusive term
    flux_Mass -= C * 0.5 * (rho_L - rho_R)
    flux_Momx -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
    flux_Momy -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)
    flux_Energy -= C * 0.5 * (en_L - en_R)
    flux_By -= C * 0.5 * (By_L - By_R)

    return flux_Mass, flux_Momx, flux_Momy, flux_Energy, flux_By


def main():
    """Finite Volume simulation — Orszag-Tang vortex"""
    print("=" * 70)
    print("2D MHD Orszag-Tang Vortex Simulation (Constrained Transport)")
    print("=" * 70)

    # Simulation parameters
    N = 128  # resolution
    boxsize = 1.0
    gamma = 5 / 3  # ideal gas gamma
    courant_fac = 0.4
    t = 0
    tEnd = 0.5
    tOut = 0.01  # draw frequency
    useSlopeLimiting = True
    plotRealTime = True  # switch on for plotting as the simulation goes along

    # Mesh
    dx = boxsize / N
    vol = dx**2
    xlin = np.linspace(0.5 * dx, boxsize - 0.5 * dx, N)
    Y, X = np.meshgrid(xlin, xlin)
    xlin_node = np.linspace(dx, boxsize, N)
    Yn, Xn = np.meshgrid(xlin_node, xlin_node)

    # Generate Initial Conditions
    rho = (gamma**2 / (4 * np.pi)) * np.ones(X.shape)
    vx = -np.sin(2 * np.pi * Y)
    vy = np.sin(2 * np.pi * X)
    P = (gamma / (4 * np.pi)) * np.ones(X.shape)  # init. gas pressure

    # magnetic field IC
    # (Az is at top-right node of each cell)
    Az = np.cos(4 * np.pi * X) / (4 * np.pi * np.sqrt(4 * np.pi)) + np.cos(
        2 * np.pi * Y
    ) / (2 * np.pi * np.sqrt(4 * np.pi))
    bx, by = getCurl(Az, dx)
    Bx, By = getBavg(bx, by)

    # add magnetic pressure to get the total pressure
    P = P + 0.5 * (Bx**2 + By**2)

    # Get conserved variables
    Mass, Momx, Momy, Energy = getConserved(rho, vx, vy, P, Bx, By, gamma, vol)

    # ---------- 诊断追踪 ----------
    t_hist = []
    diag = {"Etotal": [], "Emag": [], "Ekin": [], "Eint": [], "mean_divB": []}

    # ---------- 快照存储 ----------
    saved_snapshots = set()
    panel_snapshots = {}  # t -> {"rho": ..., "vx": ..., "vy": ..., ...}

    TOL = 1e-6  # 浮点容差

    # ---------- 保存初始状态 (t=0.0) ----------
    save_density_frame(rho, 0.0, OUT_DIR)
    saved_snapshots.add(0.0)
    if 0.0 in PANEL_TIMES:
        panel_snapshots[0.0] = {
            "rho": rho.copy(),
            "vx": vx.copy(),
            "vy": vy.copy(),
            "Bx": Bx.copy(),
            "By": By.copy(),
        }

    # prep figure
    fig = plt.figure(figsize=(4, 4), dpi=80)
    outputCount = 1

    # Simulation Main Loop
    print(f"\nGrid: {N}x{N}, gamma={gamma}, CFL={courant_fac}")
    print(f"Simulating from t=0 to t={tEnd} (density output every tOut={tOut})")
    print(f"Snapshot times: {SNAPSHOT_TIMES}")
    print(f"Output directory: {OUT_DIR}\n")

    while t < tEnd:
        # get Primitive variables
        Bx, By = getBavg(bx, by)
        rho, vx, vy, P = getPrimitive(Mass, Momx, Momy, Energy, Bx, By, gamma, vol)

        # get time step (CFL) = dx / max signal speed
        c0 = np.sqrt(gamma * (P - 0.5 * (Bx**2 + By**2)) / rho)
        ca = np.sqrt((Bx**2 + By**2) / rho)
        cf = np.sqrt(0.5 * (c0**2 + ca**2) + 0.5 * np.sqrt((c0**2 + ca**2) ** 2))
        dt = courant_fac * np.min(dx / (cf + np.sqrt(vx**2 + vy**2)))
        plotThisTurn = False
        if t + dt > outputCount * tOut:
            dt = outputCount * tOut - t
            plotThisTurn = True

        # calculate gradients
        rho_dx, rho_dy = getGradient(rho, dx)
        vx_dx, vx_dy = getGradient(vx, dx)
        vy_dx, vy_dy = getGradient(vy, dx)
        P_dx, P_dy = getGradient(P, dx)
        Bx_dx, Bx_dy = getGradient(Bx, dx)
        By_dx, By_dy = getGradient(By, dx)

        # slope limit gradients
        if useSlopeLimiting:
            rho_dx, rho_dy = slopeLimit(rho, dx, rho_dx, rho_dy)
            vx_dx, vx_dy = slopeLimit(vx, dx, vx_dx, vx_dy)
            vy_dx, vy_dy = slopeLimit(vy, dx, vy_dx, vy_dy)
            P_dx, P_dy = slopeLimit(P, dx, P_dx, P_dy)
            Bx_dx, Bx_dy = slopeLimit(Bx, dx, Bx_dx, Bx_dy)
            By_dx, By_dy = slopeLimit(By, dx, By_dx, By_dy)

        # extrapolate half-step in time
        rho_prime = rho - 0.5 * dt * (
            vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy
        )
        vx_prime = vx - 0.5 * dt * (
            vx * vx_dx
            + vy * vx_dy
            + (1 / rho) * P_dx
            - (2 * Bx / rho) * Bx_dx
            - (By / rho) * Bx_dy
            - (Bx / rho) * By_dy
        )
        vy_prime = vy - 0.5 * dt * (
            vx * vy_dx
            + vy * vy_dy
            + (1 / rho) * P_dy
            - (2 * By / rho) * By_dy
            - (Bx / rho) * By_dx
            - (By / rho) * Bx_dx
        )
        P_prime = P - 0.5 * dt * (
            (gamma * (P - 0.5 * (Bx**2 + By**2)) + By**2) * vx_dx
            - Bx * By * vy_dx
            + vx * P_dx
            + (gamma - 2) * (Bx * vx + By * vy) * Bx_dx
            - By * Bx * vx_dy
            + (gamma * (P - 0.5 * (Bx**2 + By**2)) + Bx**2) * vy_dy
            + vy * P_dy
            + (gamma - 2) * (Bx * vx + By * vy) * By_dy
        )
        Bx_prime = Bx - 0.5 * dt * (-By * vx_dy + Bx * vy_dy + vy * Bx_dy - vx * By_dy)
        By_prime = By - 0.5 * dt * (By * vx_dx - Bx * vy_dx - vy * Bx_dx + vx * By_dx)

        # extrapolate in space to face centers
        rho_XL, rho_XR, rho_YL, rho_YR = extrapolateInSpaceToFace(
            rho_prime, rho_dx, rho_dy, dx
        )
        vx_XL, vx_XR, vx_YL, vx_YR = extrapolateInSpaceToFace(
            vx_prime, vx_dx, vx_dy, dx
        )
        vy_XL, vy_XR, vy_YL, vy_YR = extrapolateInSpaceToFace(
            vy_prime, vy_dx, vy_dy, dx
        )
        P_XL, P_XR, P_YL, P_YR = extrapolateInSpaceToFace(P_prime, P_dx, P_dy, dx)
        Bx_XL, Bx_XR, Bx_YL, Bx_YR = extrapolateInSpaceToFace(
            Bx_prime, Bx_dx, Bx_dy, dx
        )
        By_XL, By_XR, By_YL, By_YR = extrapolateInSpaceToFace(
            By_prime, By_dx, By_dy, dx
        )

        # compute fluxes (local Lax-Friedrichs/Rusanov)
        flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X, flux_By_X = getFlux(
            rho_XL,
            rho_XR,
            vx_XL,
            vx_XR,
            vy_XL,
            vy_XR,
            P_XL,
            P_XR,
            Bx_XL,
            Bx_XR,
            By_XL,
            By_XR,
            gamma,
        )
        flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y, flux_Bx_Y = getFlux(
            rho_YL,
            rho_YR,
            vy_YL,
            vy_YR,
            vx_YL,
            vx_YR,
            P_YL,
            P_YR,
            By_YL,
            By_YR,
            Bx_YL,
            Bx_YR,
            gamma,
        )

        # update solution
        Mass = applyFluxes(Mass, flux_Mass_X, flux_Mass_Y, dx, dt)
        Momx = applyFluxes(Momx, flux_Momx_X, flux_Momx_Y, dx, dt)
        Momy = applyFluxes(Momy, flux_Momy_X, flux_Momy_Y, dx, dt)
        Energy = applyFluxes(Energy, flux_Energy_X, flux_Energy_Y, dx, dt)
        bx, by = constrainedTransport(bx, by, flux_By_X, flux_Bx_Y, dx, dt)

        # update time
        t += dt

        # check div B
        divB = getDiv(bx, by, dx)
        mean_divB = float(np.mean(np.abs(divB)))

        # ---------- 诊断记录 ----------
        Bx_c, By_c = getBavg(bx, by)
        rho_c, vx_c, vy_c, P_c = getPrimitive(Mass, Momx, Momy, Energy, Bx_c, By_c, gamma, vol)
        P_gas = P_c - 0.5 * (Bx_c**2 + By_c**2)
        Emag = float(0.5 * np.sum(Bx_c**2 + By_c**2) * vol)
        Ekin = float(0.5 * np.sum(rho_c * (vx_c**2 + vy_c**2)) * vol)
        Eint = float(np.sum(P_gas / (gamma - 1)) * vol)
        Etotal = Emag + Ekin + Eint

        t_hist.append(t)
        diag["Etotal"].append(Etotal)
        diag["Emag"].append(Emag)
        diag["Ekin"].append(Ekin)
        diag["Eint"].append(Eint)
        diag["mean_divB"].append(mean_divB)

        # print status periodically
        if plotThisTurn:
            status = (f"t = {t:.3f}, Etot = {Etotal:.4f}, "
                      f"<|divB|> = {mean_divB:.2e}")
            print(status)

        # ---------- 快照保存 ----------
        for st in SNAPSHOT_TIMES:
            if st not in saved_snapshots and abs(t - st) < TOL:
                rho_snap, vx_snap, vy_snap, P_snap = getPrimitive(
                    Mass, Momx, Momy, Energy, Bx_c, By_c, gamma, vol
                )
                save_density_frame(rho_snap, t, OUT_DIR)
                saved_snapshots.add(st)

                # 收集面板图所需数据
                if st in PANEL_TIMES:
                    panel_snapshots[st] = {
                        "rho": rho_snap.copy(),
                        "vx": vx_snap.copy(),
                        "vy": vy_snap.copy(),
                        "Bx": Bx_c.copy(),
                        "By": By_c.copy(),
                    }
                break

        # plot in real time
        if (plotRealTime and plotThisTurn) or (t >= tEnd):
            plt.cla()
            plt.imshow(rho.T, cmap="jet")
            plt.clim(0.06, 0.5)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect("equal")
            plt.pause(0.001)
            outputCount += 1

    # ========== 模拟结束，生成汇总图 ==========
    print(f"\nSimulation completed (t = {t:.3f})")
    print(f"Final mean |divB| = {mean_divB:.2e}")
    print(f"Final Etotal = {Etotal:.6f}")
    print("\nGenerating summary figures...")

    # 面板拼图
    create_panel_figure(panel_snapshots, OUT_DIR)

    # 诊断量曲线
    plot_diagnostics(t_hist, diag, OUT_DIR)

    # GIF 动图（需 imageio）
    save_animation_gif(OUT_DIR, fps=4)

    # 保存最终实时图
    plt.savefig(OUT_DIR / "final_frame.png", dpi=240)
    try:
        plt.show(block=False)
        plt.pause(0.5)
    except Exception:
        pass

    print("\nDone! Generated files in:", OUT_DIR)
    print("  - frame_t=*.png         (11 density snapshots)")
    print("  - panel_density_evolution.png (6-panel evolution)")
    print("  - diagnostics.png       (energy + divB time series)")
    print("  - final_frame.png       (final state)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    main()
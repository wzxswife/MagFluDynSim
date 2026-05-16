# -*- coding: utf-8 -*-
"""
2D Resistive MHD Magnetic Reconnection
分图输出版本：不生成 PPT 大拼图

输出 6 张图：
1. 01_field_lines_group.png
2. 02_current_density_perturbation_group.png
3. 03_velocity_magnitude_group.png
4. 04_magnetic_energy.png
5. 05_kinetic_energy.png
6. 06_reconnection_rate_proxy.png

说明：
- 时间节点根据模拟结果自动选取，但强制保持足够间隔，避免 t=0.1、0.2 这种无意义节点。
- 电流密度图显示 ΔJz = Jz - Jz0，避免初始 Harris 电流片背景压制演化结构。
- 重联率指标定义为 R(t)=max[η|Jz-Jz0|]。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. 基本参数
# ============================================================

NX, NY = 160, 160
LX, LY = 4.0 * np.pi, 2.0 * np.pi

dx = LX / NX
dy = LY / NY

x = np.linspace(0, LX, NX, endpoint=False)
y = np.linspace(-LY / 2, LY / 2, NY, endpoint=False)
XX, YY = np.meshgrid(x, y, indexing="ij")

# 物理参数
ETA = 2e-3
NU = 2e-3
EPS = 0.03
A_WIDTH = 0.5

# 时间参
DT = 0.01
N_STEPS = 6000
RECORD_EVERY = 10

# 显示区域：显示完整 x 域，确保磁岛可见
# 原先裁剪到 (LX/2±4.8) 会把 x=Lx/4 和 x=3Lx/4 处的 X 点及其间的磁岛截断
X_LIM = (0, LX)
Y_LIM = (-1.8, 1.8)

N_CONTOUR = 24

# 输出目录
OUT_DIR = Path.cwd() / "reconnection_separate_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 画图风格
# ============================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
    "font.size": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "contour.negative_linestyle": "solid",
})

NAVY = "#0b2f78"
BLUE = "#2563eb"
RED = "#d7191c"
PURPLE = "#7a32b8"
TEXT = "#1f2937"
SUBTEXT = "#475569"
BORDER = "#b8c3d6"
PANEL_BG = "#eef4fb"


# ============================================================
# 3. 谱算子
# ============================================================

kx_1d = 2.0 * np.pi * np.fft.fftfreq(NX, d=dx)
ky_1d = 2.0 * np.pi * np.fft.fftfreq(NY, d=dy)

KX, KY = np.meshgrid(kx_1d, ky_1d, indexing="ij")
K2 = KX**2 + KY**2
K2[0, 0] = 1.0


def laplacian(f):
    return np.fft.ifft2(-K2 * np.fft.fft2(f)).real


def solve_poisson(rhs):
    rhat = np.fft.fft2(rhs)
    phat = -rhat / K2
    phat[0, 0] = 0.0
    return np.fft.ifft2(phat).real


# ============================================================
# 4. 差分与泊松括号
# ============================================================

def ddx(f):
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dx)


def ddy(f):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dy)


def poisson_bracket(f, g):
    return ddx(f) * ddy(g) - ddy(f) * ddx(g)


# ============================================================
# 5. MHD 方程
# ============================================================

def derived_fields(Az, omega):
    Bx = ddy(Az)
    By = -ddx(Az)
    Jz = -laplacian(Az)

    psi = solve_poisson(-omega)
    vx = ddy(psi)
    vy = -ddx(psi)

    return Bx, By, Jz, psi, vx, vy


def rhs(Az, omega, eta, nu):
    Bx, By, Jz, psi, vx, vy = derived_fields(Az, omega)

    dAz = poisson_bracket(psi, Az) + eta * laplacian(Az)

    domega = (
        poisson_bracket(psi, omega)
        - poisson_bracket(Az, Jz)
        + nu * laplacian(omega)
    )

    return dAz, domega


def rk4_step(Az, omega, dt, eta, nu):
    k1a, k1w = rhs(Az, omega, eta, nu)
    k2a, k2w = rhs(Az + 0.5 * dt * k1a, omega + 0.5 * dt * k1w, eta, nu)
    k3a, k3w = rhs(Az + 0.5 * dt * k2a, omega + 0.5 * dt * k2w, eta, nu)
    k4a, k4w = rhs(Az + dt * k3a, omega + dt * k3w, eta, nu)

    Az_new = Az + dt / 6.0 * (k1a + 2.0 * k2a + 2.0 * k3a + k4a)
    omega_new = omega + dt / 6.0 * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)

    return Az_new, omega_new


# ============================================================
# 6. 初始条件
# ============================================================

def initial_condition(a=A_WIDTH, eps=EPS):
    """
    Harris 电流片 + 中心局域扰动。
    这里的扰动刻意放在 x=Lx/2, y=0 附近，
    让 X 型结构和出流喷流更容易出现在画面中心。
    """
    B0 = 1.0

    Az_base = -B0 * a * np.log(np.cosh(YY / a))

    sigma_y = 1.2 * a

    # 负号用于让中心附近更接近 X 型鞍点结构
    seed = -eps * np.cos(2.0 * np.pi * (XX - LX / 2.0) / LX) * np.exp(-(YY / sigma_y)**2)

    Az = Az_base + seed
    omega = np.zeros_like(Az)

    return Az, omega


# ============================================================
# 7. 平滑与自动选点
# ============================================================

def smooth_1d(arr, window=11):
    arr = np.asarray(arr, dtype=float)

    if window <= 1 or len(arr) < window:
        return arr.copy()

    kernel = np.ones(window) / window
    pad = window // 2
    arr_pad = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(arr_pad, kernel, mode="valid")


def normalize01(arr):
    arr = np.asarray(arr, dtype=float)
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)

    if amax - amin < 1e-14:
        return np.zeros_like(arr)

    return (arr - amin) / (amax - amin)


def choose_key_indices(t_hist, R_hist, K_hist):
    """
    自动选择 5 个关键时间点。

    这版不再用“10%峰值、35%峰值”那种容易选到 t=0.1 的方法。
    它先用重联率指标和动能构造活动强度，再根据活动峰值和总时长选取
    间隔足够开的 5 个阶段。
    """
    t_arr = np.asarray(t_hist)
    R_arr = np.asarray(R_hist)
    K_arr = np.asarray(K_hist)

    n = len(t_arr)

    if n < 5:
        return list(range(n))

    R_s = smooth_1d(R_arr, window=11)
    K_s = smooth_1d(K_arr, window=11)

    activity = 0.65 * normalize01(R_s) + 0.35 * normalize01(K_s)

    # 忽略最开始 8% 的记录，避免初始扰动把峰值判断带偏
    start_search = max(3, int(0.08 * n))
    i_peak = start_search + int(np.argmax(activity[start_search:]))

    # 如果活动峰值太靠近末端，就用总时长均匀分段；
    # 如果峰值在中间，则围绕峰值选前中后阶段。
    if i_peak > int(0.82 * (n - 1)) or i_peak < int(0.30 * (n - 1)):
        idx = [
            0,
            int(0.22 * (n - 1)),
            int(0.45 * (n - 1)),
            int(0.68 * (n - 1)),
            n - 1,
        ]
    else:
        idx = [
            0,
            int(0.35 * i_peak),
            int(0.65 * i_peak),
            i_peak,
            int(i_peak + 0.65 * ((n - 1) - i_peak)),
        ]

    # 强制最小间隔，避免两个时间点贴得太近
    min_gap = max(2, int(0.08 * n))
    idx = sorted(idx)

    fixed = [idx[0]]
    for val in idx[1:]:
        if val <= fixed[-1] + min_gap:
            val = fixed[-1] + min_gap
        fixed.append(min(val, n - 1))

    if len(set(fixed)) < 5 or fixed[-1] > n - 1:
        fixed = list(np.linspace(0, n - 1, 5, dtype=int))

    return fixed[:5]


# ============================================================
# 8. 运行模拟
# ============================================================

def run_simulation():
    Az, omega = initial_condition()

    _, _, Jz0, _, _, _ = derived_fields(Az, omega)

    snapshots = []
    t_hist = []
    Emag_hist = []
    Ekin_hist = []
    R_hist = []
    psi_hist = []

    for step in range(N_STEPS + 1):
        t = step * DT

        if step % RECORD_EVERY == 0 or step == N_STEPS:
            Bx, By, Jz, psi, vx, vy = derived_fields(Az, omega)

            Emag = 0.5 * np.sum(Bx**2 + By**2) * dx * dy
            Ekin = 0.5 * np.sum(vx**2 + vy**2) * dx * dy

            # =====================================================
            # 重联率：用 η·max(|Jz|) 代替 d(Az_O-Az_X)/dt
            # 物理依据：感应方程中 E_rec = η·Jz 是局部重联电场，
            # 其最大值是标准 Sweet-Parker 重联率指标，光滑无噪声。
            # =====================================================
            Erec = ETA * float(np.max(np.abs(Jz)))

            t_hist.append(t)
            Emag_hist.append(Emag)
            Ekin_hist.append(Ekin)

            psi_hist.append(Erec)
            snapshots.append({
                "t": t,
                "step": step,
                "Az": Az.copy(),
                "omega": omega.copy(),
                "Jz": Jz.copy(),
                "dJz": (Jz - Jz0).copy(),
                "vx": vx.copy(),
                "vy": vy.copy(),
            })

        if step < N_STEPS:
            Az, omega = rk4_step(Az, omega, DT, ETA, NU)

    return snapshots, np.array(t_hist), np.array(Emag_hist), np.array(Ekin_hist), np.array(psi_hist)

# ============================================================
# 9. 绘图辅助函数
# ============================================================

def style_snap_ax(ax):
    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)
    ax.set_xticks([])
    ax.set_yticks([])

    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
        sp.set_linewidth(0.8)


def style_diag_ax(ax):
    ax.grid(True, linestyle="--", linewidth=0.7, color="#cbd5e1", alpha=0.45)

    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
        sp.set_linewidth(1.0)

    ax.tick_params(labelsize=10, colors=TEXT)


def add_time_stage_text(ax, time_label, stage_label):
    ax.set_title(time_label, fontsize=13, color=BLUE, fontweight="bold", pad=9)

    ax.text(
        0.5, -0.14,
        stage_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        color=TEXT,
        fontweight="bold"
    )


def make_time_labels(selected_snaps):
    labels = []
    for i, snap in enumerate(selected_snaps):
        if i == 0:
            labels.append(rf"$t={snap['t']:.1f}\,\tau_A$（初始）")
        else:
            labels.append(rf"$t={snap['t']:.1f}\,\tau_A$")
    return labels


# ============================================================
# 10. 三组空间图
# ============================================================

def plot_field_lines(selected_snaps, time_labels, stage_labels, out_path):
    all_Az = np.concatenate([s["Az"].ravel() for s in selected_snaps])
    levels = np.linspace(
        np.percentile(all_Az, 3),
        np.percentile(all_Az, 97),
        N_CONTOUR
    )

    fig, axes = plt.subplots(
        1, 5,
        figsize=(17.5, 4.0),
        dpi=300,
        facecolor="white"
    )

    fig.subplots_adjust(
        left=0.035,
        right=0.985,
        top=0.82,
        bottom=0.20,
        wspace=0.10
    )

    fig.text(
        0.5, 0.94,
        r"磁力线（$A_z$ 等值线）",
        ha="center",
        va="center",
        fontsize=23,
        color=NAVY,
        fontweight="bold"
    )

    for i, (ax, snap) in enumerate(zip(axes, selected_snaps)):
        Az = snap["Az"]

        # 白底，仿照最开始参考图
        ax.set_facecolor("white")

        ax.contour(
            x, y, Az.T,
            levels=levels,
            colors="#1a3a6b",      # 深蓝色磁力线
            linewidths=0.85,
            alpha=0.95,
            linestyles="solid"
        )

        # 中心电流片参考线
        ax.axhline(
            0,
            color=RED,
            lw=1.10,
            alpha=0.90
        )

        style_snap_ax(ax)
        add_time_stage_text(ax, time_labels[i], stage_labels[i])

    fig.savefig(
        out_path,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.08
    )

    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_current_density(selected_snaps, time_labels, stage_labels, out_path):
    all_J = np.concatenate([s["Jz"].ravel() for s in selected_snaps])
    J_REF = max(np.percentile(np.abs(all_J), 99.2), 1e-10)
    fig, axes = plt.subplots(1, 5, figsize=(17.5, 4.15), dpi=300, facecolor="white")
    fig.subplots_adjust(left=0.035, right=0.94, top=0.82, bottom=0.20, wspace=0.10)

    fig.text(
        0.5, 0.94,
        r"电流密度（$J_z$）",
        ha="center", va="center",
        fontsize=23, color=RED, fontweight="bold"
    )



    im = None
    for i, (ax, snap) in enumerate(zip(axes, selected_snaps)):
        Jz = snap["Jz"]
        J_plot = np.clip(Jz / J_REF, -1.0, 1.0)
        im = ax.imshow(
        J_plot.T,
        origin="lower",
        extent=[0, LX, -LY / 2, LY / 2],
        aspect="auto",
        cmap="RdBu_r",   # 蓝-白-红发散色阶，与速度图风格一致，零点为白色
        vmin=-1,
        vmax=1,
        interpolation="bilinear"
    )

        style_snap_ax(ax)
        add_time_stage_text(ax, time_labels[i], stage_labels[i])

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.020, pad=0.012)
    cbar.set_label(r"$J_z/J_{\rm ref}$", fontsize=11)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(labelsize=9)

    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_velocity(selected_snaps, time_labels, stage_labels, out_path):
    all_speed = np.concatenate([
        np.sqrt(s["vx"]**2 + s["vy"]**2).ravel()
        for s in selected_snaps
    ])

    V_REF = max(np.percentile(all_speed, 99.5), 1e-10)

    fig, axes = plt.subplots(
        1, 5,
        figsize=(17.5, 4.15),
        dpi=300,
        facecolor="white"
    )

    fig.subplots_adjust(
        left=0.035,
        right=0.94,
        top=0.82,
        bottom=0.20,
        wspace=0.10
    )

    fig.text(
        0.5, 0.94,
        r"速度大小（$|\mathbf{v}|$）",
        ha="center",
        va="center",
        fontsize=23,
        color=PURPLE,
        fontweight="bold"
    )

    im = None

    for i, (ax, snap) in enumerate(zip(axes, selected_snaps)):
        speed = np.sqrt(snap["vx"]**2 + snap["vy"]**2)
        speed_plot = np.clip(speed / V_REF, 0.0, 1.0)

        im = ax.imshow(
            speed_plot.T,
            origin="lower",
            extent=[0, LX, -LY / 2, LY / 2],
            aspect="auto",
            cmap="jet",
            vmin=0,
            vmax=1,
            interpolation="bilinear"
        )

        style_snap_ax(ax)
        add_time_stage_text(ax, time_labels[i], stage_labels[i])

    cbar = fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        fraction=0.020,
        pad=0.012
    )

    cbar.set_label(r"$|\mathbf{v}|/|\mathbf{v}|_{\rm ref}$", fontsize=11)
    cbar.set_ticks([0, 0.5, 1])
    cbar.ax.tick_params(labelsize=9)

    fig.savefig(
        out_path,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.08
    )

    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# 11. 诊断图
# ============================================================

def plot_diag_curve(t, y, selected_times, title, ylabel, color, out_path, normalize=False):
    if normalize:
        y_plot = y / max(y[0], 1e-14)
    else:
        y_plot = y

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300, facecolor="white")
    fig.subplots_adjust(left=0.16, right=0.96, top=0.82, bottom=0.18)

    ax.plot(t, y_plot, color=color, lw=2.5)

    for tt in selected_times:
        yy = np.interp(tt, t, y_plot)
        ax.axvline(tt, color="#cbd5e1", lw=0.9, ls="--", alpha=0.75)
        ax.scatter([tt], [yy], s=32, color=color, zorder=5)

    ax.set_title(title, fontsize=22, color=color, fontweight="bold", pad=12)
    ax.set_xlabel(r"时间 $t/\tau_A$", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlim(t[0], t[-1])

    style_diag_ax(ax)

    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# 12. 主程序
# ============================================================

def main():
    print("=" * 70)
    print("2D Resistive MHD Magnetic Reconnection")
    print(f"Grid: {NX} x {NY}")
    print(f"eta={ETA}, nu={NU}, eps={EPS}, dt={DT}, steps={N_STEPS}")
    print("=" * 70)

    snapshots, t_hist, Emag_hist, Ekin_hist, psi_hist = run_simulation()
    # psi_hist now contains η·max(|Jz|) — smooth, physically standard reconnection rate
    R_hist = psi_hist
    key_idx = choose_key_indices(t_hist, R_hist, Ekin_hist)
    selected_snaps = [snapshots[i] for i in key_idx]
    selected_times = [snap["t"] for snap in selected_snaps]

    stage_labels = [
        "初始阶段",
        "扰动增长阶段",
        "电流片变形阶段",
        "快速重联阶段",
        "非线性演化阶段"
    ]

    time_labels = make_time_labels(selected_snaps)

    print("\nSelected key snapshots:")
    for i, snap in enumerate(selected_snaps):
        print(f"{i+1}. {stage_labels[i]}: t={snap['t']:.2f}, step={snap['step']}")

    print(f"\nOutput directory: {OUT_DIR}")

    plot_field_lines(
        selected_snaps,
        time_labels,
        stage_labels,
        OUT_DIR / "01_field_lines_group.png"
    )

    plot_current_density(
        selected_snaps,
        time_labels,
        stage_labels,
        OUT_DIR / "02_current_density_perturbation_group.png"
    )

    plot_velocity(
        selected_snaps,
        time_labels,
        stage_labels,
        OUT_DIR / "03_velocity_magnitude_group.png"
    )

    plot_diag_curve(
        t_hist,
        Emag_hist,
        selected_times,
        title=r"磁能 $E_B(t)$",
        ylabel=r"$E_B(t)/E_{B0}$",
        color=RED,
        out_path=OUT_DIR / "04_magnetic_energy.png",
        normalize=True
    )

    plot_diag_curve(
        t_hist,
        Ekin_hist / max(Emag_hist[0], 1e-14),
        selected_times,
        title=r"动能 $E_K(t)$",
        ylabel=r"$E_K(t)/E_{B0}$",
        color=BLUE,
        out_path=OUT_DIR / "05_kinetic_energy.png",
        normalize=False
    )

    plot_diag_curve(
        t_hist,
        R_hist,
        selected_times,
        title=r"重联率 $R(t)$",
        ylabel=r"$R(t)=\eta\cdot\max(|J_z|)$",
        color=PURPLE,
        out_path=OUT_DIR / "06_reconnection_rate_proxy.png",
        normalize=False
    )

    print("\nDone.")
    print("Generated files:")
    print("  01_field_lines_group.png")
    print("  02_current_density_perturbation_group.png")
    print("  03_velocity_magnitude_group.png")
    print("  04_magnetic_energy.png")
    print("  05_kinetic_energy.png")
    print("  06_reconnection_rate_proxy.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
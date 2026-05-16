# -*- coding: utf-8 -*-
"""
1D EMHD whistler-wave propagation

修正版要点：
1. B 定义在整格点 zi、整时间步 n；
2. E 定义在半格点 zh、半时间步 n+1/2；
3. J 定义在半格点 zh、整时间步 n；
4. 采用周期边界条件；
5. J 的回旋项使用中心化更新，比直接显式欧拉更稳定、更符合蛙跳格式思想。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 全局绘图风格：LaTeX + 放大字体
# ============================================================
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,

    # 全局字体放大
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})

# 线条颜色：深蓝色 + 黑色
deep_blue = "#0b3c8c"
black = "#000000"

# ============================================================
# 1. 参数设置
# ============================================================
omega = 0.4          # 哨声波频率，单位 Omega_e
omega_pe = 5.0       # 等离子体频率，单位 Omega_e
c2 = omega_pe**2     # 归一化后 c = omega_pe，因此 c^2 = omega_pe^2

z_min, z_max = -50.0, 150.0   # 空间范围，单位 d_e
Nz = 2000                     # 网格数
dz = (z_max - z_min) / Nz
dt = 0.01                     # 时间步长，单位 Omega_e^{-1}

T_end = 50.0                  # 总模拟时间
save_dt = 0.2                 # 时空图保存间隔
out_dir = "outputs"           # 输出文件夹

os.makedirs(out_dir, exist_ok=True)


# ============================================================
# 2. 色散关系与群速度
# ============================================================
def whistler_k(w, wpe):
    """
    平行传播哨声波色散关系：
        k^2 d_e^2 = (omega/omega_pe)^2 + omega/(1 - omega)
    这里 Omega_e 已归一化为 1，c = omega_pe。

    这个写法与后面的 J = [k - omega^2/(k c^2)] A(z) 形式自洽。
    """
    return np.sqrt((w / wpe) + w / (1.0 - w))


k = whistler_k(omega, omega_pe)

# 数值微分计算群速度 vg = d omega / d k
domega = 1e-6
kp = whistler_k(omega + domega, omega_pe)
km = whistler_k(omega - domega, omega_pe)
vg = 2.0 * domega / (kp - km)

print(f"Wave number  k*d_e = {k:.6f}")
print(f"Phase speed  omega/k = {omega/k:.6f} V_Ae")
print(f"Group velocity vg    = {vg:.6f} V_Ae")
print(f"dz = {dz:.6f}, dt = {dt:.6f}, dt/dz = {dt/dz:.6f}")


# ============================================================
# 3. 网格
# ============================================================
# 整格点：B 使用
zi = z_min + np.arange(Nz) * dz

# 半格点：E 和 J 使用
zh = zi + 0.5 * dz

idx = np.arange(Nz)
ip = (idx + 1) % Nz
im = (idx - 1) % Nz

r = dt / dz


# ============================================================
# 4. 初始波包包络 A(z)
# ============================================================
def amplitude(z):
    """
    只在 0 <= z <= 100 d_e 内泵入波包。
    A(z) = 0.005 * [-cos(2*pi*z/100) + 1]
    因此最大振幅为 0.01。
    """
    A = np.zeros_like(z, dtype=float)
    mask = (z >= 0.0) & (z <= 100.0)
    A[mask] = 0.005 * (-np.cos(2.0 * np.pi * z[mask] / 100.0) + 1.0)
    return A


# ============================================================
# 5. 解析形式的初始场
# ============================================================
def analytic_fields(t_plot):
    """
    返回同一物理时刻 t_plot 的解析波包，用于画 t=0 的初始图。
    注意：这里 E 也取在 t_plot，而不是半时间步。
    """
    Ai = amplitude(zi)
    Ah = amplitude(zh)

    Bx = Ai * np.sin(k * zi - omega * t_plot)
    By = Ai * np.cos(k * zi - omega * t_plot)

    Ex = (omega / k) * Ah * np.cos(k * zh - omega * t_plot)
    Ey = -(omega / k) * Ah * np.sin(k * zh - omega * t_plot)

    coeff_J = k - omega**2 / (k * c2)
    Jx = coeff_J * Ah * np.sin(k * zh - omega * t_plot)
    Jy = coeff_J * Ah * np.cos(k * zh - omega * t_plot)

    return Bx, By, Ex, Ey, Jx, Jy


def initial_numerical_fields():
    """
    构造数值推进需要的初始量：
    B^0, J^0 在 t = 0；
    E^{1/2} 在 t = +dt/2。
    """
    Ai = amplitude(zi)
    Ah = amplitude(zh)

    t0 = 0.0
    t_half = 0.5 * dt

    Bx = Ai * np.sin(k * zi - omega * t0)
    By = Ai * np.cos(k * zi - omega * t0)

    Ex = (omega / k) * Ah * np.cos(k * zh - omega * t_half)
    Ey = -(omega / k) * Ah * np.sin(k * zh - omega * t_half)

    coeff_J = k - omega**2 / (k * c2)
    Jx = coeff_J * Ah * np.sin(k * zh - omega * t0)
    Jy = coeff_J * Ah * np.cos(k * zh - omega * t0)

    return Bx, By, Ex, Ey, Jx, Jy


# ============================================================
# 6. 单步推进
# ============================================================
def advance_one_step(Bx, By, Ex, Ey, Jx, Jy):
    """
    从：
        B^n, J^n, E^{n+1/2}
    推进到：
        B^{n+1}, J^{n+1}, E^{n+3/2}

    更新顺序：
        1. 用 E^{n+1/2} 更新 B^{n+1}
        2. 用 E^{n+1/2} 中心化更新 J^{n+1}
        3. 用 B^{n+1}, J^{n+1} 更新 E^{n+3/2}
    """

    # ---------- 1) B 更新 ----------
    Bx_new = Bx + r * (Ey - Ey[im])
    By_new = By - r * (Ex - Ex[im])

    # ---------- 2) J 更新 ----------
    # 方程：
    #   dJx/dt = Ex - Jy
    #   dJy/dt = Ey + Jx
    #
    # 中心化写法：
    #   (Jx^{n+1}-Jx^n)/dt = Ex^{n+1/2} - (Jy^{n+1}+Jy^n)/2
    #   (Jy^{n+1}-Jy^n)/dt = Ey^{n+1/2} + (Jx^{n+1}+Jx^n)/2
    #
    # 这是一个 2x2 线性系统，可直接解析求解。
    h = 0.5 * dt
    rhs1 = Jx + dt * Ex - h * Jy
    rhs2 = Jy + dt * Ey + h * Jx
    den = 1.0 + h * h

    Jx_new = (rhs1 - h * rhs2) / den
    Jy_new = (h * rhs1 + rhs2) / den

    # ---------- 3) E 更新 ----------
    Ex_new = Ex + dt * (-c2 * (By_new[ip] - By_new) / dz - c2 * Jx_new)
    Ey_new = Ey + dt * ( c2 * (Bx_new[ip] - Bx_new) / dz - c2 * Jy_new)

    return Bx_new, By_new, Ex_new, Ey_new, Jx_new, Jy_new


# ============================================================
# 7. 主模拟：保存 Bx 时空演化
# ============================================================
def run_full_simulation():
    Bx, By, Ex, Ey, Jx, Jy = initial_numerical_fields()

    Nt = int(round(T_end / dt))
    save_every = max(1, int(round(save_dt / dt)))

    t_save = [0.0]
    Bx_save = [Bx.copy()]

    for n in range(Nt):
        Bx, By, Ex, Ey, Jx, Jy = advance_one_step(Bx, By, Ex, Ey, Jx, Jy)

        t_now = (n + 1) * dt
        if (n + 1) % save_every == 0:
            t_save.append(t_now)
            Bx_save.append(Bx.copy())

    return np.array(t_save), np.array(Bx_save)


# ============================================================
# 8. 获取指定时刻快照
# ============================================================
def run_to_snapshot(T_snap):
    """
    返回 t=0 的解析场和 t=T_snap 附近的数值场。
    对于 E，由于数值格式中 E 在半时间步上，这里用相邻两个半步的平均值
    近似得到整数时刻的 E。
    """
    snap0 = analytic_fields(0.0)

    Bx, By, Ex, Ey, Jx, Jy = initial_numerical_fields()
    Nt = int(round(T_snap / dt))

    Ex_old = Ex.copy()
    Ey_old = Ey.copy()

    for _ in range(Nt):
        Ex_old = Ex.copy()
        Ey_old = Ey.copy()
        Bx, By, Ex, Ey, Jx, Jy = advance_one_step(Bx, By, Ex, Ey, Jx, Jy)

    # E^{n} ≈ 0.5 * (E^{n-1/2} + E^{n+1/2})
    Ex_center = 0.5 * (Ex_old + Ex)
    Ey_center = 0.5 * (Ey_old + Ey)

    snapT = (Bx.copy(), By.copy(), Ex_center.copy(), Ey_center.copy(), Jx.copy(), Jy.copy())
    return snap0, snapT


# ============================================================
# 9. 作图
# ============================================================
def savefig_both(fig, basename):
    png_path = os.path.join(out_dir, basename + ".png")
    pdf_path = os.path.join(out_dir, basename + ".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def plot_snapshot(fields, title, basename):
    Bx, By, Ex, Ey, Jx, Jy = fields

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.8), sharex=True)
    fig.suptitle(title, fontsize=15,y=0.97)

    # ======================================================
    # 1) B
    # ======================================================
    ax = axes[0]
    ax_r = ax.twinx()

    ax.plot(zi, Bx, lw=1.4, color="black")
    ax.plot(zi, By, lw=1.4, color=deep_blue)

    ax.set_ylim(-0.022, 0.022)
    ax_r.set_ylim(-0.022, 0.022)
    ax.set_xlim(-50, 150)

    ax.axvline(0, lw=0.9, ls=":", color="black")

    # 左右标签分别设置
    ax.set_ylabel(r"$B_x/B_0$", fontsize=13, color="black")
    ax_r.set_ylabel(r"$B_y/B_0$", fontsize=13, color=deep_blue)

    # 刻度颜色
    ax.tick_params(axis="y", labelcolor="black", labelsize=13)
    ax_r.tick_params(axis="y", labelcolor="black", labelsize=13)
    ax.tick_params(axis="x", labelsize=13)

    # 右轴只保留标签，也可保留刻度
    # 如果你不想显示右边刻度数字，可以用下面这一行：
    # ax_r.set_yticks([])

    # ======================================================
    # 2) E
    # ======================================================
    ax = axes[1]
    ax_r = ax.twinx()

    ax.plot(zh, Ex, lw=1.4, color="black")
    ax.plot(zh, Ey, lw=1.4, color=deep_blue)

    ax.set_ylim(-0.012, 0.012)
    ax_r.set_ylim(-0.012, 0.012)
    ax.set_xlim(-50, 150)

    ax.axvline(0, lw=0.9, ls=":", color="black")

    ax.set_ylabel(r"$E_x/(V_{Ae}B_0)$", fontsize=13, color="black")
    ax_r.set_ylabel(r"$E_y/(V_{Ae}B_0)$", fontsize=13, color=deep_blue)

    ax.tick_params(axis="y", labelcolor="black", labelsize=13)
    ax_r.tick_params(axis="y", labelcolor="black", labelsize=13)
    ax.tick_params(axis="x", labelsize=13)

    # ax_r.set_yticks([])

    # ======================================================
    # 3) J
    # ======================================================
    ax = axes[2]
    ax_r = ax.twinx()

    ax.plot(zh, Jx, lw=1.4, color="black")
    ax.plot(zh, Jy, lw=1.4, color=deep_blue)

    ax.set_ylim(-0.012, 0.012)
    ax_r.set_ylim(-0.012, 0.012)
    ax.set_xlim(-50, 150)

    ax.axvline(0, lw=0.9, ls=":", color="black")

    ax.set_ylabel(r"$J_x/(en_eV_{Ae})$", fontsize=13, color="black")
    ax_r.set_ylabel(r"$J_y/(en_eV_{Ae})$", fontsize=13, color=deep_blue)

    ax.tick_params(axis="y", labelcolor="black", labelsize=13)
    ax_r.tick_params(axis="y", labelcolor="black", labelsize=13)
    ax.tick_params(axis="x", labelsize=13)

    ax.set_xlabel(r"$z/d_e$", fontsize=13)

    # ax_r.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    savefig_both(fig, basename)
    plt.close(fig)

def plot_bx_contour(t_save, Bx_save):
    fig, ax = plt.subplots(figsize=(7.2, 6.0))

    zz, tt = np.meshgrid(zi, t_save)

    vmax = 0.010
    cm = ax.pcolormesh(
        zz, tt, Bx_save,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        shading="auto"
    )

    cbar = fig.colorbar(cm, ax=ax)
    cbar.set_label(r"$B_x/B_0$", fontsize=13)

    # 手动设置色标刻度
    ticks = [-0.010, -0.005, 0.000, 0.005, 0.010]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.3f}" for t in ticks])
    cbar.ax.tick_params(labelsize=13)

    ax.set_xlabel(r"$z/d_e$", fontsize=13)
    ax.set_ylabel(r"$\Omega_e t$", fontsize=13)
    ax.set_title(r"Space-time evolution of $B_x$", fontsize=15)

    ax.set_xlim(z_min, z_max)
    ax.set_ylim(0.0, T_end)

    plt.tight_layout()
    savefig_both(fig, "bx_contour")
    plt.close(fig)

# ============================================================
# 10. 运行
# ============================================================
if __name__ == "__main__":
    # 生成 t=0 和 t=20 的空间分布图
    snap0, snap20 = run_to_snapshot(20.0)
    plot_snapshot(snap0,  r"$\Omega_e t = 0$",  "EBJ0")
    plot_snapshot(snap20, r"$\Omega_e t = 20$", "EBJ20")

    # 生成 Bx 时空演化图
    t_save, Bx_save = run_full_simulation()
    plot_bx_contour(t_save, Bx_save)

    print("\nDone.")
    print(f"All figures are saved in: {os.path.abspath(out_dir)}")
    print(f"Theoretical group velocity : vg = {vg:.6f} V_Ae")
    print(f"Wave number                : k  = {k:.6f} d_e^(-1)")

import os
import numpy as np
import matplotlib.pyplot as plt

# --- 物理参数与初始条件 [cite: 19, 29, 31] ---
gamma = 1.4
t_end = 0.14
x_range = (-1.0, 1.0)
W_L = np.array([0.445, 0.311, 8.928]) # [rho, m, E]
W_R = np.array([0.5, 0, 1.4275])

def get_primitive(w):
    rho = w[0]
    u = w[1] / rho
    p = (gamma - 1) * (w[2] - 0.5 * rho * u**2)
    return rho, u, p

def get_flux(w):
    rho, u, p = get_primitive(w)
    return np.array([w[1], w[1]*u + p, (w[2] + p)*u])

# --- 数值格式实现 ---

def lax_wendroff(w, dx, dt):
    """使用 Richtmyer 两步法实现的 LW 格式，增加人工粘性抑制振荡"""
    nx = w.shape[1]
    w_new = w.copy()

    # 第一步：计算半步中心点的值 (Lax step)
    w_half = np.zeros((3, nx-1))
    for i in range(nx-1):
        f_i = get_flux(w[:, i])
        f_ip1 = get_flux(w[:, i+1])
        w_half[:, i] = 0.5*(w[:, i] + w[:, i+1]) - (dt/(2*dx))*(f_ip1 - f_i)

    # 第二步：使用半步值更新原格点 (Leapfrog-like step)
    for i in range(1, nx-1):
        f_half_i = get_flux(w_half[:, i])
        f_half_im1 = get_flux(w_half[:, i-1])
        w_new[:, i] = w[:, i] - (dt/dx)*(f_half_i - f_half_im1)

    # 添加人工粘性抑制振荡 (Jameson-Schmidt-Turkel 风格)
    # 计算局部声速用于缩放粘性系数
    rho, u, p = get_primitive(w)
    c = np.sqrt(gamma * p / rho)
    max_speed = np.max(np.abs(u) + c)

    # 二阶人工粘性项
    eps_2 = 0.1  # 二阶粘性系数
    for i in range(2, nx-2):
        for k in range(3):  # 三个守恒变量
            # 二阶导数近似
            d2w = w[k, i+1] - 2*w[k, i] + w[k, i-1]
            # 粘性项：基于当地流动特征速度
            local_c = np.sqrt(gamma * p[i] / rho[i]) if rho[i] > 0 else 1e-10
            local_max_speed = np.abs(u[i]) + local_c
            viscosity = eps_2 * dx * local_max_speed
            w_new[k, i] += viscosity * d2w / dx

    return w_new

def tvd_van_leer(w, dx, dt):
    """van Leer 通量分裂 TVD 格式 [cite: 50, 58]"""
    nx = w.shape[1]
    w_new = w.copy()
    f_p = np.zeros_like(w)
    f_m = np.zeros_like(w)
    
    for j in range(nx):
        rho, u, p = get_primitive(w[:, j])
        c = np.sqrt(gamma * p / rho)
        ma = u / c
        if ma >= 1:
            f_p[:, j] = get_flux(w[:, j]); f_m[:, j] = 0
        elif ma <= -1:
            f_p[:, j] = 0; f_m[:, j] = get_flux(w[:, j])
        else:
            # 分裂通量公式 [cite: 58]
            fac = rho * c / 4 * (ma + 1)**2
            f_p[:, j] = fac * np.array([1, 2*c/gamma*(1 + (gamma-1)/2*ma), 2*c**2/(gamma**2-1)*(1 + (gamma-1)/2*ma)**2])
            fac_m = -rho * c / 4 * (ma - 1)**2
            f_m[:, j] = fac_m * np.array([1, 2*c/gamma*(-1 + (gamma-1)/2*ma), 2*c**2/(gamma**2-1)*(1 - (gamma-1)/2*ma)**2])
            
    for j in range(1, nx - 1):
        w_new[:, j] = w[:, j] - (dt/dx) * (f_m[:, j+1] - f_m[:, j] + f_p[:, j] - f_p[:, j-1])
    return w_new

def upwind_characteristic(w, dx, dt):
    """基于特征分解的迎风格式 - 使用 HLL 格式 (更稳定)"""
    nx = w.shape[1]
    w_new = w.copy()

    # 预计算界面通量
    flux = np.zeros((3, nx))

    for i in range(1, nx - 1):
        wL = w[:, i-1]
        wR = w[:, i]

        rhoL, uL, pL = get_primitive(wL)
        rhoR, uR, pR = get_primitive(wR)

        aL = np.sqrt(gamma * pL / rhoL)
        aR = np.sqrt(gamma * pR / rhoR)

        # 左右最值特征速度
        Sm = min(uL - aL, uR - aR)
        Sp = max(uL + aL, uR + aR)

        # 防止除零
        if abs(Sp - Sm) < 1e-10:
            flux[:, i] = 0.5 * (get_flux(wL) + get_flux(wR))
            continue

        fL = get_flux(wL)
        fR = get_flux(wR)

        # HLL 通量
        flux[:, i] = (Sp * fL - Sm * fR + Sp * Sm * (wR - wL)) / (Sp - Sm)

    # 应用通量差分
    for i in range(1, nx - 1):
        w_new[:, i] = w[:, i] - (dt / dx) * (flux[:, i+1] - flux[:, i])

    return w_new

# --- 主模拟函数 ---
def run_simulation(nx, cfl, method, t_target=0.14):
    dx = (x_range[1] - x_range[0]) / (nx - 1)
    x = np.linspace(x_range[0], x_range[1], nx)
    w = np.zeros((3, nx))
    w[:, x < 0] = W_L.reshape(3, 1)
    w[:, x >= 0] = W_R.reshape(3, 1)
    
    t = 0
    while t < t_target:
        rho, u, p = get_primitive(w)
        # 防止压力或密度为负
        rho = np.maximum(rho, 1e-10)
        p = np.maximum(p, 1e-10)
        c = np.sqrt(gamma * p / rho)
        dt = cfl * dx / np.max(np.abs(u) + c)
        if t + dt > t_target: dt = t_target - t

        if method == 'lw': w = lax_wendroff(w, dx, dt)
        elif method == 'tvd': w = tvd_van_leer(w, dx, dt)
        elif method == 'upwind': w = upwind_characteristic(w, dx, dt)

        # 确保物理量非负
        rho_new, u_new, p_new = get_primitive(w)
        w[0, :] = np.maximum(w[0, :], 1e-10)
        w[2, :] = np.maximum(w[2, :], 1e-10)

        t += dt
    return x, w

# --- 绘图函数 ---
def plot_results(x, w, title, fig_num, save_filename=None):
    rho, u, p = get_primitive(w)
    m = rho * u
    E = w[2]
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    fig.suptitle(title)

    # 模拟 PDF 中的布局 [cite: 109, 120, 136]
    labels = ['Density p', 'Mass Flow m=pu', 'Energy E']
    data = [rho, m, E]
    for i in range(3):
        axes[i].plot(x, data[i], 'go-', markersize=4, label='Numerical')
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True)
    plt.tight_layout()

    # 保存图片
    if save_filename is not None:
        fig.savefig(save_filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_filename}")
    plt.show()

# --- 生成 PDF 中的所有图像 [cite: 96, 97, 98] ---
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 图1: Lax-Wendroff, 300网格, CFL=0.22
x1, w1 = run_simulation(300, 0.22, 'lw')
plot_results(x1, w1, "Fig 1: Lax-Wendroff (300 grids, CFL=0.22)", 1,
            save_filename=os.path.join(output_dir, "fig1_lax_wendroff.png"))

# 图2: TVD, 300网格, CFL=0.1
x2, w2 = run_simulation(300, 0.1, 'tvd')
plot_results(x2, w2, "Fig 2: TVD (300 grids, CFL=0.1)", 2,
            save_filename=os.path.join(output_dir, "fig2_tvd_300.png"))

# 图3: TVD, 600网格, CFL=0.1
x3, w3 = run_simulation(600, 0.1, 'tvd')
plot_results(x3, w3, "Fig 3: TVD (600 grids, CFL=0.1)", 3,
            save_filename=os.path.join(output_dir, "fig3_tvd_600.png"))

# 图4: Upwind, 200网格, CFL=0.01
x4, w4 = run_simulation(200, 0.01, 'upwind')
plot_results(x4, w4, "Fig 4: Upwind (200 grids, CFL=0.01)", 4,
            save_filename=os.path.join(output_dir, "fig4_upwind.png"))
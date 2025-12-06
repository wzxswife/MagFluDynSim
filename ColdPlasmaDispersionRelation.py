import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def get_dispersion_k(w_vals, w_pe, w_ce, theta_deg, k_max_limit=10.0):
    """
    计算给定频率下的归一化波数 kc/Omega_e。
    关键修正：不删除数组元素，而是将无效值设为 NaN，防止绘图时错误连接。
    """
    theta = np.deg2rad(theta_deg)
    sin_t2 = np.sin(theta)**2
    cos_t2 = np.cos(theta)**2
    
    # 初始化 k 数组为全 NaN
    k1 = np.full_like(w_vals, np.nan)
    k2 = np.full_like(w_vals, np.nan)

    # 忽略除零警告 (处理共振点)
    with np.errstate(divide='ignore', invalid='ignore'):
        # Stix Parameters
        X = (w_pe / w_vals)**2
        Y = w_ce / w_vals
        
        # R, L 包含 singularity (w = w_ce)，可能会产生 inf
        # 使用 epsilon 避免完全除零 (或者依赖 numpy 的 inf 处理)
        denom_R = 1.0 - Y
        denom_L = 1.0 + Y
        
        R = 1.0 - X / denom_R
        L = 1.0 - X / denom_L
        P = 1.0 - X
        S = 0.5 * (R + L)
        D = 0.5 * (R - L)
        
        # Coefficients for n^2
        A = S * sin_t2 + P * cos_t2
        B = R * L * sin_t2 + P * S * (1.0 + cos_t2)
        C = P * R * L
        
        # Discriminant F^2
        F2 = (R * L - P * S)**2 * (np.sin(theta)**4) + 4 * (P**2) * (D**2) * cos_t2
        F = np.sqrt(np.maximum(0, F2)) # maximum 避免数值误差导致的微小负数
        
        # Calculate n^2 solutions
        # 2*A 可能会是 0 (Resonance cone)，但这会产生 inf，后续会被过滤
        n2_1 = (B + F) / (2 * A)
        n2_2 = (B - F) / (2 * A)
        
        # --- 修正核心：过滤逻辑 ---
        
        # 1. 过滤倏逝波 (n^2 < 0)
        # 2. 过滤极大的 n^2 (接近共振) 以防止横向拉丝
        #    根据 w = kc/n => k = w*n，如果 n 很大，k 也会很大
        
        # 处理第一个根
        valid_1 = (n2_1 > 0)
        n_1 = np.sqrt(n2_1, where=valid_1, out=np.full_like(n2_1, np.nan))
        k_calc_1 = w_vals * n_1
        # 再次过滤：如果 k 超出绘图范围太多，设为 nan
        k_calc_1[k_calc_1 > k_max_limit] = np.nan
        k1 = k_calc_1

        # 处理第二个根
        valid_2 = (n2_2 > 0)
        n_2 = np.sqrt(n2_2, where=valid_2, out=np.full_like(n2_2, np.nan))
        k_calc_2 = w_vals * n_2
        k_calc_2[k_calc_2 > k_max_limit] = np.nan
        k2 = k_calc_2
        
    return k1, k2

def plot_panel(ax, w_pe, w_ce, x_lim, y_lim, title):
    # 参数设置
    thetas = np.linspace(0, 90, 91) # 增加密度使颜色更连续
    w_vals = np.linspace(0.001, y_lim, 2000) # 频率分辨率要足够高
    
    # 颜色设置
    cmap = cm.jet
    norm = mcolors.Normalize(vmin=0, vmax=90)
    
    # 绘制光速线 (k = w)
    ax.plot([0, x_lim], [0, x_lim], 'k-', lw=0.5, alpha=0.8)
    
    for theta in thetas:
        color = cmap(norm(theta))
        
        # 传入略大于 x_lim 的限制，以便线条能画满边缘但不乱飞
        k1, k2 = get_dispersion_k(w_vals, w_pe, w_ce, theta, k_max_limit=x_lim * 1.2)
        
        # 绘图：由于 k1, k2 包含了 NaN，matplotlib 会自动处理断点
        ax.plot(k1, w_vals, color=color, linewidth=0.6)
        ax.plot(k2, w_vals, color=color, linewidth=0.6)

    # 装饰
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)
    ax.set_xlabel(r'$kc/\Omega_e$', fontsize=14)
    ax.set_ylabel(r'$\omega/\Omega_e$', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.tick_params(direction='in', top=True, right=True)

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图
    plot_panel(ax1, w_pe=2.0, w_ce=1.0, x_lim=6.0, y_lim=3.0, 
               title=r'$\omega_{pe}/\Omega_e=2.0$')
    
    # 右图
    plot_panel(ax2, w_pe=0.5, w_ce=1.0, x_lim=3.0, y_lim=1.5, 
               title=r'$\omega_{pe}/\Omega_e=0.5$')
    
    # Colorbar
    sm = cm.ScalarMappable(cmap=cm.jet, norm=mcolors.Normalize(vmin=0, vmax=90))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='vertical', fraction=0.05, pad=0.02)
    cbar.set_label(r'$\theta$ (Deg)', fontsize=14)
    cbar.set_ticks([0, 30, 60, 90])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
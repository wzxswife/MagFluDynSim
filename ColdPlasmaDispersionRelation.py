import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def get_dispersion_k(w_vals, w_pe, w_ce, theta_deg, k_max_limit=10.0):
    """
    计算给定频率下的归一化波数 kc/Omega_e。
    保持之前的修正：使用 NaN 处理断点，过滤极大值。
    """
    theta = np.deg2rad(theta_deg)
    sin_t2 = np.sin(theta)**2
    cos_t2 = np.cos(theta)**2
    
    k1 = np.full_like(w_vals, np.nan)
    k2 = np.full_like(w_vals, np.nan)

    with np.errstate(divide='ignore', invalid='ignore'):
        # Stix Parameters
        X = (w_pe / w_vals)**2
        Y = w_ce / w_vals
        
        denom_R = 1.0 - Y
        denom_L = 1.0 + Y
        
        R = 1.0 - X / denom_R
        L = 1.0 - X / denom_L
        P = 1.0 - X
        S = 0.5 * (R + L)
        D = 0.5 * (R - L)
        
        A = S * sin_t2 + P * cos_t2
        B = R * L * sin_t2 + P * S * (1.0 + cos_t2)
        C = P * R * L
        
        F2 = (R * L - P * S)**2 * (np.sin(theta)**4) + 4 * (P**2) * (D**2) * cos_t2
        F = np.sqrt(np.maximum(0, F2))
        
        n2_1 = (B + F) / (2 * A)
        n2_2 = (B - F) / (2 * A)
        
        # Root 1
        valid_1 = (n2_1 > 0)
        n_1 = np.sqrt(n2_1, where=valid_1, out=np.full_like(n2_1, np.nan))
        k_calc_1 = w_vals * n_1
        k_calc_1[k_calc_1 > k_max_limit] = np.nan
        k1 = k_calc_1

        # Root 2
        valid_2 = (n2_2 > 0)
        n_2 = np.sqrt(n2_2, where=valid_2, out=np.full_like(n2_2, np.nan))
        k_calc_2 = w_vals * n_2
        k_calc_2[k_calc_2 > k_max_limit] = np.nan
        k2 = k_calc_2
        
    return k1, k2

def plot_panel(ax, w_pe, w_ce, x_lim, y_lim, title):
    # 增加线条密度以获得更平滑的效果
    thetas = np.linspace(0, 90, 91)
    w_vals = np.linspace(0.001, y_lim, 2000)
    
    cmap = cm.jet
    norm = mcolors.Normalize(vmin=0, vmax=90)
    
    # 光速线
    ax.plot([0, x_lim], [0, x_lim], 'k-', lw=0.5, alpha=0.8)
    
    for theta in thetas:
        color = cmap(norm(theta))
        k1, k2 = get_dispersion_k(w_vals, w_pe, w_ce, theta, k_max_limit=x_lim * 1.2)
        
        ax.plot(k1, w_vals, color=color, linewidth=0.6)
        ax.plot(k2, w_vals, color=color, linewidth=0.6)

    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)
    ax.set_xlabel(r'$kc/\Omega_e$', fontsize=14)
    ax.set_ylabel(r'$\omega/\Omega_e$', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.tick_params(direction='in', top=True, right=True)

def main():
    # 创建画布，预留稍大的空间
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 绘图
    plot_panel(ax1, w_pe=2.0, w_ce=1.0, x_lim=6.0, y_lim=3.0, 
               title=r'$\omega_{pe}/\Omega_e=2.0$')
    
    plot_panel(ax2, w_pe=0.5, w_ce=1.0, x_lim=3.0, y_lim=1.5, 
               title=r'$\omega_{pe}/\Omega_e=0.5$')
    
    # --- 布局调整核心代码 ---
    
    # 1. 调整子图布局，右侧留出 15% 的空白给 colorbar (right=0.85)
    #    wspace 调整两张图之间的间距
    plt.subplots_adjust(left=0.08, right=0.85, bottom=0.12, top=0.9, wspace=0.25)
    
    # 2. 手动添加一个新的 Axes 用于放置 Colorbar
    #    参数格式: [left, bottom, width, height] (归一化坐标 0-1)
    #    这里放在 x=0.87 的位置，宽度 0.02
    cax = fig.add_axes([0.87, 0.12, 0.02, 0.78])
    
    # 3. 创建 Colorbar 并关联到这个新的 axes (cax)
    sm = cm.ScalarMappable(cmap=cm.jet, norm=mcolors.Normalize(vmin=0, vmax=90))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    
    # 设置标签
    cbar.set_label(r'$\theta$ (Deg)', fontsize=14)
    cbar.set_ticks([0, 30, 60, 90])
    
    # 显示
    plt.show()

if __name__ == "__main__":
    main()
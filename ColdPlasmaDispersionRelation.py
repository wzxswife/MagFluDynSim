import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def calculate_dispersion(w_vals, w_pe, w_ce, theta_deg):
    """
    根据给定的频率数组 w_vals，计算对应的归一化波数 kc/Omega_e。
    w_vals: 归一化频率 w/Omega_e
    w_pe:   归一化电子等离子体频率 w_pe/Omega_e
    w_ce:   归一化电子回旋频率 (在此归一化下通常为 1.0)
    theta_deg: 波矢量与磁场夹角 (度)
    """
    # 将角度转换为弧度
    theta = np.deg2rad(theta_deg)
    sin_t2 = np.sin(theta)**2
    cos_t2 = np.cos(theta)**2
    
    # 初始化输出数组 (两个根)
    k_norm_1 = np.full_like(w_vals, np.nan)
    k_norm_2 = np.full_like(w_vals, np.nan)

    # 遍历频率计算 (向量化计算可能会遇到除以零，这里为了清晰分步处理)
    # 计算 Stix 参数
    # X = w_pe^2 / w^2
    # Y = w_ce / w
    # R = 1 - X / (1 - Y)
    # L = 1 - X / (1 + Y)
    # P = 1 - X
    # S = (R + L) / 2
    # D = (R - L) / 2
    
    # 避免 w=0 或 w=w_ce 处的除零错误，使用 mask 或忽略警告
    with np.errstate(divide='ignore', invalid='ignore'):
        X = (w_pe / w_vals)**2
        Y = w_ce / w_vals
        
        # Stix parameters
        R = 1.0 - X / (1.0 - Y)
        L = 1.0 - X / (1.0 + Y)
        P = 1.0 - X
        S = 0.5 * (R + L)
        D = 0.5 * (R - L)
        
        # 图片公式 (5.45) - (5.47)
        A = S * sin_t2 + P * cos_t2
        B = R * L * sin_t2 + P * S * (1.0 + cos_t2)
        C = P * R * L
        
        # 图片公式 (5.49) - 判别式 F^2
        # F2 = (B^2 - 4AC)
        F2 = (R * L - P * S)**2 * (np.sin(theta)**4) + 4 * (P**2) * (D**2) * cos_t2
        
        # 计算 n^2 (公式 5.48)
        # 注意：如果 F2 < 0 (数学上不应在无耗冷等离子体中发生，除非数值误差)，取 abs
        F = np.sqrt(np.abs(F2))
        
        n2_a = (B + F) / (2 * A)
        n2_b = (B - F) / (2 * A)
        
        # 计算归一化 k = (w/Omega_e) * n
        # k = (w/c) * n  =>  kc/Omega_e = (w/Omega_e) * n
        
        # 只取实部传播模式 (n^2 > 0)
        mask_a = n2_a > 0
        mask_b = n2_b > 0
        
        k_norm_1[mask_a] = w_vals[mask_a] * np.sqrt(n2_a[mask_a])
        k_norm_2[mask_b] = w_vals[mask_b] * np.sqrt(n2_b[mask_b])
        
    return k_norm_1, k_norm_2

def plot_dispersion_panel(ax, w_pe_val, w_ce_val, theta_list, x_limit, y_limit, title_str):
    # 设置颜色映射 (Blue -> Green -> Red)
    cmap = cm.jet
    norm = mcolors.Normalize(vmin=0, vmax=90)
    
    # 为了画图平滑且不连接回旋共振处的断点，我们将频率分为几段
    # 这里的断点主要在 w/Omega_e = 1 (回旋共振)
    w_range1 = np.linspace(0.01, 0.99, 500) # 低频段
    w_range2 = np.linspace(1.01, y_limit, 500) # 高频段
    
    # 光速线 (k = w/c => kc/Omega_e = w/Omega_e)
    ax.plot([0, x_limit], [0, x_limit], 'k-', linewidth=0.8, alpha=0.7)
    
    for theta in theta_list:
        color = cmap(norm(theta))
        
        # 计算低频段
        k1_low, k2_low = calculate_dispersion(w_range1, w_pe_val, w_ce_val, theta)
        # 计算高频段
        k1_high, k2_high = calculate_dispersion(w_range2, w_pe_val, w_ce_val, theta)
        
        # 绘制线条 (分别绘制 k1 和 k2 两个根)
        # 过滤掉极大的 k 值以防画图混乱
        limit_mask = 100 
        
        # Low freq
        ax.plot(k1_low[k1_low < limit_mask], w_range1[k1_low < limit_mask], color=color, linewidth=0.8)
        ax.plot(k2_low[k2_low < limit_mask], w_range1[k2_low < limit_mask], color=color, linewidth=0.8)
        
        # High freq
        ax.plot(k1_high[k1_high < limit_mask], w_range2[k1_high < limit_mask], color=color, linewidth=0.8)
        ax.plot(k2_high[k2_high < limit_mask], w_range2[k2_high < limit_mask], color=color, linewidth=0.8)

    # 设置坐标轴
    ax.set_xlim(0, x_limit)
    ax.set_ylim(0, y_limit)
    ax.set_xlabel(r'$kc/\Omega_e$', fontsize=14)
    ax.set_ylabel(r'$\omega/\Omega_e$', fontsize=14)
    ax.set_title(title_str, fontsize=16)
    
    # 增加刻度处理
    ax.tick_params(direction='in', top=True, right=True)

# 主程序
def main():
    # 参数设置
    w_ce = 1.0 # 归一化参考
    thetas = np.linspace(0, 90, 50) # 0到90度，取50条线
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 左图: w_pe / Omega_e = 2.0
    plot_dispersion_panel(ax1, w_pe_val=2.0, w_ce_val=w_ce, theta_list=thetas, 
                          x_limit=6, y_limit=3, 
                          title_str=r'$\omega_{pe}/\Omega_e=2.0$')
    
    # 右图: w_pe / Omega_e = 0.5
    plot_dispersion_panel(ax2, w_pe_val=0.5, w_ce_val=w_ce, theta_list=thetas, 
                          x_limit=3, y_limit=1.5, 
                          title_str=r'$\omega_{pe}/\Omega_e=0.5$')

    # 添加 Colorbar
    sm = cm.ScalarMappable(cmap=cm.jet, norm=mcolors.Normalize(vmin=0, vmax=90))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='vertical', fraction=0.05, pad=0.02)
    cbar.set_label(r'$\theta$ (Deg)', fontsize=14)
    cbar.set_ticks([0, 30, 60, 90])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
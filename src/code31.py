import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'

g = 5/3
s = [1.5, 0.5]

sin15 = np.sin(np.radians(15))
cos15 = np.cos(np.radians(15))

x1 = 2/(g-1) * sin15

x2 = 1/(2*(g-1) - 0.5*g**2*sin15**2) * (
     sin15*(2-g)*(1+s[1]) + 2*cos15*np.sqrt(
     (g-1)*(1-s[1])**2 + s[1]*g**2*sin15**2))

y3 = (1 + s[1]*(g-1))*sin15 / ((1-s[1])*(g-1) - g*sin15**2)

h = np.arange(0, 1.001, 0.001)
N = len(h)

y1 = np.full((2, N), np.nan)
y2 = np.full((2, N), np.nan)
h1 = np.tile(h, (2,1)).astype(float)
h2 = np.tile(h, (2,1)).astype(float)

for j in range(2):
    for i in range(N):
        B  = (g/2)*h[i]*sin15 - (1 - s[j])
        C  = 2*sin15 - (g-1)*h[i]
        RX = B**2 + C*(h[i] + 2*s[j]*sin15)
        if RX >= 0 and abs(C) > 1e-14:
            v1 = (B + np.sqrt(RX)) / C
            v2 = (B - np.sqrt(RX)) / C
            if v1 < 0: h1[j,i] = np.nan
            else:       y1[j,i] = v1
            if v2 < 0: h2[j,i] = np.nan
            else:       y2[j,i] = v2
        else:
            h1[j,i] = np.nan; h2[j,i] = np.nan

# ── 绘图──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor('white')

# Asymptote line (dashed) at the top
ax.plot(h1[0,:], y1[0,:], 'k',   lw=1.5)

ax.plot(h1[1,:], y1[1,:], 'k--', lw=1.2)  # Dashed line for asymptote
ax.plot(h2[1,:], y2[1,:], 'k--', lw=1.2)

ax.plot([x1, x1], [-50, 250], 'k-.',  lw=0.8)
ax.plot([x2, x2], [-50, 250], 'k:',   lw=0.9)
ax.plot([0, 1.05], [0, 0], 'k:', lw=1.0)
ax.plot([0, 0], [0, 0], 'k', lw=1.0)

ax.set_xlim([0, 1])
ax.set_ylim([-50, 250])
ax.set_xticks([]); ax.set_yticks([])

# Modify axis positions: move the x-axis to the bottom
for sp in ['right', 'top']: ax.spines[sp].set_visible(False)
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('outward', 40))

# 坐标轴箭头
ax.annotate('', xy=(1.04, 0), xytext=(-0.01, 0),
    arrowprops=dict(arrowstyle='->', color='k', lw=1.0))
ax.annotate('', xy=(0, 245), xytext=(0, -45),
    arrowprops=dict(arrowstyle='->', color='k', lw=1.0))

# ── 文字标注（对应 MATLAB text(...)）─────────────────────────────────────
ax.text(1.01, 0,
    r'$\frac{[1+s_0(\gamma-1)]\sin\theta_0}{(1-s_0)(\gamma-1)-\gamma\sin^2\theta_0}$',
    fontsize=12, ha='left', va='center', clip_on=False)

ax.text(0.02, 12,
    r'$s_0\geq1-\gamma\sin^2\theta_0/(\gamma-1)$',
    fontsize=8.5, rotation=2, ha='left')

ax.text(0.02, -10,
    r'$s_0<1-\gamma\sin^2\theta_0/(\gamma-1)$',
    fontsize=8.5, ha='left')

ax.text(-0.35, 6,
    r'$\frac{\sqrt{(1-s_0)^2+4s_0\sin^2\theta_0}-(1-s_0)}{2\sin\theta_0}$',
    fontsize=12, ha='left', va='center', clip_on=False)

ax.annotate('', xy=(0, 5), xytext=(-0.05, 5),
    arrowprops=dict(arrowstyle='->', color='k', lw=0.8))
ax.annotate('', xy=(0, 1), xytext=(-0.05, 1),
    arrowprops=dict(arrowstyle='->', color='k', lw=0.8))

ax.text(x1-0.01, -65, r'$\hat{h}_f$',
    fontsize=12, ha='center', clip_on=False)
ax.text(x2-0.01, -65, r'$\hat{\hat{h}}_f$',
    fontsize=12, ha='center', clip_on=False)
ax.text(1.02, -60, r'$h_f$',
    fontsize=10, ha='left', clip_on=False)
ax.text(-0.08, 130, r'$\frac{X^\pm_f}{h_f}$',
    fontsize=12, ha='center', clip_on=False)

# ∞ 和箭头
ax.text(x1+0.012, 195, r'$\infty$', fontsize=12, ha='left')
ax.annotate('', xy=(x1+0.016, 185), xytext=(x1+0.016, 155),
    arrowprops=dict(arrowstyle='->', color='k', lw=0.9))

# X+/hf 标注
ax.text(0.76, 8,  r'$\frac{X^+_f}{h_f}$', fontsize=12, ha='center')
ax.text(x1-0.03, 100, r'$\frac{X^+_f}{h_f}$', fontsize=12, ha='right')

# X-/hf 标注
idx = np.where(np.isfinite(y2[1,:]) & (h > x1) & (y2[1,:] < 180))[0]
if len(idx) > 5:
    mid = idx[len(idx)//4]
    ax.text(h[mid]+0.01, y2[1,mid]+12,
        r'$\frac{X^-_f}{h_f}$', fontsize=12, ha='left')

ax.text(-0.02, -65, '0', fontsize=9, ha='center', clip_on=False)

plt.savefig('FShock.pdf', format='pdf', dpi=150, bbox_inches='tight')
plt.savefig('FShock.png', format='png', dpi=150, bbox_inches='tight')
plt.show()
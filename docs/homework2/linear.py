
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.size'] = 13
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14


def initial0(x):
    if x < -0.4 or (-0.2 <= x < -0.1) or x >= 0.0:
        return 0.0
    if -0.1 <= x < 0.0:
        return 1.0
    if -0.4 <= x < -0.2:
        return 1.0 - abs(x + 0.3) / 0.1


def upwind_scheme(u0, C, delta_x, t):
    u = np.zeros((int(t / C / delta_x), len(u0)))
    u[0, :] = u0
    for i in range(1, int(t / C / delta_x)):
        u[i, :] = [u[i - 1, j] - C * (u[i - 1, j] - u[i - 1, j - 1])
                   for j in range(len(u0))]
    return u


def lax_wendroff_scheme(u0, C, delta_x, t):
    u1 = np.zeros((int(t / C / delta_x), len(u0)))
    u1[0, :] = u0
    for i in range(1, int(t / C / delta_x)):
        u1[i, 0:len(u0) - 1] = [
            u1[i - 1, j]
            - 0.5 * C * (u1[i - 1, j + 1] - u1[i - 1, j - 1])
            + 0.5 * C * C * (u1[i - 1, j + 1] - 2 * u1[i - 1, j] + u1[i - 1, j - 1])
            for j in range(len(u0) - 1)
        ]
    return u1


def minmod(a, b):
    if a * b <= 0:
        return 0
    if abs(a) <= abs(b):
        return a
    return b


def upwind_scheme_minmod(u0, C, delta_x, t):
    u = np.zeros((int(t / C / delta_x), len(u0)))
    u[0, :] = u0
    for i in range(1, int(t / C / delta_x)):
        u[i, 0:len(u0) - 1] = [
            u[i - 1, j]
            - C * (u[i - 1, j] - u[i - 1, j - 1])
            - 0.5 * C * (1 - C) * (
                minmod(u[i - 1, j] - u[i - 1, j - 1], u[i - 1, j + 1] - u[i - 1, j])
                - minmod(u[i - 1, j - 1] - u[i - 1, j - 2], u[i - 1, j] - u[i - 1, j - 1])
            )
            for j in range(len(u0) - 1)
        ]
    return u


def main():
    x = np.linspace(-1, 2, 517)
    delta_x = x[1] - x[0]
    u0 = np.array([initial0(i) for i in x])
    Exact = np.array([initial0(i - 0.5) for i in x])
    t = 0.5

    fig = plt.figure(figsize=(12, 12))
    axs = []

    axs.append(fig.add_subplot(321))
    axs[0].plot(x, u0, '--', linewidth=1.0, color='dimgray',label='Initial value')
    axs[0].plot(x, Exact, '-', linewidth=1.0, color='black',label='Exact solution')
    axs[0].plot(x, upwind_scheme(u0, 0.05, delta_x, t)[int(0.5 / 0.05 / delta_x) - 1, :],
                'o', markersize=2.9,markerfacecolor='none',
            markeredgewidth=1.0,color='royalblue', label='Numerical solution')
    axs[0].set_title(r'Upwind $C=0.05$, Time$=0.5$')
    axs[0].text(-1, 0.6, r'$(a)$', fontsize=15)
    axs[0].legend(loc='upper right', fontsize=8)
    axs[0].set_xlim(-1, 2)
    axs[0].set_ylim(0, 1.02)

    axs.append(fig.add_subplot(322))
    axs[1].plot(x, u0, '--', linewidth=1.0,color='dimgray', label='Initial value')
    axs[1].plot(x, Exact, '-', linewidth=1.0, color='black',label='Exact solution')
    axs[1].plot(x, upwind_scheme(u0, 0.5, delta_x, t)[int(0.5 / 0.5 / delta_x) - 1, :],
                'o', markersize=2.9, markerfacecolor='none',
            markeredgewidth=1.0,color='royalblue',label='Numerical solution')
    axs[1].set_title(r'Upwind $C=0.5$, Time$=0.5$')
    axs[1].text(-1, 0.6, r'$(b)$', fontsize=15)
    axs[1].legend(loc='upper right', fontsize=8)
    axs[1].set_xlim(-1, 2)
    axs[1].set_ylim(0, 1.02)

    axs.append(fig.add_subplot(323))
    axs[2].plot(x, u0, '--', linewidth=1.0,color='dimgray', label='Initial value')
    axs[2].plot(x, Exact, '-', linewidth=1.0, color='black',label='Exact solution')
    axs[2].plot(x, upwind_scheme(u0, 0.95, delta_x, t)[int(0.5 / 0.95 / delta_x) - 1, :],
                'o', markersize=2.9, markerfacecolor='none',
            markeredgewidth=1.0,color='royalblue',label='Numerical solution')
    axs[2].set_title(r'Upwind $C=0.95$, Time$=0.5$')
    axs[2].text(-1, 0.6, r'$(c)$', fontsize=15)
    axs[2].legend(loc='upper right', fontsize=8)
    axs[2].set_xlim(-1, 2)
    axs[2].set_ylim(0, 1.02)

    axs.append(fig.add_subplot(324))
    axs[3].plot(x, u0, '--', linewidth=1.0, color='dimgray',label='Initial value')
    axs[3].plot(x, Exact, '-', linewidth=1.0,color='black', label='Exact solution')
    axs[3].plot(x, upwind_scheme(u0, 1, delta_x, t)[int(0.5 / 1 / delta_x) - 1, :],
                'o', markersize=2.9,markerfacecolor='none',
            markeredgewidth=1.0, color='royalblue',label='Numerical solution')
    axs[3].set_title(r'Upwind $C=1$, Time$=0.5$')
    axs[3].text(-1, 0.6, r'$(d)$', fontsize=15)
    axs[3].legend(loc='upper right', fontsize=8)
    axs[3].set_xlim(-1, 2)
    axs[3].set_ylim(0, 1.02)

    axs.append(fig.add_subplot(325))
    axs[4].plot(x, u0, '--', linewidth=1.0,color='dimgray', label='Initial value')
    axs[4].plot(x, Exact, '-', linewidth=1.0, color='black',label='Exact solution')
    axs[4].plot(x, upwind_scheme_minmod(u0, 0.95, delta_x, t)[int(0.5 / 0.95 / delta_x) - 1, :],
                'o', markersize=2.9, markerfacecolor='none',
            markeredgewidth=1.0,color='royalblue',label='Numerical solution')
    axs[4].set_title(r'Upwind minmod $C=0.95$ Time$=0.5$')
    axs[4].text(-1, 0.6, r'$(e)$', fontsize=15)
    axs[4].legend(loc='upper right', fontsize=8)
    axs[4].set_xlim(-1, 2)
    axs[4].set_ylim(0, 1.02)

    axs.append(fig.add_subplot(326))
    axs[5].plot(x, u0, '--', linewidth=1.0, color='dimgray',label='Initial value')
    axs[5].plot(x, Exact, '-', linewidth=1.0, color='black',label='Exact solution')
    axs[5].plot(x, lax_wendroff_scheme(u0, 0.95, delta_x, t)[int(0.5 / 0.95 / delta_x) - 1, :],
                'o', markersize=2.9, markerfacecolor='none',
            markeredgewidth=1.0,color='royalblue',label='Numerical solution')
    axs[5].set_title(r'Lax wendroff $C=0.95$ Time$=0.5$')
    axs[5].text(-1, 0.6, r'$(f)$', fontsize=15)
    axs[5].legend(loc='upper right', fontsize=8)
    axs[5].set_xlim(-1, 2)
    axs[5].set_ylim(-0.2, 1.2)

    plt.tight_layout()
    plt.savefig('linear_results.pdf', bbox_inches='tight')
    plt.savefig("linear_results.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    main()
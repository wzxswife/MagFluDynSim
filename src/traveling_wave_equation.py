"""
Traveling Wave Equation - Lax-Friedrichs Method
Equation: u_t + (1/2 * u^2)_x = 0
Initial condition: u(x,0) = sin(x)
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def traveling_wave_lax_friedrichs():
    """
    Solve traveling wave equation using Lax-Friedrichs method
    u_t + (1/2 * u^2)_x = 0
    """
    # Parameters
    L = 2 * np.pi  # Domain length
    N = 200        # Number of spatial grid points
    dx = L / N     # Spatial step
    dt = 0.005     # Time step
    t_end = 1.5    # End time
    
    # Spatial grid
    x = np.linspace(0, L, N + 1)[:-1]  # Remove last point to avoid duplication
    
    # Initial condition: u(x,0) = sin(x)
    u = np.sin(x)
    
    # Time iteration
    n = 0
    t = 0.0
    
    while t < t_end:
        # Ensure time step does not exceed end time
        if t + dt > t_end:
            dt = t_end - t
        
        # Lax-Friedrichs scheme
        # u^{n+1}_i = 0.5*(u^n_{i+1} + u^n_{i-1}) - dt/(2*dx)*(F^n_{i+1} - F^n_{i-1})
        # where F = 0.5 * u^2
        
        F = 0.5 * u**2
        
        u_new = np.zeros_like(u)
        for i in range(N):
            u_new[i] = 0.5 * (u[(i+1) % N] + u[(i-1) % N]) - \
                       dt / (2 * dx) * (F[(i+1) % N] - F[(i-1) % N])
        
        u = u_new
        t += dt
        n += 1
    
    return x, u, t


def traveling_wave_analytical(t):
    """
    Analytical solution of traveling wave equation
    Using method of characteristics: dx/dt = u, u constant along characteristics
    For initial condition sin(x), analytical solution requires solving implicit equation
    """
    # For initial condition u(x,0) = sin(x), solution via implicit equation
    # x - u*t = xi, u = sin(xi)
    pass


def main():
    print("Traveling Wave Equation - Lax-Friedrichs Method")
    print("=" * 50)
    print("Equation: u_t + (1/2 * u^2)_x = 0")
    print("Initial condition: u(x,0) = sin(x)")
    print("Domain: [0, 2pi], t = 1.5")
    print("=" * 50)
    
    # Numerical solution
    x, u_numerical, t = traveling_wave_lax_friedrichs()
    
    print(f"\nCalculation completed!")
    print(f"Final time: t = {t:.4f}")
    print(f"Number of spatial points: N = {len(x)}")
    print(f"Numerical solution range: [{u_numerical.min():.4f}, {u_numerical.max():.4f}]")
    
    # 绘图
    plt.figure(figsize=(12, 5))
    
    # 数值解
    plt.subplot(1, 2, 1)
    plt.plot(x, u_numerical, 'b-', linewidth=2, label='Numerical Solution')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f'Traveling Wave Equation - Numerical Solution (t={t:.2f})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 初始条件对比
    plt.subplot(1, 2, 2)
    u_initial = np.sin(x)
    plt.plot(x, u_initial, 'r--', linewidth=2, label='Initial: u(x,0)=sin(x)')
    plt.plot(x, u_numerical, 'b-', linewidth=2, label=f'Solution at t={t:.2f}')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Initial Condition vs Numerical Solution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    script_dir = os.path.dirname(__file__)
    save_path = os.path.join(script_dir, '..', 'results', 'traveling_wave_solution.pdf')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    
    # 保存数据
    # np.savez('traveling_wave_result.npz', x=x, u=u_numerical, t=t)
    # print("\nResults saved to traveling_wave_result.npz")


if __name__ == "__main__":
    main()
"""
Burgers Equation - Characteristic Method
Equation: u_t + u*u_x = 0
Initial condition: u(x,0) = sin(x)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def burgers_characteristic_method():
    """
    Solve inviscid Burgers equation using method of characteristics
    u_t + u*u_x = 0
    
    Characteristics: dx/dt = u
    Along characteristics: du/dt = 0, so u is constant
    
    Analytical solution: u(x,t) = u_0(xi), where x = xi + u_0(xi)*t
    """
    # Parameters
    L = 2 * np.pi  # Domain length
    N = 200        # Number of spatial grid points
    t_end = 1.0    # End time
    
    # Spatial grid
    x = np.linspace(0, L, N)
    
    # Initial condition
    u0 = lambda xi: np.sin(xi)
    
    def find_xi(x_val, t_val):
        """
        Solve implicit equation: x = xi + sin(xi)*t
        Using Newton iteration
        """
        if t_val == 0:
            return x_val % (2 * np.pi)
        
        # Initial guess
        xi = x_val % (2 * np.pi)
        
        # Newton iteration
        for _ in range(50):
            f = xi + u0(xi) * t_val - x_val
            f_prime = 1 + np.cos(xi) * t_val
            if abs(f_prime) < 1e-10:
                break
            xi_new = xi - f / f_prime
            if abs(xi_new - xi) < 1e-8:
                xi = xi_new
                break
            xi = xi_new
        
        return xi
    
    # Compute numerical solution
    u = np.zeros_like(x)
    for i in range(N):
        xi = find_xi(x[i], t_end)
        u[i] = u0(xi)
    
    return x, u, t_end


def burgers_lax_friedrichs():
    """
    Solve Burgers equation using Lax-Friedrichs method for comparison
    u_t + (0.5*u^2)_x = 0
    """
    # Parameters
    L = 2 * np.pi
    N = 200
    dx = L / N
    dt = 0.005
    t_end = 1.0
    
    # Spatial grid
    x = np.linspace(0, L, N)
    
    # Initial condition: u(x,0) = sin(x)
    u = np.sin(x)
    
    # Time iteration
    t = 0.0
    
    while t < t_end:
        if t + dt > t_end:
            dt = t_end - t
        
        # Lax-Friedrichs scheme
        # F = 0.5 * u^2
        F = 0.5 * u**2
        
        u_new = np.zeros_like(u)
        for i in range(N):
            u_new[i] = 0.5 * (u[(i+1) % N] + u[(i-1) % N]) - \
                       dt / (2 * dx) * (F[(i+1) % N] - F[(i-1) % N])
        
        u = u_new
        t += dt
    
    return x, u, t


def check_shock_formation():
    """
    Check shock formation
    For Burgers equation, shock forms when u_x < 0
    Initial condition u(x,0) = sin(x) has u_x < 0 in x in (pi, 2pi)
    """
    t_critical = 1.0  # Critical time for shock formation
    print(f"\nShock Analysis:")
    print(f"Initial condition u(x,0) = sin(x) has negative derivative in x in (pi, 2pi)")
    print(f"Shock formation expected at t ~ 1.0")


def main():
    print("Burgers Equation - Characteristic Method")
    print("=" * 50)
    print("Equation: u_t + u*u_x = 0")
    print("Initial condition: u(x,0) = sin(x)")
    print("Domain: [0, 2pi], t = 1.0")
    print("=" * 50)
    
    # Characteristic method analytical solution
    x_char, u_char, t_char = burgers_characteristic_method()
    
    print(f"\nCharacteristic method calculation completed!")
    print(f"Final time: t = {t_char:.4f}")
    print(f"Numerical solution range: [{u_char.min():.4f}, {u_char.max():.4f}]")
    
    # Lax-Friedrichs method numerical solution
    x_lf, u_lf, t_lf = burgers_lax_friedrichs()
    
    print(f"\nLax-Friedrichs method calculation completed!")
    print(f"Final time: t = {t_lf:.4f}")
    print(f"Numerical solution range: [{u_lf.min():.4f}, {u_lf.max():.4f}]")
    
    # 检查激波形成
    check_shock_formation()
    
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 特征线法解
    axes[0].plot(x_char, u_char, 'b-', linewidth=2)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x,t)')
    axes[0].set_title(f'Characteristic Method - Analytical Solution (t={t_char:.2f})')
    axes[0].grid(True, alpha=0.3)
    
    # Lax-Friedrichs解
    axes[1].plot(x_lf, u_lf, 'r-', linewidth=2)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('u(x,t)')
    axes[1].set_title(f'Lax-Friedrichs Method - Numerical Solution (t={t_lf:.2f})')
    axes[1].grid(True, alpha=0.3)
    
    # 初始条件对比
    axes[2].plot(x_char, np.sin(x_char), 'k--', linewidth=2, label='Initial: sin(x)')
    axes[2].plot(x_char, u_char, 'b-', linewidth=2, label='Characteristic')
    axes[2].plot(x_lf, u_lf, 'r-', linewidth=2, label='Lax-Friedrichs')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('u')
    axes[2].set_title('Method Comparison')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    script_dir = os.path.dirname(__file__)
    save_path = os.path.join(script_dir, '..', 'results', 'burgers_equation_solution.pdf')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()
    
    # 保存数据
    # np.savez('burgers_equation_result.npz', 
    #          x_char=x_char, u_char=u_char, t_char=t_char,
    #          x_lf=x_lf, u_lf=u_lf, t_lf=t_lf)
    # print("\nResults saved to burgers_equation_result.npz")


if __name__ == "__main__":
    main()
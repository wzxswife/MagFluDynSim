"""
Traveling Wave Equation - Upwind, Minmod, Lax-Wendroff Methods
Equation: u_t + u_x = 0
Initial condition: piecewise step function
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def initial_condition(x):
    """
    Initial condition for traveling wave equation:
    u(x,0) = {
        0.0,                    x < -0.4,
        1.0 - |x + 0.3|/0.1,   -0.4 <= x < -0.2,
        0.0,                    -0.2 <= x < -0.1,
        1.0,                    -0.1 <= x < 0.0,
        0.0,                    x >= 0.0
    }
    """
    u = np.zeros_like(x)
    
    # u(x,0) = 0.0, x < -0.4
    mask1 = x < -0.4
    u[mask1] = 0.0
    
    # u(x,0) = 1.0 - |x + 0.3|/0.1, -0.4 <= x < -0.2
    mask2 = (x >= -0.4) & (x < -0.2)
    u[mask2] = 1.0 - np.abs(x[mask2] + 0.3) / 0.1
    
    # u(x,0) = 0.0, -0.2 <= x < -0.1
    mask3 = (x >= -0.2) & (x < -0.1)
    u[mask3] = 0.0
    
    # u(x,0) = 1.0, -0.1 <= x < 0.0
    mask4 = (x >= -0.1) & (x < 0.0)
    u[mask4] = 1.0
    
    # u(x,0) = 0.0, x >= 0.0
    mask5 = x >= 0.0
    u[mask5] = 0.0
    
    return u


def exact_solution(x, t, L):
    """
    Exact solution for traveling wave equation u_t + u_x = 0
    Using method of characteristics: dx/dt = 1, so u(x,t) = u_0(x - t)
    Apply periodic boundary conditions
    """
    x_shifted = x - t
    # Apply periodic boundary conditions
    x_shifted = ((x_shifted + L/2) % L) - L/2
    return initial_condition(x_shifted)


def minmod(a, b):
    """
    Minmod limiter function:
    minmod(a, b) = {
        a, if |a| < |b| and ab > 0,
        b, if |a| > |b| and ab > 0,
        0, if ab <= 0
    }
    """
    if a * b > 0:
        if np.abs(a) < np.abs(b):
            return a
        else:
            return b
    else:
        return 0.0


def minmod_vectorized(a, b):
    """
    Vectorized minmod function for arrays
    """
    result = np.zeros_like(a)
    mask = a * b > 0
    result[mask & (np.abs(a) < np.abs(b))] = a[mask & (np.abs(a) < np.abs(b))]
    result[mask & (np.abs(a) >= np.abs(b))] = b[mask & (np.abs(a) >= np.abs(b))]
    return result


def upwind_scheme(u, dx, dt):
    """
    Upwind scheme for u_t + u_x = 0 (with a = 1 > 0)
    u^{n+1}_j = u^n_j - dt/dx * (u^n_j - u^n_{j-1})
    """
    N = len(u)
    u_new = np.zeros_like(u)
    
    for i in range(N):
        u_new[i] = u[i] - dt / dx * (u[i] - u[(i - 1) % N])
    
    return u_new


def minmod_scheme(u, dx, dt):
    """
    Minmod scheme for u_t + u_x = 0 (with a = 1 > 0)
    u^{n+1}_i = u^n_i - dt/dx * (u^n_i - u^n_{i-1}) 
                - 1/2 * dt/dx * (dx - dt) * (sigma^n_i - sigma^n_{i-1})
    where sigma = minmod((u_i - u_{i-1})/dx, (u_{i+1} - u_i)/dx)
    """
    N = len(u)
    C = dt / dx  # Courant number
    
    # Compute sigma values
    sigma = np.zeros(N)
    for i in range(N):
        left = (u[i] - u[(i - 1) % N]) / dx
        right = (u[(i + 1) % N] - u[i]) / dx
        sigma[i] = minmod(left, right)
    
    # Update solution
    u_new = np.zeros_like(u)
    for i in range(N):
        u_new[i] = (u[i] 
                    - C * (u[i] - u[(i - 1) % N])
                    - 0.5 * C * (dx - dt) * (sigma[i] - sigma[(i - 1) % N]))
    
    return u_new


def lax_wendroff_scheme(u, dx, dt):
    """
    Lax-Wendroff scheme for u_t + u_x = 0 (with a = 1 > 0)
    u^{n+1}_j = u^n_j - dt/(2dx)*(u^n_{j+1} - u^n_{j-1}) 
                + 1/2*(dt/dx)^2*(u^n_{j+1} - 2*u^n_j + u^n_{j-1})
    """
    N = len(u)
    C = dt / dx  # Courant number
    
    u_new = np.zeros_like(u)
    for i in range(N):
        u_new[i] = (u[i] 
                    - 0.5 * C * (u[(i + 1) % N] - u[(i - 1) % N])
                    + 0.5 * C**2 * (u[(i + 1) % N] - 2 * u[i] + u[(i - 1) % N]))
    
    return u_new


def solve_equation(scheme_func, x, t_end, C):
    """
    Solve the traveling wave equation using specified scheme
    
    Parameters:
    -----------
    scheme_func : function
        Numerical scheme to use
    x : array
        Spatial grid
    t_end : float
        End time
    C : float
        Courant number (C = a*dt/dx = dt/dx since a=1)
    
    Returns:
    --------
    u : array
        Numerical solution at time t_end
    t : float
        Actual time reached
    """
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    dt = C * dx
    
    u = initial_condition(x)
    t = 0.0
    
    while t < t_end:
        if t + dt > t_end:
            dt = t_end - t
            C = dt / dx
        
        u = scheme_func(u, dx, dt)
        t += dt
    
    return u, t


def plot_results(x, u_initial, schemes_results, t_end, L, N, save_path=None):
    """
    Plot comparison of different schemes
    
    Parameters:
    -----------
    x : array
        Spatial grid
    u_initial : array
        Initial condition
    schemes_results : dict
        Dictionary with scheme names as keys and (u_numerical, label) tuples as values
    t_end : float
        End time
    L : float
        Domain length
    N : int
        Number of grid points
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Exact solution
    u_exact = exact_solution(x, t_end, L)
    
    # Plot each scheme
    for idx, (ax, (scheme_name, (u_num, label))) in enumerate(zip(axes, schemes_results.items())):
        # Plot initial condition (dotted line)
        ax.plot(x, u_initial, 'k:', linewidth=1.5, label='Initial value')
        
        # Plot exact solution (solid line)
        ax.plot(x, u_exact, 'g-', linewidth=1.5, label='Exact solution')
        
        # Plot numerical solution (with circles at grid points)
        ax.plot(x, u_num, 'b-', linewidth=1.5, label='Numerical solution')
        ax.plot(x, u_num, 'bo', markersize=3, markerfacecolor='blue')
        
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.set_title(f'{label}\nTime={t_end}')
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([-0.1, 1.2])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        # Set x-axis ticks to show domain [0, 2π] range
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def main():
    print("Traveling Wave Equation - Numerical Schemes Comparison")
    print("=" * 60)
    print("Equation: u_t + u_x = 0")
    print("Initial condition: piecewise step function")
    print("=" * 60)
    
    # Parameters (matching the assignment)
    L = 2 * np.pi  # Domain length [0, 2π]
    N = 517        # Number of grid points (as specified in assignment)
    dx = L / N
    
    # For the initial condition to work properly, we shift the domain
    # The initial condition is defined on [-π, π], but we simulate on [0, 2π]
    # Shift x to be centered at 0
    x = np.linspace(0, L, N + 1)[:-1]  # Remove duplicate point
    x_shifted = x - np.pi  # Shift to [-π, π]
    
    # Time parameter
    t_end = 0.5  # End time as specified in assignment
    
    # Test different Courant numbers
    C_values = [0.05, 0.5, 0.95, 1.0]
    
    # Initial condition
    u_initial = initial_condition(x_shifted)
    
    print(f"\nParameters:")
    print(f"  Domain: [0, 2π] (shifted to [-π, π] for IC)")
    print(f"  Grid points: N = {N}")
    print(f"  Spatial step: dx = {dx:.6f}")
    print(f"  End time: t = {t_end}")
    print(f"  Courant numbers: {C_values}")
    print()
    
    # Solve with different schemes
    schemes_results = {}
    
    # Upwind scheme with different C values
    for C in C_values:
        u_num, t = solve_equation(upwind_scheme, x_shifted, t_end, C)
        schemes_results[f'upwind_C{C}'] = (u_num, f'Upwind C={C}')
        print(f"Upwind C={C}: t = {t:.4f}, u range = [{u_num.min():.4f}, {u_num.max():.4f}]")
    
    # Minmod scheme (C = 0.95)
    C_minmod = 0.95
    u_minmod, t = solve_equation(minmod_scheme, x_shifted, t_end, C_minmod)
    schemes_results['minmod'] = (u_minmod, f'Upwind minmod C={C_minmod}')
    print(f"Minmod C={C_minmod}: t = {t:.4f}, u range = [{u_minmod.min():.4f}, {u_minmod.max():.4f}]")
    
    # Lax-Wendroff scheme (C = 0.95)
    C_lw = 0.95
    u_lw, t = solve_equation(lax_wendroff_scheme, x_shifted, t_end, C_lw)
    schemes_results['lax_wendroff'] = (u_lw, f'Lax wendroff C={C_lw}')
    print(f"Lax-Wendroff C={C_lw}: t = {t:.4f}, u range = [{u_lw.min():.4f}, {u_lw.max():.4f}]")
    
    # Reorder results to match Linear.pdf layout
    ordered_results = [
        ('upwind_C0.05', schemes_results['upwind_C0.05']),
        ('upwind_C0.5', schemes_results['upwind_C0.5']),
        ('upwind_C0.95', schemes_results['upwind_C0.95']),
        ('upwind_C1.0', schemes_results['upwind_C1.0']),
        ('minmod', schemes_results['minmod']),
        ('lax_wendroff', schemes_results['lax_wendroff']),
    ]
    
    # Save path
    script_dir = os.path.dirname(__file__)
    save_path = os.path.join(script_dir, '..', 'results', 'Linear.pdf')
    
    # Ensure results directory exists
    results_dir = os.path.join(script_dir, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert to dict for plotting
    schemes_dict = dict(ordered_results)
    
    # Plot results
    plot_results(x_shifted, u_initial, schemes_dict, t_end, L, N, save_path)
    
    print(f"\nCalculation completed!")
    print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    main()

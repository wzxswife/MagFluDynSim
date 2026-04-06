"""
Burgers Equation - Numerical Methods Comparison
Assignment 2: Upwind, Minmod, and Lax-Wendroff Schemes

Equation: u_t + (0.5*u^2)_x = 0
Initial condition: u(x,0) = exp(-x^2)
Boundary conditions: u(0,t) = u(2,t) = 0
Domain: [0, 2], t = 0.6
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def interpolate_linear(x, y, x_new):
    """
    Simple linear interpolation without scipy
    """
    y_new = np.zeros_like(x_new)
    for i, xi in enumerate(x_new):
        # Find the correct interval
        for j in range(len(x) - 1):
            if x[j] <= xi <= x[j+1]:
                t = (xi - x[j]) / (x[j+1] - x[j])
                y_new[i] = y[j] + t * (y[j+1] - y[j])
                break
        else:
            # Extrapolation
            if xi < x[0]:
                y_new[i] = y[0]
            else:
                y_new[i] = y[-1]
    return y_new


def minmod(a, b):
    """
    Minmod limiter function
    minmod(a, b) = {
        0, if a*b < 0
        sign(a)*min(|a|, |b|), otherwise
    }
    """
    if a * b <= 0:
        return 0.0
    return np.sign(a) * min(abs(a), abs(b))


def minmod_slope(u_left, u_center, u_right):
    """
    Calculate minmod slope for TVD schemes
    sigma = minmod((u_center - u_left)/dx, (u_right - u_center)/dx)
    """
    dx = 1.0  # Will be properly set in the main function
    sigma_left = (u_center - u_left)
    sigma_right = (u_right - u_center)
    return minmod(sigma_left, sigma_right)


def initial_condition(x):
    """
    Initial condition: u(x,0) = exp(-x^2)
    """
    return np.exp(-x**2)


def exact_solution(x, t):
    """
    Exact solution using method of characteristics for inviscid Burgers equation
    u_t + u*u_x = 0
    
    Characteristics: dx/dt = u
    Along characteristic: u(x,t) = u_0(xi) where x = xi + u_0(xi)*t
    
    For u_0(xi) = exp(-xi^2), we solve: x = xi + exp(-xi^2)*t
    Using Newton iteration to find xi for each x
    """
    u0 = lambda xi: np.exp(-xi**2)
    
    def find_xi(x_val, t_val):
        if t_val == 0:
            return x_val
        
        # Initial guess
        xi = x_val
        
        # Newton iteration: f(xi) = xi + u0(xi)*t - x = 0
        for _ in range(100):
            f = xi + u0(xi) * t_val - x_val
            f_prime = 1 - 2 * xi**2 * t_val * np.exp(-xi**2)
            if abs(f_prime) < 1e-12:
                break
            xi_new = xi - f / f_prime
            if abs(xi_new - xi) < 1e-10:
                xi = xi_new
                break
            xi = xi_new
        
        return xi
    
    u = np.zeros_like(x)
    for i in range(len(x)):
        xi = find_xi(x[i], t)
        u[i] = u0(xi)
    
    return u


def upwind_scheme(N=200, dt=0.0005, t_end=0.6):
    """
    Upwind scheme for Burgers equation using Rusanov flux
    First-order accurate in space and time
    """
    L = 2.0
    dx = L / N
    
    # Spatial grid
    x = np.linspace(0, L, N)
    
    # Initial condition
    u = initial_condition(x)
    
    # Time iteration
    t = 0.0
    
    while t < t_end:
        if t + dt > t_end:
            dt = t_end - t
        
        # CFL condition check
        u_max = max(1e-10, max(abs(u)))
        cfl = u_max * dt / dx
        if cfl > 0.45:
            dt = 0.45 * dx / u_max
        
        # Compute Rusanov flux at cell interfaces
        F = 0.5 * u**2  # Flux
        F_half = np.zeros(N + 1)
        
        for i in range(N + 1):
            if i == 0:
                u_left = 0.0
                u_right = u[0]
            elif i == N:
                u_left = u[N-1]
                u_right = 0.0
            else:
                u_left = u[i-1]
                u_right = u[i]
            
            alpha = max(abs(u_left), abs(u_right))
            F_left = 0.5 * u_left**2
            F_right = 0.5 * u_right**2
            
            F_half[i] = 0.5 * (F_left + F_right) - 0.5 * alpha * (u_right - u_left)
        
        # Update using conservative form
        u_new = np.zeros_like(u)
        for i in range(1, N-1):
            u_new[i] = u[i] - dt / dx * (F_half[i+1] - F_half[i])
        
        # Boundary conditions
        u_new[0] = 0.0
        u_new[-1] = 0.0
        
        u = u_new
        t += dt
    
    return x, u, t


def minmod_scheme(N=200, dt=0.0005, t_end=0.6):
    """
    Minmod scheme (TVD) for Burgers equation using MUSCL
    Second-order accurate in space using minmod limiter
    """
    L = 2.0
    dx = L / N
    
    # Spatial grid
    x = np.linspace(0, L, N)
    
    # Initial condition
    u = initial_condition(x)
    
    # Time iteration
    t = 0.0
    
    while t < t_end:
        if t + dt > t_end:
            dt = t_end - t
        
        # CFL condition check
        u_max = max(1e-10, max(abs(u)))
        cfl = u_max * dt / dx
        if cfl > 0.45:
            dt = 0.45 * dx / u_max
        
        # Reconstruct left and right states at cell interfaces using minmod
        u_left_states = np.zeros(N + 1)
        u_right_states = np.zeros(N + 1)
        
        for i in range(N + 1):
            if i == 0:
                # Left boundary
                u_left_states[i] = 0.0
                # Slope for right state
                delta = u[0]  # u[0] - u_boundary(0)
                u_right_states[i] = u[0] - 0.5 * minmod(delta, 2*delta)
            elif i == N:
                # Right boundary
                u_left_states[i] = u[N-1] + 0.5 * minmod(u[N-1] - u[N-2], 2*(u[N-1] - u[N-2]))
                u_right_states[i] = 0.0
            else:
                # Interior points
                # Minmod slope
                delta_left = u[i] - u[i-1]
                delta_right = u[i+1] - u[i] if i < N-1 else u[i] - u[i-1]
                
                sigma = minmod(delta_left, delta_right)
                
                u_left_states[i] = u[i-1] + 0.5 * sigma
                u_right_states[i] = u[i] - 0.5 * sigma
        
        # Compute Rusanov flux at interfaces
        F_half = np.zeros(N + 1)
        for i in range(N + 1):
            alpha = max(abs(u_left_states[i]), abs(u_right_states[i]))
            F_left = 0.5 * u_left_states[i]**2
            F_right = 0.5 * u_right_states[i]**2
            F_half[i] = 0.5 * (F_left + F_right) - 0.5 * alpha * (u_right_states[i] - u_left_states[i])
        
        # Update
        u_new = np.zeros_like(u)
        for i in range(1, N-1):
            u_new[i] = u[i] - dt / dx * (F_half[i+1] - F_half[i])
        
        # Boundary conditions
        u_new[0] = 0.0
        u_new[-1] = 0.0
        
        u = u_new
        t += dt
    
    return x, u, t


def lax_wendroff_scheme(N=200, dt=0.0005, t_end=0.6):
    """
    Lax-Wendroff scheme for Burgers equation
    Second-order accurate in space and time
    
    Conservative form:
    u_i^{n+1} = u_i^n - dt/dx * (F_{i+1/2} - F_{i-1/2})
    where F_{i+1/2} is the Rusanov flux
    """
    L = 2.0
    dx = L / N
    
    # Spatial grid
    x = np.linspace(0, L, N)
    
    # Initial condition
    u = initial_condition(x)
    
    # Time iteration
    t = 0.0
    
    while t < t_end:
        if t + dt > t_end:
            dt = t_end - t
        
        # CFL condition check - be conservative
        u_max = max(1e-10, max(abs(u)))
        cfl = u_max * dt / dx
        if cfl > 0.45:
            dt = 0.45 * dx / u_max
        
        # Compute Rusanov flux at cell interfaces
        # F_{i+1/2} = 0.5 * (F(u_i) + F(u_{i+1})) - 0.5 * alpha * (u_{i+1} - u_i)
        # where alpha = max(|u_i|, |u_{i+1}|)
        
        F = 0.5 * u**2  # Flux
        
        # Compute interface fluxes
        F_half = np.zeros(N + 1)
        for i in range(N + 1):
            if i == 0:
                u_left = 0.0
                u_right = u[0]
            elif i == N:
                u_left = u[N-1]
                u_right = 0.0
            else:
                u_left = u[i-1]
                u_right = u[i]
            
            alpha = max(abs(u_left), abs(u_right))
            F_left = 0.5 * u_left**2
            F_right = 0.5 * u_right**2
            
            F_half[i] = 0.5 * (F_left + F_right) - 0.5 * alpha * (u_right - u_left)
        
        # Update
        u_new = np.zeros_like(u)
        for i in range(1, N-1):
            u_new[i] = u[i] - dt / dx * (F_half[i+1] - F_half[i])
        
        # Boundary conditions
        u_new[0] = 0.0
        u_new[-1] = 0.0
        
        u = u_new
        t += dt
    
    return x, u, t


def main():
    print("=" * 60)
    print("Burgers Equation - Numerical Methods Comparison")
    print("=" * 60)
    print("Equation: u_t + (0.5*u^2)_x = 0")
    print("Initial condition: u(x,0) = exp(-x^2)")
    print("Boundary conditions: u(0,t) = u(2,t) = 0")
    print("Domain: [0, 2], t = 0.6")
    print("=" * 60)
    
    # Parameters
    N = 200
    dt = 0.001
    t_end = 0.6
    
    print(f"\nGrid: N = {N}, dt = {dt}, t_end = {t_end}")
    
    # Calculate exact solution
    x_exact = np.linspace(0, 2, 500)
    print("\nCalculating exact solution (method of characteristics)...")
    u_exact = exact_solution(x_exact, t_end)
    
    # Upwind scheme
    print("Computing Upwind scheme...")
    x_up, u_up, t_up = upwind_scheme(N, dt, t_end)
    
    # Minmod scheme
    print("Computing Minmod scheme...")
    x_min, u_min, t_min = minmod_scheme(N, dt, t_end)
    
    # Lax-Wendroff scheme
    print("Computing Lax-Wendroff scheme...")
    x_lw, u_lw, t_lw = lax_wendroff_scheme(N, dt, t_end)
    
    # Print results summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'Method':<20} {'Min':<12} {'Max':<12} {'L2 Error vs Exact':<20}")
    print("-" * 60)
    print(f"{'Exact':<20} {u_exact.min():<12.6f} {u_exact.max():<12.6f} {'-':<20}")
    
    # Interpolate exact solution for error calculation
    u_up_exact = interpolate_linear(x_exact, u_exact, x_up)
    u_min_exact = interpolate_linear(x_exact, u_exact, x_min)
    u_lw_exact = interpolate_linear(x_exact, u_exact, x_lw)
    
    # L2 norm error
    dx = 2.0 / N
    l2_up = np.sqrt(np.sum((u_up - u_up_exact)**2) * dx)
    l2_min = np.sqrt(np.sum((u_min - u_min_exact)**2) * dx)
    l2_lw = np.sqrt(np.sum((u_lw - u_lw_exact)**2) * dx)
    
    print(f"{'Upwind':<20} {u_up.min():<12.6f} {u_up.max():<12.6f} {l2_up:<20.6f}")
    print(f"{'Minmod':<20} {u_min.min():<12.6f} {u_min.max():<12.6f} {l2_min:<20.6f}")
    print(f"{'Lax-Wendroff':<20} {u_lw.min():<12.6f} {u_lw.max():<12.6f} {l2_lw:<20.6f}")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: All methods comparison
    ax1 = axes[0, 0]
    ax1.plot(x_exact, u_exact, 'k-', linewidth=2.5, label='Exact')
    ax1.plot(x_up, u_up, 'b--', linewidth=1.5, label='Upwind')
    ax1.plot(x_min, u_min, 'r-.', linewidth=1.5, label='Minmod')
    ax1.plot(x_lw, u_lw, 'g:', linewidth=1.5, label='Lax-Wendroff')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('u(x,t)', fontsize=12)
    ax1.set_title(f'Comparison of All Methods (t = {t_end})', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 2])
    ax1.set_ylim([-0.1, 1.1])
    
    # Plot 2: Upwind vs Exact
    ax2 = axes[0, 1]
    ax2.plot(x_exact, u_exact, 'k-', linewidth=2.5, label='Exact')
    ax2.plot(x_up, u_up, 'b--', linewidth=1.5, label=f'Upwind (L2={l2_up:.4f})')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('u(x,t)', fontsize=12)
    ax2.set_title('Upwind Scheme vs Exact Solution', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 2])
    ax2.set_ylim([-0.1, 1.1])
    
    # Plot 3: Minmod vs Exact
    ax3 = axes[1, 0]
    ax3.plot(x_exact, u_exact, 'k-', linewidth=2.5, label='Exact')
    ax3.plot(x_min, u_min, 'r-.', linewidth=1.5, label=f'Minmod (L2={l2_min:.4f})')
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('u(x,t)', fontsize=12)
    ax3.set_title('Minmod Scheme vs Exact Solution', fontsize=14)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 2])
    ax3.set_ylim([-0.1, 1.1])
    
    # Plot 4: Lax-Wendroff vs Exact
    ax4 = axes[1, 1]
    ax4.plot(x_exact, u_exact, 'k-', linewidth=2.5, label='Exact')
    ax4.plot(x_lw, u_lw, 'g:', linewidth=1.5, label=f'Lax-Wendroff (L2={l2_lw:.4f})')
    ax4.set_xlabel('x', fontsize=12)
    ax4.set_ylabel('u(x,t)', fontsize=12)
    ax4.set_title('Lax-Wendroff Scheme vs Exact Solution', fontsize=14)
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 2])
    ax4.set_ylim([-0.1, 1.1])
    
    plt.suptitle('Burgers Equation: u_t + (0.5*u²)_x = 0\nInitial: u(x,0) = exp(-x²), t = 0.6', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    script_dir = os.path.dirname(__file__)
    save_path = os.path.join(script_dir, '..', 'results', 'Burgers.pdf')
    
    # Ensure results directory exists
    results_dir = os.path.dirname(save_path)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    
    # Also save as PNG for preview
    png_path = save_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Preview saved to: {png_path}")
    
    plt.close()
    
    print("\n" + "=" * 60)
    print("Computation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

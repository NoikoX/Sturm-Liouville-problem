import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

def runge_kutta_4(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h/2, y + h*k1/2)
    k3 = f(x + h/2, y + h*k2/2)
    k4 = f(x + h, y + h*k3)
    return y + h*(k1 + 2*k2 + 2*k3 + k4)/6


def forward_euler(f, x, y, h):
    return y + h * f(x, y)

def shoot_fe(lambda_val, x_start, x_end, n_points):
    """
    just a shooting method with Forward Euler integration
    Returns x values and corresponding solution y
    """
    h = (x_end - x_start)/(n_points - 1)
    x = np.linspace(x_start, x_end, n_points)
    
    
    y = np.zeros((2, n_points))
    y[:, 0] = [0, 1]  # tnitial conditions: y(-1)=0, y'(-1)=1
    
    # solvee using Forward Euler
    for i in range(n_points-1):
        y[:, i+1] = forward_euler(
            lambda x, y: legendre_system(x, y, lambda_val),
            x[i],
            y[:, i],
            h
        )
    
    return x, y[0, :]  # here i return x and y values (not y')

def compare_methods(n_eigenvalues=8, n_points=100):
    """
    Compare RK4 and Forward Euler methods
    """
    x_start, x_end = -1, 1
    
    rk4_eigenvalues = []
    fe_eigenvalues = []
    rk4_eigenfunctions = []
    fe_eigenfunctions = []
    
    for n in range(n_eigenvalues):
        lambda_guess = n * (n + 1)
        lambda_range = (max(0, lambda_guess - 0.1), lambda_guess + 0.1)
        
        try:
            rk4_eigenvalue = bisection_eigenvalue(
                lambda_range[0], lambda_range[1],
                x_start, x_end, n_points,
                shoot_func=shoot,
                tol=1e-4  
            )
            x, y_rk4 = shoot(rk4_eigenvalue, x_start, x_end, n_points)
            norm_rk4 = np.sqrt(np.trapezoid(y_rk4**2, x))
            y_rk4 = y_rk4/norm_rk4
            
            fe_eigenvalue = bisection_eigenvalue(
                lambda_range[0], lambda_range[1],
                x_start, x_end, n_points,
                shoot_func=shoot_fe,
                tol=1e-4 
            )
            _, y_fe = shoot_fe(fe_eigenvalue, x_start, x_end, n_points)
            norm_fe = np.sqrt(np.trapezoid(y_fe**2, x))
            y_fe = y_fe/norm_fe
            
            rk4_eigenvalues.append(rk4_eigenvalue)
            fe_eigenvalues.append(fe_eigenvalue)
            rk4_eigenfunctions.append(y_rk4)
            fe_eigenfunctions.append(y_fe)
            
        except Exception as e:
            print(f"Error computing eigenvalue {n}: {str(e)}")
            continue
    
    rk4_eigenvalues = np.array(rk4_eigenvalues)
    fe_eigenvalues = np.array(fe_eigenvalues)
    rk4_eigenfunctions = np.array(rk4_eigenfunctions)
    fe_eigenfunctions = np.array(fe_eigenfunctions)
    
    # simple visuals
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 3, figure=fig)
    
    for i in range(len(rk4_eigenvalues)):
        ax = fig.add_subplot(gs[i//3, i%3])
        ax.plot(x, rk4_eigenfunctions[i], 'b-', label='RK4', alpha=0.7, linewidth=2)
        ax.plot(x, fe_eigenfunctions[i], 'r--', label='Forward Euler', alpha=0.7, linewidth=2)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title(f'Eigenfunction {i+1}\nRK4 λ={rk4_eigenvalues[i]:.2f}\nFE λ={fe_eigenvalues[i]:.2f}')
        ax.set_xlim(-1, 1)
        
        if i == 0:  
            ax.legend()
    
    ax_error = fig.add_subplot(gs[2, 2])
    relative_errors = np.abs(rk4_eigenvalues - fe_eigenvalues) / rk4_eigenvalues * 100
    ax_error.bar(range(len(rk4_eigenvalues)), relative_errors, color='darkred')
    ax_error.set_title('Method Comparison: Relative Error')
    ax_error.set_xlabel('Eigenvalue Index')
    ax_error.set_ylabel('Relative Error (%)')
    ax_error.grid(True, alpha=0.3)
    
    print(f"\nMethod Comparison (RK4 vs Forward Euler) with {n_points} points:")
    print(f"Step size h = {(x_end - x_start)/(n_points-1):.6f}")
    print("-" * 70)
    print(f"{'n':>3} {'RK4 λ':>15} {'FE λ':>15} {'Abs Diff':>15} {'Error %':>10}")
    print("-" * 70)
    
    for i in range(len(rk4_eigenvalues)):
        diff = abs(rk4_eigenvalues[i] - fe_eigenvalues[i])
        error = diff / rk4_eigenvalues[i] * 100
        print(f"{i:3d} {rk4_eigenvalues[i]:15.6f} {fe_eigenvalues[i]:15.6f} "
              f"{diff:15.6f} {error:10.4f}")
    
    plt.suptitle(f"Comparison of RK4 and Forward Euler Methods\nSturm-Liouville Problem (n_points={n_points})", 
                 fontsize=14)
    plt.tight_layout()
    return fig


def legendre_system(x, y, lambda_val):
    """
    this returns derivatives [y', y''] for the Legendre equation
    y[0] = y
    y[1] = y'
    """
    eps = 1e-10  # just to handle singular points
    return np.array([
        y[1],
        (-2*x*y[1] - lambda_val*y[0])/(1 - x**2 + eps)
    ])

def shoot(lambda_val, x_start, x_end, n_points):

    h = (x_end - x_start)/(n_points - 1)
    x = np.linspace(x_start, x_end, n_points)
    
    
    y = np.zeros((2, n_points))
    y[:, 0] = [0, 1]  
    
    for i in range(n_points-1):
        y[:, i+1] = runge_kutta_4(
            lambda x, y: legendre_system(x, y, lambda_val),
            x[i],
            y[:, i],
            h
        )
    
    return x, y[0, :] 

def bisection_eigenvalue(lambda_left, lambda_right, x_start, x_end, n_points, tol=1e-6, max_iter=50, shoot_func=shoot):
    def boundary_value(lambda_val):
        _, y = shoot_func(lambda_val, x_start, x_end, n_points)
        return y[-1]
    
    for _ in range(max_iter):
        lambda_mid = (lambda_left + lambda_right)/2
        
        if abs(boundary_value(lambda_mid)) < tol:
            return lambda_mid
            
        if boundary_value(lambda_left) * boundary_value(lambda_mid) < 0:
            lambda_right = lambda_mid
        else:
            lambda_left = lambda_mid
            
    return (lambda_left + lambda_right)/2

def find_eigenpairs(n_eigenvalues=8, n_points=1000):
    x_start, x_end = -1, 1
    eigenvalues = []
    eigenfunctions = []
    
    for n in range(n_eigenvalues):
        lambda_guess = n*(n+1)
        lambda_range = (lambda_guess-0.5, lambda_guess+0.5)
        
        eigenvalue = bisection_eigenvalue(
            lambda_range[0], lambda_range[1],
            x_start, x_end, n_points
        )
        
        x, y = shoot(eigenvalue, x_start, x_end, n_points)
        
        norm = np.sqrt(np.trapezoid(y**2, x))
        y = y/norm
        
        eigenvalues.append(eigenvalue)
        eigenfunctions.append(y)
    
    return np.array(eigenvalues), np.array(eigenfunctions), x

def plot_eigensolutions(eigenvalues, eigenfunctions, x, save_path=None):
    """
    Create enhanced visualization of eigensolutions
    """
    plt.style.use('default')  
    
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 3, figure=fig)
    colors = plt.cm.viridis(np.linspace(0, 1, len(eigenvalues)))
    
    for i in range(len(eigenvalues)):
        ax = fig.add_subplot(gs[i//3, i%3])
        
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_facecolor('#f0f0f0')  
        
        ax.plot(x, eigenfunctions[i], color=colors[i], linewidth=2)
        ax.fill_between(x, eigenfunctions[i], alpha=0.2, color=colors[i])
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        

        ax.set_title(f'Eigenfunction {i+1}\nλ = {eigenvalues[i]:.2f}', 
                    fontsize=10, pad=10)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1.5, 1.5)
        
        if i >= 5:
            ax.set_xlabel('x')
        
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
            
    ax_summary = fig.add_subplot(gs[2, 2])
    ax_summary.grid(True, linestyle='--', alpha=0.7)
    ax_summary.set_facecolor('#f0f0f0')
    
    for i in range(len(eigenvalues)):
        ax_summary.plot(x, eigenfunctions[i], 
                       label=f'λ₍{i+1}₎={eigenvalues[i]:.1f}',
                       color=colors[i], alpha=0.7)
    
    ax_summary.set_title('All Eigenfunctions', fontsize=10)
    ax_summary.legend(fontsize='x-small', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_summary.set_xlim(-1, 1)
    ax_summary.set_ylim(-1.5, 1.5)
    
    fig.suptitle("Sturm-Liouville Problem: -(1-x²)y'' + 2xy' + λy = 0", 
                 fontsize=14, y=0.95)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def print_eigenvalue_analysis(eigenvalues):
    
    print("\njust eigenvalue analysis:")
    print("-" * 60)
    print(f"{'n':>3} {'Computed λ':>15} {'Theoretical λ':>15} {'Error %':>10}")
    print("-" * 60)
    
    for i, ev in enumerate(eigenvalues):
        theoretical = i * (i + 1)  
        

        if theoretical == 0:
            error = abs(ev - theoretical)
            error_str = f"{error:.6f}"
        else:
            error = abs(ev - theoretical) / theoretical * 100
            error_str = f"{error:.4f}%"
            
        print(f"{i:3d} {ev:15.6f} {theoretical:15.6f} {error_str:>10}")

def find_eigenpairs(n_eigenvalues=8, n_points=1000):
    
    x_start, x_end = -1, 1
    eigenvalues = []
    eigenfunctions = []
    
    for n in range(n_eigenvalues):
        # this is a more precise initial guess range
        lambda_guess = n * (n + 1)
        lambda_range = (max(0, lambda_guess - 0.1), lambda_guess + 0.1)
        
        # Find eigenvalue
        eigenvalue = bisection_eigenvalue(
            lambda_range[0], lambda_range[1],
            x_start, x_end, n_points,
            tol=1e-8 
        )
        x, y = shoot(eigenvalue, x_start, x_end, n_points)
        
        norm = np.sqrt(np.trapezoid(y**2, x))
        y = y/norm
        
        eigenvalues.append(eigenvalue)
        eigenfunctions.append(y)
    
    return np.array(eigenvalues), np.array(eigenfunctions), x

if __name__ == "__main__":
    n_eigenvalues = 8
    n_points = 1000
    eigenvalues, eigenfunctions, x = find_eigenpairs(n_eigenvalues, n_points)
    
    fig = plot_eigensolutions(eigenvalues, eigenfunctions, x)
    
    print_eigenvalue_analysis(eigenvalues)
    
    plt.show()
    
    fig_comparison = compare_methods()
    plt.show()

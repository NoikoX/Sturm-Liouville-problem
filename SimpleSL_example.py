import numpy as np
import matplotlib.pyplot as plt


def runge_kutta_4(f, y0, t, h):
    n = len(t)
    y = np.zeros((n, 2))
    y[0] = y0

    for i in range(n - 1):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(t[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(t[i] + h, y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y


def create_ode_system(lambda_val):
    def system(t, y):
        return np.array([y[1], -lambda_val * y[0]])
    return system


def shooting_method(lambda_val, x, h):
    y0 = np.array([0.0, 1.0])
    system = create_ode_system(lambda_val)
    solution = runge_kutta_4(system, y0, x, h)
    return solution[-1, 1]


def find_eigenvalues(n_eigenvalues, x_points):
    eigenvalues = []
    lambda_min = 0.0
    h = x_points[1] - x_points[0]

    for n in range(n_eigenvalues):
        lambda_max = (n + 1.5) ** 2

        while lambda_max - lambda_min > 1e-6:
            lambda_mid = (lambda_min + lambda_max) / 2
            f_mid = shooting_method(lambda_mid, x_points, h)
            f_min = shooting_method(lambda_min, x_points, h)

            if abs(f_mid) < 1e-6:
                break

            if f_min * f_mid < 0:
                lambda_max = lambda_mid
            else:
                lambda_min = lambda_mid

        eigenvalues.append(lambda_mid)
        lambda_min = lambda_mid + 0.1

    return np.array(eigenvalues)


def compute_eigenfunction(lambda_val, x, h):
    y0 = np.array([0.0, 1.0])
    system = create_ode_system(lambda_val)
    solution = runge_kutta_4(system, y0, x, h)
    return solution[:, 0] / np.max(np.abs(solution[:, 0]))


# setting the problem
x = np.linspace(0, np.pi, 200)
h = x[1] - x[0]

# finding eigenvalues and compute eigenfunctions
n_eigenvalues = 8
eigenvalues = find_eigenvalues(n_eigenvalues, x)
eigenfunctions = [compute_eigenfunction(lambda_val, x, h) for lambda_val in eigenvalues]

print("\nEigenvalues vs Analytical Values:")
print("n\tNumerical\tAnalytical\tError (%)")

for i in range(n_eigenvalues):
    analytical = (0.5 + i) ** 2
    error = abs(eigenvalues[i] - analytical) / analytical * 100
    print(f"{i + 1}\t{eigenvalues[i]:.6f}\t{analytical:.6f}\t{error:.6f}")

# justt plotting eigenfunctions
plt.figure(figsize=(10, 6))
colors = plt.cm.rainbow(np.linspace(0, 1, n_eigenvalues))
for i, (ef, color) in enumerate(zip(eigenfunctions, colors)):
    plt.plot(x, ef, color=color, label=f'n={i + 1}')

plt.title('Eigenfunctions of y″ + λy = 0')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
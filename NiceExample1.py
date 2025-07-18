import numpy as np
import matplotlib.pyplot as plt


def rk4_step(f, t, y, h):
    # f is jist a function that returns dy/dt
    # i presented y as a vector (y, y')
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h * k1 / 2)
    k3 = f(t + h / 2, y + h * k2 / 2)
    k4 = f(t + h, y + h * k3)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def solve_ivp_rk4(f, t_span, y0, h):
    #y_0 is an initial conditino (y(0), y'(0))
    t = np.arange(t_span[0], t_span[1] + h, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(len(t) - 1):
        y[i + 1] = rk4_step(f, t[i], y[i], h)

    return t, y


def system_equation(lambda_val):
    def f(t, y):
        # y[0] = y, y[1] = y'
        return np.array([y[1], -3 * y[1] - 2 * y[0] - lambda_val * y[0]])

    return f


def shooting_method(lambda_val, h=0.01):
    t_span = [0, 1]
    y0 = np.array([0, 1])  # y(0)=0, y'(0)=1 (just a normalized initial slope)

    t, y = solve_ivp_rk4(system_equation(lambda_val), t_span, y0, h)
    return y[-1, 0]


def find_eigenvalue(lambda_guess, tolerance=1e-6, max_iter=100):

    # lambdass using bisection meth
    lambda_left = lambda_guess - 1
    lambda_right = lambda_guess + 1

    for _ in range(max_iter):
        lambda_mid = (lambda_left + lambda_right) / 2
        y_end = shooting_method(lambda_mid)

        if abs(y_end) < tolerance:
            return lambda_mid
        elif y_end * shooting_method(lambda_left) < 0:
            lambda_right = lambda_mid
        else:
            lambda_left = lambda_mid

    return lambda_mid


def compute_eigenfunction(lambda_val, h=0.01):

    t_span = [0, 1]
    y0 = np.array([0, 1])

    t, y = solve_ivp_rk4(system_equation(lambda_val), t_span, y0, h)

    # just normalize the the eigenfunction
    norm = np.sqrt(np.trapezoid(y[:, 0] ** 2, t))
    y[:, 0] /= norm

    return t, y[:, 0]


def find_first_n_eigenvalues(n, initial_guesses=None):


    if initial_guesses is None:
        # Based on the analytical solution pattern: λₙ = (1 + 4n²π²)/4
        initial_guesses = [(1 + 4 * (k ** 2) * (np.pi ** 2)) / 4 for k in range(1, n + 1)]

    eigenvalues = []
    eigenfunctions = []

    for guess in initial_guesses:
        eigenval = find_eigenvalue(guess)
        t, eigenfunc = compute_eigenfunction(eigenval)

        eigenvalues.append(eigenval)
        eigenfunctions.append(eigenfunc)

    return np.array(eigenvalues), t, np.array(eigenfunctions)


n_eigenvalues = 8
eigenvalues, t, eigenfunctions = find_first_n_eigenvalues(n_eigenvalues)

plt.figure(figsize=(12, 8))

colors = plt.cm.viridis(np.linspace(0, 1, n_eigenvalues))

for i in range(n_eigenvalues):
    plt.plot(t, eigenfunctions[i], color=colors[i],
             label=f'λ₍{i + 1}₎ = {eigenvalues[i]:.2f}',
             linewidth=2)

plt.title('Eigenfunctions of the Sturm-Liouville Problem\n' +
          r'$y^{\prime\prime} + 3y^\prime + 2y + \lambda y = 0$, ' +
          r'$y(0) = y(1) = 0$',
          fontsize=14, pad=20)
plt.xlabel('x', fontsize=12)
plt.ylabel('y(x)', fontsize=12)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
           borderaxespad=0., frameon=True)

plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()

print("\nEigenvalues:")
for i, ev in enumerate(eigenvalues, 1):
    print(f"λ_{i} = {ev:.6f}")

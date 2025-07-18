import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def sturm_liouville_solver(p, q, r, a, b, a1, a2, b1, b2, x_span, eigenvalue_guesses):
    # just solving the SL problem using shooting method
    # so the arguments are:
    # p, r, q functions
    # a, b boundary points
    # a_n, b_n boundary condition coefficients
    # x_span just an array of x values for the solutionn
    # and an initiail guess

    eigenvalues = []
    eigenfunctions = []

    def ode_system(x, y_lambda, lam):
        # our systemm
        y, dy = y_lambda
        if p(x) == 0:
          ddy = 0
        else:
          ddy = (-q(x) * y - lam * r(x) * y) / p(x)  # Dividing by p(x) here

        return [dy, ddy]

    def boundary_residual(lam):
        """
        Calculates the residual of the right boundary condition for a given eigenvalue guess.
        """
        # handling the cases when i divide by 0
        if a2 == 0 or p(a) == 0:
            y0 = [0, 1]  # settingg y(a) = 0, y'(a) = 1
        else:
            y0 = [1, -a1 / (a2 * p(a))]

        sol = solve_ivp(
            lambda x, y_lambda: ode_system(x, y_lambda, lam),
            [a, b],
            y0,
            t_eval=x_span,
            method='RK45',
        )

        y_b = sol.y[0, -1]
        dy_b = sol.y[1, -1]

        if b2 == 0 or p(b) == 0:
          return y_b
        else:
          return b1 * y_b + b2 * p(b) * dy_b

    #rootfinding loop (using bisection as an example)
    for i, lam_guess in enumerate(eigenvalue_guesses):
        tolerance = 1e-6
        max_iterations = 100

        # bracket the eigenvalue
        lam_lower = lam_guess - 0.5
        lam_upper = lam_guess + 0.5

        while boundary_residual(lam_lower) * boundary_residual(lam_upper) > 0:
            lam_lower -= 0.5
            lam_upper += 0.5

        # here is the loop for bisection method
        for _ in range(max_iterations):
            lam_mid = (lam_lower + lam_upper) / 2
            residual_mid = boundary_residual(lam_mid)

            if abs(residual_mid) < tolerance:
                break

            if boundary_residual(lam_lower) * residual_mid < 0:
                lam_upper = lam_mid
            else:
                lam_lower = lam_mid
        else:
            print("upss: max iterations reached for eigenvalue", i)

        # solve with the found eigenvalue: NOW WE CHECK FOR THE CONDITION AGAIN
        if a2 == 0 or p(a) == 0:
            y0 = [0, 1]
        else:
            y0 = [1, -a1 / (a2 * p(a))]

        sol = solve_ivp(
            lambda x, y_lambda: ode_system(x, y_lambda, lam_mid),
            [a, b],
            y0,
            t_eval=x_span,
            method='RK45',
        )

        eigenvalues.append(lam_mid)
        eigenfunctions.append(sol.y[0, :])

    return eigenvalues, eigenfunctions



def p(x):
    return 1 + 1e-6

def q(x):
    return 0

def r(x):
    return 1

a = 0
b = 1
a1 = 1
a2 = 0
b1 = 1
b2 = 0
x_span = np.linspace(a, b, 200)
eigenvalue_guesses = [10, 40, 90, 150, 240, 340, 460, 600]  # just some initial guesses

eigenvalues, eigenfunctions = sturm_liouville_solver(p, q, r, a, b, a1, a2, b1, b2, x_span, eigenvalue_guesses)


for i in range(len(eigenvalues)):
    plt.plot(x_span, eigenfunctions[i], label=f"λ = {eigenvalues[i]:.2f}")

plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("Eigenfunctions")
plt.legend()
plt.grid(True)
plt.show()

print("Eigenvalues:", eigenvalues)
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
b = 0.2  # Damping coefficient
beta = 1.0  # Linear stiffness
omega = 1.0  # Forcing frequency
T = 2 * np.pi / omega  # Forcing period
t_transient = 500  # Time to discard as transient
t_total = 1000  # Total simulation time
dt = 0.01  # Time step for integration
F_values = np.linspace(0.1, 0.7, 200)  # Range of forcing amplitudes


# Duffing equation (forced)
def duffing_forced(t, y, F):
    x, v = y
    dxdt = v
    dvdt = -b * v + beta * x - x ** 3 + F * np.cos(omega * t)
    return [dxdt, dvdt]


# Initial conditions
x0, v0 = 0.1, 0.0  # Small displacement and zero velocity

# Prepare bifurcation data
bifurcation_x = []
bifurcation_F = []

for F in F_values:
    # Integrate the system
    solution = solve_ivp(
        duffing_forced, [0, t_total], [x0, v0],
        t_eval=np.arange(0, t_total, dt), args=(F,), method='RK45'
    )
    t_vals = solution.t
    x_vals = solution.y[0]

    # Sample the long-term behavior (after transient)
    sampled_x = []
    for i, t in enumerate(t_vals):
        if t > t_transient and np.isclose(t % T, 0, atol=dt):  # After transient, multiple of T
            sampled_x.append(x_vals[i])

    # Append data for bifurcation diagram
    bifurcation_x.extend(sampled_x)
    bifurcation_F.extend([F] * len(sampled_x))

# Plot the bifurcation diagram
plt.figure(figsize=(10, 8))
plt.scatter(bifurcation_F, bifurcation_x, s=0.5, color='blue', alpha=0.8)
plt.title("Bifurcation Diagram of the Duffing Equation")
plt.xlabel("Forcing Amplitude (F)")
plt.ylabel("Displacement (x)")
plt.grid()
plt.savefig("bifurcation_diagram.png")
plt.show()

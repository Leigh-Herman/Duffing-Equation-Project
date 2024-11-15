import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
b = 0.2  # Damping coefficient
beta_values = [-0.5, 0.0, 0.5]  # Values of beta to simulate
dt = 0.01  # Time step for integration
t_max = 50  # Total simulation time


# Duffing equation (unforced)
def duffing_unforced(t, y, beta):
    x, v = y
    dxdt = v
    dvdt = -b * v + beta * x - x ** 3
    return [dxdt, dvdt]


# Initial conditions
x0, v0 = 1.0, 0.0  # Starting from displacement 1.0 and zero velocity

# Time array for dense integration
t_eval = np.arange(0, t_max, dt)

# Plot phase portraits for each beta
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for i, beta in enumerate(beta_values):
    # Solve the system
    solution = solve_ivp(duffing_unforced, [0, t_max], [x0, v0], t_eval=t_eval, args=(beta,), method='RK45')
    x_vals = solution.y[0]
    v_vals = solution.y[1]

    # Plot phase portrait
    axes[i].plot(x_vals, v_vals, color='blue', alpha=0.7)
    axes[i].set_title(f"Phase Portrait for β = {beta}")
    axes[i].set_xlabel("Displacement (x)")
    if i == 0:
        axes[i].set_ylabel("Velocity ($\dot{x}$)")
    axes[i].grid()

# Save the figure
plt.tight_layout()
plt.savefig("unforced_duffing_phase_portraits.png")
plt.show()

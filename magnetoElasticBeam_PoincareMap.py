import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
b = 0.2  # Damping coefficient
beta = 1.0  # Linear stiffness
F = 0.3  # Forcing amplitude
omega = 1.0  # Forcing frequency
t_max = 500  # Total simulation time
T = 2 * np.pi / omega  # Forcing period
dt = 0.01  # Time step for integration

# Duffing equation
def duffing(t, y):
    x, v = y
    dxdt = v
    dvdt = -b * v + beta * x - x**3 + F * np.cos(omega * t)
    return [dxdt, dvdt]

# Initial conditions
x0, v0 = 0.1, 0.0  # Small displacement and zero velocity
y0 = [x0, v0]

# Time array for dense integration
t_eval = np.arange(0, t_max, dt)

# Integrate the system
solution = solve_ivp(duffing, [0, t_max], y0, t_eval=t_eval, method='RK45')

# Extract solutions
t_vals = solution.t
x_vals = solution.y[0]
v_vals = solution.y[1]

# Construct the Poincaré map
poincare_x = []
poincare_v = []

for i, t in enumerate(t_vals):
    if np.isclose(t % T, 0, atol=dt):  # Check if t is a multiple of T
        poincare_x.append(x_vals[i])
        poincare_v.append(v_vals[i])

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Phase Portrait
axes[0].plot(x_vals, v_vals, color='blue', alpha=0.7, linewidth=0.8)
axes[0].set_title("Phase Portrait of the Duffing Equation")
axes[0].set_xlabel("Displacement (x)")
axes[0].set_ylabel("Velocity ($\dot{x}$)")
axes[0].grid()

# Poincaré Map
axes[1].scatter(poincare_x, poincare_v, s=5, color='red', alpha=0.7)
axes[1].set_title("Poincaré Map of the Duffing Equation")
axes[1].set_xlabel("Displacement (x)")
axes[1].set_ylabel("Velocity ($\dot{x}$)")
axes[1].grid()

# Save both plots
plt.tight_layout()
plt.savefig("duffing_phase_and_poincare.png")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
b = 0.2  # Damping coefficient
beta = 1.0  # Linear stiffness
F = 0.5  # Forcing amplitude (chaotic regime)
omega = 1.0  # Forcing frequency
T = 2 * np.pi / omega  # Forcing period
t_max = 5000  # Long simulation time to capture fine details
dt = 0.01  # Time step for integration

# Duffing equation (forced)
def duffing_forced(t, y):
    x, v = y
    dxdt = v
    dvdt = -b * v + beta * x - x**3 + F * np.cos(omega * t)
    return [dxdt, dvdt]

# Initial conditions
x0, v0 = 0.1, 0.0  # Initial displacement and velocity

# Time array for dense integration
t_eval = np.arange(0, t_max, dt)

# Integrate the system
solution = solve_ivp(duffing_forced, [0, t_max], [x0, v0], t_eval=t_eval, method='RK45')
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

# Plot the Poincaré map
plt.figure(figsize=(8, 8))
plt.scatter(poincare_x, poincare_v, s=1, color='red', alpha=0.7)
plt.title("High-Resolution Poincaré Map: Strange Attractor")
plt.xlabel("Displacement (x)")
plt.ylabel("Velocity ($\dot{x}$)")
plt.grid()

# Save the figure
plt.savefig("strange_attractor_poincare_map.png")
plt.show()

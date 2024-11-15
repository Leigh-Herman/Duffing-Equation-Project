import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
b = 0.2  # Damping coefficient
beta = 1.0  # Linear stiffness
F = 0.5  # Forcing amplitude (chaotic regime)
omega = 1.0  # Forcing frequency
dt = 0.01  # Time step for integration
t_max = 50  # Time for forward and backward integration

# Duffing equation (forced)
def duffing_forced(t, y, direction=1):
    x, v = y
    dxdt = v
    dvdt = -b * v + beta * x - x**3 + F * np.cos(omega * t)
    return direction * np.array([dxdt, dvdt])

# Initial conditions near the saddle point
saddle_point = [0.0, 0.0]  # Saddle point for the Duffing equation
perturbation = 0.01  # Small perturbation to escape the saddle

# Stable manifold: Integrate backward in time
stable_initial_conditions = [
    [saddle_point[0] + perturbation, saddle_point[1]],  # Perturbed in x
    [saddle_point[0], saddle_point[1] + perturbation],  # Perturbed in v
]
stable_trajectories = []
for y0 in stable_initial_conditions:
    sol = solve_ivp(duffing_forced, [0, -t_max], y0, t_eval=np.arange(0, -t_max, -dt), args=(-1,), method='RK45')
    stable_trajectories.append(sol)

# Unstable manifold: Integrate forward in time
unstable_initial_conditions = [
    [saddle_point[0] + perturbation, saddle_point[1]],  # Perturbed in x
    [saddle_point[0], saddle_point[1] + perturbation],  # Perturbed in v
]
unstable_trajectories = []
for y0 in unstable_initial_conditions:
    sol = solve_ivp(duffing_forced, [0, t_max], y0, t_eval=np.arange(0, t_max, dt), args=(1,), method='RK45')
    unstable_trajectories.append(sol)

# Plot stable and unstable manifolds
plt.figure(figsize=(8, 8))
for sol in stable_trajectories:
    plt.plot(sol.y[0], sol.y[1], color='blue', alpha=0.7, label="Stable Manifold" if "Stable" not in plt.gca().get_legend_handles_labels()[1] else "")
for sol in unstable_trajectories:
    plt.plot(sol.y[0], sol.y[1], color='red', alpha=0.7, label="Unstable Manifold" if "Unstable" not in plt.gca().get_legend_handles_labels()[1] else "")

plt.scatter(saddle_point[0], saddle_point[1], color='black', label="Saddle Point")
plt.title("Invariant Manifolds of the Duffing Equation")
plt.xlabel("Displacement (x)")
plt.ylabel("Velocity ($\dot{x}$)")
plt.legend()
plt.grid()

# Save the figure
plt.savefig("invariant_manifolds.png")
plt.show()

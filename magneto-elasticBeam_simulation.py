import numpy as np
import matplotlib.pyplot as plt

# Parameters
m = 1.0  # Mass
beta = 1.0  # Parameter in the potential
dt = 0.01  # Time step
t_max = 50  # Simulation time

# Hamiltonian equations of motion
def hamiltonian_eqs(y):
    q, p = y
    dq_dt = p / m
    dp_dt = beta**2 * q - q**3
    return np.array([dq_dt, dp_dt])

# Runge-Kutta 4 method
def runge_kutta_4_step(f, y, dt):
    k1 = f(y)
    k2 = f(y + dt / 2 * k1)
    k3 = f(y + dt / 2 * k2)
    k4 = f(y + dt * k3)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# Time array
t_vals = np.arange(0, t_max, dt)

# Create a grid of initial conditions
q_vals_ic = np.linspace(-2, 2, 10)  # Initial displacement values
p_vals_ic = np.linspace(-2, 2, 10)  # Initial momentum values

initial_conditions = [(q, p) for q in q_vals_ic for p in p_vals_ic]

# Solve the system for each initial condition
trajectories = []
for y0 in initial_conditions:
    y = np.array(y0)
    trajectory = []
    for t in t_vals:
        trajectory.append(y)
        y = runge_kutta_4_step(hamiltonian_eqs, y, dt)
    trajectories.append(np.array(trajectory))

# Plot the phase portrait
plt.figure(figsize=(10, 10))
for trajectory in trajectories:
    q_vals, p_vals = trajectory[:, 0], trajectory[:, 1]
    plt.plot(q_vals, p_vals, alpha=0.6, linewidth=0.8)

plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.title("Phase Portrait of Magneto-Elastic Beam Dynamics")
plt.xlabel("Displacement (q)")
plt.ylabel("Momentum (p)")
plt.grid()
plt.savefig("magneto-elastic_Beam_phase_Portrait.png")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
b = 0.2  # Damping coefficient
beta = 1.0  # Linear stiffness
omega = 1.0  # Forcing frequency
F_values = [0.1, 0.3, 0.5]  # Forcing amplitudes to simulate
T = 2 * np.pi / omega  # Forcing period
t_max = 500  # Total simulation time
dt = 0.01  # Time step for integration


# Duffing equation (forced)
def duffing_forced(t, y, F):
    x, v = y
    dxdt = v
    dvdt = -b * v + beta * x - x ** 3 + F * np.cos(omega * t)
    return [dxdt, dvdt]


# Initial conditions
x0, v0 = 0.1, 0.0  # Small displacement and zero velocity

# Time array for dense integration
t_eval = np.arange(0, t_max, dt)

# Simulate for each forcing amplitude
for F in F_values:
    # Integrate the system
    solution = solve_ivp(duffing_forced, [0, t_max], [x0, v0], t_eval=t_eval, args=(F,), method='RK45')
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

    # Plot phase portrait and Poincaré map
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Time series
    axes[0].plot(t_vals, x_vals, color='blue', alpha=0.7, linewidth=0.8)
    axes[0].set_title(f"Time Series (F = {F})")
    axes[0].set_xlabel("Time (t)")
    axes[0].set_ylabel("Displacement (x)")
    axes[0].grid()

    # Phase portrait
    axes[1].plot(x_vals, v_vals, color='blue', alpha=0.7, linewidth=0.8)
    axes[1].set_title(f"Phase Portrait (F = {F})")
    axes[1].set_xlabel("Displacement (x)")
    axes[1].set_ylabel("Velocity ($\dot{x}$)")
    axes[1].grid()

    # Poincaré map
    axes[2].scatter(poincare_x, poincare_v, s=5, color='red', alpha=0.7)
    axes[2].set_title(f"Poincaré Map (F = {F})")
    axes[2].set_xlabel("Displacement (x)")
    axes[2].set_ylabel("Velocity ($\dot{x}$)")
    axes[2].grid()

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"forcing_and_chaos_F_{F}.png")
    plt.show()

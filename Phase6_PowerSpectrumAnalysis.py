import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq

# Parameters
b = 0.2  # Damping coefficient
beta = 1.0  # Linear stiffness
F = 0.5  # Forcing amplitude
omega = 1.0  # Forcing frequency
t_max = 500  # Total simulation time
dt = 0.01  # Time step for integration

# Duffing equation (forced)
def duffing_forced(t, y):
    x, v = y
    dxdt = v
    dvdt = -b * v + beta * x - x**3 + F * np.cos(omega * t)
    return [dxdt, dvdt]

# Initial conditions
x0, v0 = 0.1, 0.0  # Small displacement and zero velocity

# Time array for dense integration
t_eval = np.arange(0, t_max, dt)

# Integrate the system
solution = solve_ivp(duffing_forced, [0, t_max], [x0, v0], t_eval=t_eval, method='RK45')
t_vals = solution.t
x_vals = solution.y[0]

# Compute the power spectrum
N = len(t_vals)  # Number of samples
xf = fft(x_vals)  # Compute FFT
frequencies = fftfreq(N, dt)  # Frequency bins
power_spectrum = np.abs(xf)**2  # Power spectrum

# Filter for positive frequencies
positive_frequencies = frequencies[:N // 2]
positive_power = power_spectrum[:N // 2]

# Plot the power spectrum
plt.figure(figsize=(10, 6))
plt.plot(positive_frequencies, positive_power, color='blue')
plt.title(f"Power Spectrum for Duffing Equation (F = {F})")
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.grid()
plt.xlim(0, 5)  # Focus on low frequencies for clarity

# Save the figure
plt.savefig("power_spectrum.png")
plt.show()

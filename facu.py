import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Simulation parameters
fs = 5e9  # Sampling frequency 5 GHz
f0 = 100e6  # Central frequency 100 MHz
c = 8433  # Speed of sound in silicon (m/s)

# Reflector depths (meters)
depths = np.array([150e-6, 0.5e-3, 1.0e-3])  # 150 μm groove, 500 μm wafer interface, 1 mm back surface
coeffs = np.array([0.5, 0.8, 0.6])  # Reflection coefficients

# Time vector: enough to capture the round-trip to the deepest interface
t_max = 2 * depths.max() / c + 50e-9  # add margin
t = np.arange(0, t_max, 1/fs)

# Generate Gaussian-modulated sine pulse
pulse_duration = 50e-9  # 50 ns pulse duration
tau = pulse_duration / 6
t0 = 6 * tau  # center of Gaussian
pulse = np.exp(-((t - t0)**2) / (2 * tau**2)) * np.sin(2 * np.pi * f0 * (t - t0))

# Initialize echo signal
signal = np.zeros_like(t)

# Superpose echoes with proper slicing
for depth, coeff in zip(depths, coeffs):
    delay = 2 * depth / c  # round-trip delay
    delay_samples = int(np.round(delay * fs))
    end_idx = min(delay_samples + len(pulse), len(signal))
    pulse_len = end_idx - delay_samples
    signal[delay_samples:end_idx] += coeff * pulse[:pulse_len]

# Add noise to emulate low SNR
noise = 0.2 * np.random.randn(len(t))
signal_noisy = signal + noise

# Plot results
plt.figure(figsize=(10, 4))
plt.plot(t * 1e9, signal_noisy, label="Simulated Echo Signal")
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude")
plt.title("Simulated Ultrasonic Echoes (100 MHz, fs=5 GHz)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Create a table of echo times
echo_times = 2 * depths / c * 1e9  # in ns
df = pd.DataFrame({
    "Depth (μm)": depths * 1e6,
    "Echo Time (ns)": echo_times,
    "Reflection Coef": coeffs
})

import ace_tools as tools; tools.display_dataframe_to_user(name="Echo Interfaces Summary", dataframe=df)

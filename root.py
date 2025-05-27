import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert  # para la envolvente

# Parámetros básicos
fs = 5e9         # 5 GHz (como en el experimento real)
duration = 2e-6  # 2 μs
t = np.linspace(0, duration, int(fs * duration))

# Parámetros del eco (interfaces superior e inferior, o ranuras)
echo_params_base = [
    {'A': 10, 'alpha': 300e12, 'f': 100e6, 'tau': 0.4e-6},
    {'A': 8,  'alpha': 300e12, 'f': 100e6, 'tau': 0.6e-6}
]

# Definimos el tamaño del escaneo
Nx, Ny = 100, 100
pixel_size = 0.1e-3  # 0.1 mm por píxel

# Creamos un “mapa de profundidad” de la oblea
# Por ejemplo, ranuras de 150 μm de profundidad en ciertas columnas
depth_map = np.zeros((Nx, Ny))
for i in range(Nx):
    for j in range(Ny):
        # Si estamos en una ranura (digamos j entre 30 y 50)
        if i % 4 == 0:
            depth_map[i, j] = 0.15e-3  # 150 μm
        else:
            depth_map[i, j] = 0.5e-3   # resto de la oblea

# Función de generación de eco
def generate_echo(t, A, alpha, f, tau):
    return A * np.exp(-alpha*(t - tau)**2) * np.cos(2*np.pi*f*(t - tau))

# Add noise with desired SNR
def add_noise(signal, snr_db):
    signal_power = np.mean(signal**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise

# Pre­-reserve la matriz del C-scan
cscan = np.zeros((Nx, Ny))

for i in range(Nx):
    for j in range(Ny):
        # Ajustamos tiempo de llegada según la profundidad local
        local_tau = depth_map[i, j] / 1500  # v ~1500 m/s
        params = [
            {**echo_params_base[0], 'tau': local_tau},
            {**echo_params_base[1], 'tau': local_tau + 0.2e-6}
        ]
        # Generamos la señal limpia
        clean = sum(generate_echo(t, **p) for p in params)
        # Añadimos ruido (ej. SNR 10 dB)
        noisy = add_noise(clean, snr_db=120) # Por ahora, SNR muy alto

        # TODO: Implementar el procesamiento de la señal (sacarle el ruido)

        # 1) Extraemos la envolvente con Hilbert
        env = np.abs(hilbert(noisy))
        # 2) Tomamos el valor pico de la envolvente
        cscan[i, j] = np.max(env)

# Para visualizar en escala real
x = np.arange(Nx) * pixel_size * 1e3  # en mm
y = np.arange(Ny) * pixel_size * 1e3  # en mm

plt.figure(figsize=(6,5))
plt.imshow(cscan.T, 
           extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower',
           cmap='gray')
plt.xlabel('Posición X (mm)')
plt.ylabel('Posición Y (mm)')
plt.title('C-scan (amplitud pico de la envolvente)')
plt.colorbar(label='Amplitud')
plt.tight_layout()
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Parámetros de simulación
VELOCIDAD_SONIDO = 5000  # m/s (ejemplo para acero)
RESOLUCION_ESPACIAL = 1e-4  # metros por píxel (ajustable)
AMPLITUD_PICO = 1.0  # valor del eco
UMBRAL_DETECCION = 100  # para detección de bordes

def cargar_y_procesar_imagen(path_imagen):
    # Cargar imagen en escala de grises
    img = cv2.imread(path_imagen, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("No se pudo cargar la imagen.")
    # Detección de bordes
    bordes = cv2.Canny(img, threshold1=UMBRAL_DETECCION, threshold2=2 * UMBRAL_DETECCION)
    return bordes, img.shape

def extraer_puntos_reflectores(bordes):
    # Encuentra coordenadas de los bordes (reflectores)
    puntos = np.column_stack(np.where(bordes > 0))
    # Ordenar por profundidad (eje Y)
    puntos = puntos[np.argsort(puntos[:, 0])]
    return puntos

def simular_senal_echo(puntos, altura_imagen):
    tiempos = []
    amplitudes = []

    for y, x in puntos:
        profundidad_m = y * RESOLUCION_ESPACIAL
        tiempo = 2 * profundidad_m / VELOCIDAD_SONIDO  # ida y vuelta
        tiempos.append(tiempo)
        amplitudes.append(AMPLITUD_PICO)  # todos los ecos con misma amplitud

    return np.array(tiempos), np.array(amplitudes)

def graficar_senal(tiempos, amplitudes):
    plt.figure(figsize=(10, 4))
    plt.stem(tiempos * 1e6, amplitudes)
    plt.xlabel('Tiempo (μs)')
    plt.ylabel('Amplitud')
    plt.title('Señal Ultrasónica Simulada (A-scan)')
    plt.grid(True)
    plt.show()

def main(path_imagen):
    print('a')
    bordes, tamano = cargar_y_procesar_imagen(path_imagen)
    print('b')
    puntos = extraer_puntos_reflectores(bordes)
    print('c')
    tiempos, amplitudes = simular_senal_echo(puntos, tamano[0])
    print('d')
    graficar_senal(tiempos, amplitudes)

if __name__ == "__main__":
    # Ruta de la imagen a procesar
    path_imagen = '/home/iann/Desktop/Risk/How-To-Rid-Yourself-Of-Black-and-White-Thinking.jpg'  # Cambia esto por la ruta de tu imagen
    main(path_imagen)
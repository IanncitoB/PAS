import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('./data/consumo.csv', encoding="utf-8", thousands=".", decimal=",")
data['Fecha'] = pd.to_datetime(data['Fecha'], format='%d/%m/%Y')
data.set_index('Fecha', inplace=True)

# eliminar columnas no numéricas
data = data.select_dtypes(include=['float64', 'int64'])
# eliminar columna 'año' y 'temperatura'
data = data.drop(columns=['AÑO', 'TEMPERATURA REFERENCIA MEDIA GBA °C', 'DEMANDA TOTAL'], errors='ignore')

df = pd.DataFrame(data)

print(df.head())

# Calcular la matriz de correlación
corr_matrix = df.corr()

# Configurar el tamaño de la figura
plt.figure(figsize=(12, 10))

# Crear el heatmap
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    cbar_kws={"label": "Correlación"}
)

# Título opcional
plt.title("Mapa de Calor de Correlaciones", fontsize=16)
plt.tight_layout()
plt.show()

# LIMPIEZA DE DATOS # 
# DANIEL SEPÚLVEDA 202022738 #
# INGENIERÍA DE DATOS #

# Importar librerías
import pandas as pd
import numpy as np
import missingno as mn
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo CSV
data = pd.read_csv('C:/Users/danie/Documents/PROYECTO 2/bank-full.csv', sep=';')

# Inspección inicial de los datos
print("Tamaño inicial del dataset:", data.shape)
data.info()  # Información general de las columnas y tipos de datos
print(data.head())  
print(data.shape)

data.info() #ver variables numericas y categoricas

# Visualización inicial de valores nulos
mn.matrix(data)  # Gráfica para visualizar valores nulos
plt.show()

# Verificar valores nulos (No se encuentran y por ende no se eliminan)
print("Valores nulos por columna:")
print(data.isnull().sum())

# Verificación de subniveles en variables categóricas
# Verificar nombres de columnas en el DataFrame para ajustar la lista de categorías
print("Nombres de columnas en el DataFrame:", data.columns)

#crear lista de variables categoricas
categorical_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

# Verificación de subniveles (se encuentra que todos corresponden a valores acordes y por ende no hay tratamiento)
for column in categorical_columns:
    unique_values = data[column].nunique()
    print(f"Columna '{column}' tiene {unique_values} subniveles únicos")

# Crear un gráfico de barras para cada variable categórica (para verificar subniveles extras)
for col in categorical_columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x=col, palette="viridis")
    plt.title(f'Distribución de la variable categórica: {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)  # Rotar etiquetas en caso de que haya muchas categorías
    plt.show()

# Verificación de variabilidad en columnas numéricas
print("Estadísticas descriptivas para variables numéricas:")
print(data.describe())

# Confirmación de variabilidad (desviación estándar distinta de cero)
for column in data.select_dtypes(include=[np.number]).columns:
    std_dev = data[column].std()
    print(f"Columna '{column}' tiene una desviación estándar de: {std_dev}")
    if std_dev == 0:
        print(f"Advertencia: La columna '{column}' tiene desviación estándar 0 y no aporta información.")


# Detección y eliminación de duplicados (no se encuentran)
initial_shape = data.shape
data.drop_duplicates(inplace=True)
print("Tamaño después de eliminar duplicados:", data.shape)
print(f"Se eliminaron {initial_shape[0] - data.shape[0]} filas duplicadas.")

# revisión de outliers
# Obtener las columnas numéricas
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Crear un boxplot para cada columna numérica
for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot de la variable {col}')
    plt.xlabel(col)
    plt.show()

#eliminar outlier en campaing  y dato extra en previous
lower_limit = data["campaign"].quantile(0.01)
upper_limit = data["campaign"].quantile(0.99)
data = data[(data["campaign"] >= lower_limit) & (data["campaign"] <= upper_limit)]
data = data[data['previous'] <= 100]


# Convertir variables categóricas a variables dummy
data = pd.get_dummies(data, drop_first=True)

# Verificación final de los datos
print("Información final del dataset limpio:")
print(data.info())
print(data.head())

# Exportar el DataFrame limpio a un archivo CSV
data.to_csv('datos_limpios.csv', index=False)

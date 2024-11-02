import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar el estilo visual
sns.set(style="whitegrid")

# Cargar el archivo CSV
data = pd.read_csv('datos_limpios.csv')

# Filtrar los datos de clientes que se suscribieron a depósitos a término (`y_yes` indica suscripción con True/False)
subscribed_data = data[data['y_yes'] == True]

# 1. Análisis Demográfico y Financiero

# Distribución de Edad de Clientes Suscritos a Depósitos a Término
plt.figure(figsize=(10, 6))
sns.histplot(subscribed_data['age'], bins=20, kde=True, color="skyblue")
plt.title("Distribución de Edad de Clientes Suscritos a Depósitos a Término")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.show()

# Diagrama de dispersión de Balance vs Edad para Clientes Suscritos
plt.figure(figsize=(10, 6))
sns.scatterplot(data=subscribed_data, x='age', y='balance', hue='y_yes', palette='viridis')
plt.title("Balance vs Edad de Clientes Suscritos")
plt.xlabel("Edad")
plt.ylabel("Balance")
plt.show()

# Tasa de Suscripción por Tipo de Trabajo
job_columns = [col for col in data.columns if col.startswith('job_')]
job_data = subscribed_data[job_columns].mean() * 100  # Porcentaje de suscripciones por tipo de trabajo

plt.figure(figsize=(12, 6))
job_data.sort_values().plot(kind='barh', color="lightcoral")
plt.title("Tasa de Suscripción por Tipo de Trabajo")
plt.xlabel("Tasa de Suscripción (%)")
plt.ylabel("Tipo de Trabajo")
plt.show()

# Distribución de Balance para Clientes Suscritos y No Suscritos
plt.figure(figsize=(10, 6))
sns.histplot(data, x='balance', hue='y_yes', kde=True, palette="pastel")
plt.title("Distribución de Balance para Clientes Suscritos y No Suscritos")
plt.xlabel("Balance")
plt.ylabel("Frecuencia")
plt.show()

# Estado Civil y Tasa de Suscripción
marital_columns = [col for col in data.columns if col.startswith('marital_')]
marital_data = subscribed_data[marital_columns].mean() * 100  # Porcentaje de suscripciones por estado civil

plt.figure(figsize=(10, 6))
marital_data.sort_values().plot(kind='bar', color="slateblue")
plt.title("Tasa de Suscripción por Estado Civil")
plt.xlabel("Estado Civil")
plt.ylabel("Tasa de Suscripción (%)")
plt.show()

# 2. Estrategia de Contacto

# Tasa de Suscripción vs Número de Contactos
plt.figure(figsize=(10, 6))
sns.lineplot(x='campaign', y='y_yes', data=data, estimator='mean', color="purple")
plt.title("Tasa de Suscripción vs Número de Contactos")
plt.xlabel("Número de Contactos")
plt.ylabel("Tasa de Suscripción")
plt.show()

# Suscripciones Exitosas por Mes de Contacto
month_columns = [col for col in data.columns if col.startswith('month_')]
monthly_subscriptions = data[data['y_yes'] == True][month_columns].sum()

plt.figure(figsize=(12, 6))
monthly_subscriptions.plot(kind='bar', color="teal")
plt.title("Suscripciones Exitosas por Mes de Contacto")
plt.xlabel("Mes")
plt.ylabel("Número de Suscripciones Exitosas")
plt.show()

# Distribución de Duración de Contacto para Clientes Suscritos y No Suscritos
plt.figure(figsize=(10, 6))
sns.histplot(data, x='duration', hue='y_yes', kde=True, palette="crest", bins=30)
plt.title("Distribución de Duración de Contacto para Clientes Suscritos y No Suscritos")
plt.xlabel("Duración del Contacto (segundos)")
plt.ylabel("Frecuencia")
plt.show()

# Número de Contactos Previos y Tasa de Suscripción
plt.figure(figsize=(10, 6))
sns.barplot(x='previous', y='y_yes', data=data, estimator='mean', color="coral")
plt.title("Tasa de Suscripción vs Número de Contactos Previos")
plt.xlabel("Número de Contactos Previos")
plt.ylabel("Tasa de Suscripción")
plt.show()

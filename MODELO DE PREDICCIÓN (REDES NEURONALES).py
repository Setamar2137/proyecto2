# MODELO DE PREDICCIÓN (REDES NEURONALES) # 
# DANIEL SEPÚLVEDA 202022738 #
# CIENCIA DE DATOS #

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense #type: ignore
from sklearn.metrics import classification_report, accuracy_score

# Cargar los datos limpios
data = pd.read_csv('C:/Users/danie/Documents/PROYECTO 2/datos_limpios.csv')

# Convertir valores 'True' y 'False' en todas las columnas a 1 y 0
data = data.replace({'True': 1, 'False': 0})

# Asegurarse de que las columnas booleanas sean de tipo numérico después de la conversión
data = data.apply(pd.to_numeric, errors='ignore')

# Preprocesar la columna objetivo (cambiar y_yes a 1 para suscripción, 0 para no suscripción)
data['y_yes'] = data['y_yes'].apply(lambda x: 1 if x == 1 else 0)

# Separar las variables recomendadas del analisis para cada modelo

# Modelo 1: Perfil Demográfico y Financiero
variables_modelo_1 = ['age', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 
                      'job_management', 'job_retired', 'job_self-employed', 'job_services', 
                      'job_student', 'job_technician', 'job_unemployed', 'job_unknown', 
                      'marital_married', 'marital_single', 'balance', 'previous', 'housing_yes', 'loan_yes']
X1 = data[variables_modelo_1]
y1 = data['y_yes']

# Modelo 2: Estrategia de Contacto
variables_modelo_2 = ['campaign', 'previous', 'duration', 'month_aug', 'month_dec', 'month_feb', 
                      'month_jan', 'month_jul', 'month_jun', 'month_mar', 'month_may', 
                      'month_nov', 'month_oct', 'month_sep', 'poutcome_other', 
                      'poutcome_success', 'poutcome_unknown', 'contact_telephone', 'contact_unknown']
X2 = data[variables_modelo_2]
y2 = data['y_yes']

# Dividir en conjuntos de entrenamiento y prueba para cada modelo
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Escalar las variables numéricas
scaler = StandardScaler()
X1_train = scaler.fit_transform(X1_train)
X1_test = scaler.transform(X1_test)
X2_train = scaler.fit_transform(X2_train)
X2_test = scaler.transform(X2_test)

# Definir y entrenar el modelo 1
modelo_1 = Sequential()
modelo_1.add(Dense(16, input_dim=X1_train.shape[1], activation='relu'))
modelo_1.add(Dense(8, activation='relu'))
modelo_1.add(Dense(1, activation='sigmoid'))
modelo_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history_1 = modelo_1.fit(X1_train, y1_train, epochs=20, batch_size=10, validation_split=0.2)

# Definir y entrenar el modelo 2
modelo_2 = Sequential()
modelo_2.add(Dense(16, input_dim=X2_train.shape[1], activation='relu'))
modelo_2.add(Dense(8, activation='relu'))
modelo_2.add(Dense(1, activation='sigmoid'))
modelo_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history_2 = modelo_2.fit(X2_train, y2_train, epochs=20, batch_size=10, validation_split=0.2)

# Evaluación del Modelo 1
y1_pred = (modelo_1.predict(X1_test) > 0.5).astype("int32")
print("Resultados del Modelo 1: Perfil Demográfico y Financiero")
print(classification_report(y1_test, y1_pred))
print("Exactitud del Modelo 1:", accuracy_score(y1_test, y1_pred))

# Evaluación del Modelo 2
y2_pred = (modelo_2.predict(X2_test) > 0.5).astype("int32")
print("Resultados del Modelo 2: Estrategia de Contacto")
print(classification_report(y2_test, y2_pred))
print("Exactitud del Modelo 2:", accuracy_score(y2_test, y2_pred))

# Guardar el modelo 
modelo_1.save('C:/Users/danie/Documents/PROYECTO 2/modelo_1.h5')
modelo_2.save('C:/Users/danie/Documents/PROYECTO 2/modelo_2.h5')

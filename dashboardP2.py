import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

# Cargar los datos y modelos ya entrenados
datos_limpios = pd.read_csv('C:/Users/natal/OneDrive/Documentos/U/2024-2/Analitica/Proyecto 2/datos_limpios.csv')
modelo_1 = load_model('C:/Users/natal/OneDrive/Documentos/U/2024-2/Analitica/Proyecto 2/modelo_1.h5')
modelo_2 = load_model('C:/Users/natal/OneDrive/Documentos/U/2024-2/Analitica/Proyecto 2/modelo_2.h5')

# Selección de las 7 variables base del Modelo 1
columnas_modelo_1 = ['age', 'balance', 'previous', 
                     'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 
                     'job_management', 'job_retired', 'job_self-employed', 
                     'job_services', 'job_student', 'job_technician', 'job_unemployed', 
                     'marital_married', 
                     'housing_yes', 'loan_yes', 'job_unknown','marital_single']

# Seleccionar esas columnas del conjunto de datos
datos_modelo_1 = datos_limpios[columnas_modelo_1].values.astype('float32')

# Hacer predicciones con el modelo ya entrenado
predicciones_modelo_1 = modelo_1.predict(datos_modelo_1)

# Agregar las predicciones al DataFrame original
datos_limpios['predicciones_modelo_1'] = predicciones_modelo_1

# Título del Dashboard
st.title("Análisis Predictivo de Suscripción a Depósitos a Término")

# Opción adicional: checkbox para seleccionar solo clientes con balance positivo
balance_positivo = st.checkbox("Mostrar solo clientes con balance positivo")

# Filtrar datos si se selecciona la opción de balance positivo
if balance_positivo:
    datos_filtrados = datos_limpios[datos_limpios['balance'] > 0]
else:
    datos_filtrados = datos_limpios 

# Crear dos columnas para organizar el dashboard en paralelo
col1, col2 = st.columns(2)

# ================== Columna 1: Aspectos Demográficos ==================
with col1:
    st.header("Aspectos Demográficos")
    opcion_demografica = st.selectbox(
        'Seleccione una característica demográfica para analizar con las predicciones del modelo:',
        ['Edad', 'Trabajo', 'Estado Civil']
    )

    # Mostrar gráficos en función de la selección del usuario
    if opcion_demografica == 'Edad':
        # Crear rangos de edad
        datos_filtrados['age_range'] = pd.cut(datos_filtrados['age'], bins=[18, 25, 35, 45, 55, 65], 
                                              labels=['18-25', '26-35', '36-45', '46-55', '56-65'])

        # Agrupar por rango de edad y calcular la media de las predicciones
        promedio_predicciones_edad = datos_filtrados.groupby('age_range')['predicciones_modelo_1'].mean()

        # Gráfico de barras
        fig, ax = plt.subplots()
        ax.bar(promedio_predicciones_edad.index, promedio_predicciones_edad.values)
        ax.set_xlabel("Rango de Edad")
        ax.set_ylabel("Promedio de Probabilidad de Suscripción (%)")
        ax.set_title("Probabilidad de Suscripción según Rango de Edad")
        st.pyplot(fig)

    elif opcion_demografica == 'Trabajo':
        # Crear una nueva columna 'job' para representar el tipo de trabajo
        def obtener_tipo_trabajo(fila):
            trabajos = ['job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired', 
                        'job_self-employed', 'job_services', 'job_student', 'job_technician', 'job_unemployed', 'job_unknown']
            for trabajo in trabajos:
                if fila[trabajo] == 1:
                    return trabajo.replace('job_', '')  # Eliminar el prefijo 'job_'
            return 'admin.'  # Por si acaso ninguna columna de trabajo es 1

        # Aplicar la función para crear la columna 'job'
        datos_filtrados['job'] = datos_filtrados.apply(obtener_tipo_trabajo, axis=1)

        # Agrupar por tipo de trabajo y calcular la media de las predicciones
        promedio_predicciones_trabajo = datos_filtrados.groupby('job')['predicciones_modelo_1'].mean()

        # Gráfico de barras
        fig, ax = plt.subplots()
        ax.bar(promedio_predicciones_trabajo.index, promedio_predicciones_trabajo.values)
        ax.set_xlabel("Tipo de Trabajo")
        ax.set_ylabel("Promedio de Probabilidad de Suscripción (%)")
        ax.set_title("Probabilidad de Suscripción según Tipo de Trabajo")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

    elif opcion_demografica == 'Estado Civil':
        # Crear una nueva columna 'marital' para representar el estado civil
        def obtener_estado_civil(fila):
            estados_civiles = ['marital_married', 'marital_single']
            for estado in estados_civiles:
                if estado in fila.index and fila[estado] == 1:  # Asegúrate de que el estado esté en el DataFrame
                    return estado.replace('marital_', '')  # Eliminar el prefijo 'marital_'  
            return 'divorced'  # Por si acaso ninguna columna es 1

        # Aplicar la función para crear la columna 'marital'
        datos_filtrados['marital'] = datos_filtrados.apply(obtener_estado_civil, axis=1)

        # Agrupar por estado civil y calcular la media de las predicciones
        promedio_predicciones_civil = datos_filtrados.groupby('marital')['predicciones_modelo_1'].mean()

        # Gráfico de barras
        fig, ax = plt.subplots()
        ax.bar(promedio_predicciones_civil.index, promedio_predicciones_civil.values)
        ax.set_xlabel("Estado Civil")
        ax.set_ylabel("Promedio de Probabilidad de Suscripción (%)")
        ax.set_title("Probabilidad de Suscripción según Estado Civil")
        st.pyplot(fig)

# ================== Columna 2: Situación Financiera ==================
with col2:
    st.header("Situación Financiera")
    opcion_financiera = st.selectbox(
        'Seleccione una característica financiera para analizar con las predicciones del modelo:',
        [ 'Hipoteca', 'Prestamo']
    )

    if  opcion_financiera == 'Hipoteca':
        # Agrupar por Housing y calcular la media de las predicciones
        promedio_predicciones_housing = datos_filtrados.groupby('housing_yes')['predicciones_modelo_1'].mean()

        # Gráfico de barras
        fig, ax = plt.subplots()
        ax.bar(['No', 'Sí'], promedio_predicciones_housing)
        ax.set_xlabel("Tiene Hipoteca")
        ax.set_ylabel("Promedio de Probabilidad de Suscripción (%)")
        ax.set_title("Probabilidad de Suscripción según Propiedad de Hipoteca")
        st.pyplot(fig)

    elif opcion_financiera == 'Prestamo':
        # Agrupar por Loan y calcular la media de las predicciones
        promedio_predicciones_loan = datos_filtrados.groupby('loan_yes')['predicciones_modelo_1'].mean()

        # Gráfico de barras
        fig, ax = plt.subplots()
        ax.bar(['No', 'Sí'], promedio_predicciones_loan)
        ax.set_xlabel("Tiene Préstamo")
        ax.set_ylabel("Promedio de Probabilidad de Suscripción (%)")
        ax.set_title("Probabilidad de Suscripción según Préstamos")
        st.pyplot(fig)


#------------------------------------------CALCULADORA DE PROBABILIDAD POR CLIENTE----------------------------------

# Título del Dashboard
st.title("Calculadora de Probabilidad de Suscripción a Depósitos a Término")

# Cuadro de entrada para la edad
edad = st.number_input("Introduce la edad del cliente:", min_value=18, max_value=100, value=30)

# Cuadro de selección para balance positivo o negativo
balance = st.selectbox("Selecciona el balance del cliente:", ["Positivo", "Negativo"])

# Cuadro de selección para el tipo de trabajo
tipo_trabajo = st.selectbox("Selecciona el tipo de trabajo del cliente:", [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", 
    "self-employed", "services", "student", "technician", "unemployed", "unknown"
])

# Cuadro de selección para el estado civil
estado_civil = st.selectbox("Selecciona el estado civil del cliente:", ["married", "single", "divorced"])

# Casillas de verificación para hipoteca y préstamo
hipoteca = st.checkbox("¿El cliente tiene hipoteca?")
prestamo = st.checkbox("¿El cliente tiene préstamo?")

# Mapeo de las entradas a las variables esperadas por el modelo
# Suponiendo que en el modelo "hipoteca" y "préstamo" son variables binarias: 1 para sí, 0 para no
hipoteca_valor = 1 if hipoteca else 0
prestamo_valor = 1 if prestamo else 0
balance_valor = 1 if balance == "Positivo" else -1

# Transformación del tipo de trabajo en variables dummy
trabajos = {
    "admin.": [0,0,0,0,0,0,0,0,0,0,0],  # admin. es la categoría base
    "blue-collar": [1,0,0,0,0,0,0,0,0,0,0],
    "entrepreneur": [0,1,0,0,0,0,0,0,0,0,0],
    "housemaid": [0,0,1,0,0,0,0,0,0,0,0],
    "management": [0,0,0,1,0,0,0,0,0,0,0],
    "retired": [0,0,0,0,1,0,0,0,0,0,0],
    "self-employed": [0,0,0,0,0,1,0,0,0,0,0],
    "services": [0,0,0,0,0,0,1,0,0,0,0],
    "student": [0,0,0,0,0,0,0,1,0,0,0],
    "technician": [0,0,0,0,0,0,0,0,1,0,0],
    "unemployed": [0,0,0,0,0,0,0,0,0,1,0],
    "unknown": [0,0,0,0,0,0,0,0,0,0,1]
}

# Variables para estado civil
estados_civiles = {
    "married": [1, 0],
    "single": [0, 1],
    "divorced": [0, 0]  # Si no está casado o soltero, entonces es divorciado
}

# Crear la lista de entrada para el modelo
entrada_modelo = [
    edad, balance_valor, 0  # 'previous' no se usa en la predicción del modelo en este caso
] + trabajos[tipo_trabajo] + estados_civiles[estado_civil] + [hipoteca_valor, prestamo_valor]

# Convertir la entrada en un arreglo numpy y darle la forma correcta para el modelo
entrada_modelo = np.array(entrada_modelo).reshape(1, -1)

# Hacer la predicción
if st.button("Calcular Probabilidad"):
    probabilidad = modelo_1.predict(entrada_modelo)[0][0]
    st.write(f"La probabilidad de suscripción para este cliente es: {probabilidad * 100:.2f}%")

    #------------------------------------------------------------------Pregunta de negocio 2-------------------------------------------------------

predicciones_modelo_2 = modelo_2.predict(datos_limpios[['campaign', 'previous', 'duration', 
    'month_jan', 'month_feb', 'month_mar', 'month_may', 'month_jun', 'month_jul', 'month_aug', 
    'month_sep', 'month_oct', 'month_nov', 'month_dec', 'poutcome_success', 'poutcome_unknown', 
    'poutcome_other', 'contact_telephone', 'contact_unknown']].values.astype('float32'))

datos_limpios['predicciones_modelo_2'] = predicciones_modelo_2

# Título del Dashboard
st.title("Análisis Predictivo de Estrategias de Contacto para Depósitos a Término")


# ---------------------  Duración del Contacto vs. Probabilidad de Suscripción ---------------------
st.subheader("Duración del Contacto vs. Probabilidad de Suscripción")

# Crear la columna contact_cellular (si contact_telephone y contact_unknown son 0, entonces es contact_cellular)
datos_limpios['contact_cellular'] = ((datos_limpios['contact_telephone'] == 0) & (datos_limpios['contact_unknown'] == 0)).astype(int)

# Selección del rango de duración del contacto
duracion_rango = st.slider("Seleccione el rango de duración del contacto (en segundos):", 0, 500, (0, 216))

# Selección del canal de contacto
tipo_contacto = st.selectbox("Seleccione el canal de contacto:", ['contact_telephone', 'contact_unknown', 'contact_cellular'])

# Filtrar los datos según la duración del contacto y el tipo de contacto
filtered_data = datos_limpios[
    (datos_limpios['duration'] >= duracion_rango[0]) & 
    (datos_limpios['duration'] <= duracion_rango[1]) & 
    (datos_limpios[tipo_contacto] == 1)
]

# Agrupar por duración del contacto y calcular la media de las predicciones
promedio_predicciones_duration = filtered_data.groupby('duration')['predicciones_modelo_2'].mean()

# Gráfico de líneas
fig, ax = plt.subplots()
ax.plot(promedio_predicciones_duration.index, promedio_predicciones_duration.values)
ax.set_xlabel("Duración del Contacto (segundos)")
ax.set_ylabel("Probabilidad de Suscripción (%)")
ax.set_title(f"Probabilidad de Suscripción según la Duración del Contacto ({tipo_contacto})")

# Mostrar gráfico en Streamlit
st.pyplot(fig)




# --------------------- . Probabilidad de Suscripción por Número de Contactos Previos y Canal de Contacto ---------------------
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

# Crear la columna contact_cellular (si contact_telephone y contact_unknown son 0, entonces es contact_cellular)
datos_limpios['contact_cellular'] = ((datos_limpios['contact_telephone'] == 0) & (datos_limpios['contact_unknown'] == 0)).astype(int)

# Selección del rango de número de contactos previos
st.subheader("Probabilidad de Suscripción por Número de Contactos Previos y Canal de Contacto")
rango_contactos_previos = st.slider("Seleccione el rango de número de contactos previos:", 0, 20, (0, 15))

# Filtrar los datos según el rango de número de contactos previos
filtered_data = datos_limpios[(datos_limpios['previous'] >= rango_contactos_previos[0]) & 
                              (datos_limpios['previous'] <= rango_contactos_previos[1])]

# Agrupar por número de contactos previos y canal de contacto, y calcular la media de las predicciones
promedio_predicciones_previous_contact = filtered_data.groupby(['previous', 'contact_telephone', 'contact_unknown', 'contact_cellular'])['predicciones_modelo_2'].mean().reset_index()

# Crear el gráfico de líneas
fig, ax = plt.subplots(figsize=(10, 6))

# Graficar una línea por cada canal de contacto
for tipo_contacto in ['contact_telephone', 'contact_unknown', 'contact_cellular']:
    data_contacto = promedio_predicciones_previous_contact[promedio_predicciones_previous_contact[tipo_contacto] == 1]
    ax.plot(data_contacto['previous'], data_contacto['predicciones_modelo_2'], marker='o', label=tipo_contacto)

ax.set_xlabel("Número de Contactos Previos")
ax.set_ylabel("Probabilidad de Suscripción (%)")
ax.set_title("Probabilidad de Suscripción según Número de Contactos Previos y Canal de Contacto")
ax.legend(title="Canal de Contacto")
plt.xticks(rotation=45)
st.pyplot(fig)


# --------------------- Estrategia Temporal según la Duración de la Llamada --------------------- 
st.subheader("Estrategia Temporal según la Duración de la Llamada")

# Crear las columnas 'month' y 'poutcome' para facilitar el análisis
months = ['month_jan', 'month_feb', 'month_mar', 'month_may', 'month_jun', 'month_jul', 
          'month_aug', 'month_sep', 'month_oct', 'month_nov', 'month_dec']

# Crear una columna de mes para facilitar el gráfico
def obtener_mes(fila):
    for mes in months:
        if fila[mes] == 1:
            return mes.replace('month_', '').capitalize()  # Eliminar 'month_' y capitalizar el nombre del mes
    return None

# Aplicar la función para crear la columna 'mes'
datos_limpios['mes'] = datos_limpios.apply(obtener_mes, axis=1)

# Crear una columna de resultado anterior ('poutcome') basado en los datos
def obtener_poutcome(fila):
    if fila['poutcome_success'] == 1:
        return 'Éxito'
    elif fila['poutcome_unknown'] == 1:
        return 'Desconocido'
    elif fila['poutcome_other'] == 1:
        return 'Otro'
    else:
        return 'Fracaso'

# Aplicar la función para crear la columna 'poutcome'
datos_limpios['poutcome'] = datos_limpios.apply(obtener_poutcome, axis=1)

# Filtrar los datos por resultado anterior seleccionado
resultado_anterior = st.selectbox("Seleccione el resultado anterior de la campaña (poutcome):", 
                                  ['Éxito', 'Fracaso', 'Desconocido', 'Otro'])

# Filtrar los datos por el resultado anterior
datos_filtrados = datos_limpios[datos_limpios['poutcome'] == resultado_anterior]

# Agrupar por mes y calcular la duración promedio de las llamadas
duracion_promedio_mes = datos_filtrados.groupby('mes')['duration'].mean()

# Gráfico de barras para mostrar la duración promedio de llamadas por mes
fig, ax = plt.subplots()
colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#b07aa1']
ax.bar(duracion_promedio_mes.index, duracion_promedio_mes.values, color=colores)
ax.set_xlabel("Mes de Contacto")
ax.set_ylabel("Duración Promedio de la Llamada (segundos)")
ax.set_title(f"Duración Promedio de la Llamada según Mes de Contacto y Resultado: {resultado_anterior}")

# Mostrar el gráfico en Streamlit
st.pyplot(fig)




import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.regression import predict_model

path = "D:/Downloads/INTELIGENCIA ARTIFICIAL/ACTIVIDAD VALENTINA hechaaa/STREAMLIT 22/"
# Cargar el modelo y los datos en st.session_state si no están ya cargados
if 'modelo' not in st.session_state:
    with open('best_model.pkl', 'rb') as model_file:
        st.session_state['modelo'] = pickle.load(model_file)

if 'test_data' not in st.session_state:
    st.session_state['test_data'] = pd.read_csv('prueba_APP.csv',header = 0,sep=";",decimal=",")

# Función para predicción individual
def prediccion_individual():
    st.header("Predicción manual de datos")

    Avg = st.text_input("Avg Session Length", value="32.063775")	
    Time_App = st.text_input("Time on App", value="10.71915")	
    Time_Website = st.text_input("Time on Website", value="37.712509")
    Length_Membership = st.text_input("Length of Membership", value="3.004743")
    dominio = st.selectbox("Seleccione el dominio:", ['gmail', 'Otro', 'hotmail', 'yahoo'])
    Tec = st.selectbox("Seleccione el tipo de dispositivo:", ['Smartphone', 'Portatil', 'PC', 'Iphone'])

    if st.button("Calcular predicción manual"):

        # Crear el dataframe de los inputs
    
        user = pd.DataFrame({'x1':[Avg],'x2': [Time_App],'x3': [Time_Website],
                             'x4': [Length_Membership], 'x5': [dominio], 'x6':[Tec]})
        
        prueba_ = st.session_state['test_data']
        cuantitativas = ['Avg. Session Length','Time on App',
                 'Time on Website', 'Length of Membership']
        categoricas = ['dominio', 'Tec']
        prueba1 = pd.concat([prueba_.get(cuantitativas),prueba_.get(categoricas)],axis = 1)
        user.columns = prueba1.columns
        prueba2_ = pd.concat([user, prueba1], axis=0)
        prueba2_.index = range(prueba2_.shape[0])

        
        # Hacer predicciones
        predictions = predict_model(st.session_state['modelo'], data=prueba2_)

        predictions["Price"] = predictions["prediction_label"].map(float)

        st.write(f'La predicción es: {predictions.iloc[0]["Price"]}')

    if st.button("Volver al menú principal"):
        st.session_state['menu'] = 'main'

# Función para predicción por base de datos
def prediccion_base_datos():
    st.header("Cargar archivo para predecir")
    uploaded_file = st.file_uploader("Cargar archivo Excel o CSV", type=["xlsx", "csv"])

    if st.button("Predecir con archivo"):
        if uploaded_file is not None:
            try:
                # Cargar el archivo directamente sin usar tempfile
                if uploaded_file.name.endswith(".csv"):
                    prueba = pd.read_csv(uploaded_file,header = 0,sep=";",decimal=",")
                else:
                    prueba = pd.read_excel(uploaded_file)
                covariables = ['Avg. Session Length', 'Time on App', 'Time on Website',
                           'Length of Membership', 'dominio', 'Tec']
                base = prueba.get(covariables)
                prediccion = predict_model(st.session_state['modelo'], data=base)
                predicciones = pd.DataFrame({'Email': prueba["Email"],
                                      'Precio': prediccion["prediction_label"]})
                st.write("Predicciones generadas correctamente!")
                st.write(predicciones)

                st.download_button(label="Descargar archivo de predicciones",
                               data=predicciones.to_csv(index=False),
                               file_name="Predicciones.csv",
                               mime="text/csv")

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Por favor, cargue un archivo válido.")

    if st.button("Volver al menú principal"):
        st.session_state['menu'] = 'main'

# Función principal para mostrar el menú de opciones
def menu_principal():
    st.title("API de Predicción Académica")
    option = st.selectbox("Seleccione una opción", ["", "Predicción Individual", "Predicción Base de Datos"])

    if option == "Predicción Individual":
        st.session_state['menu'] = 'individual'
    elif option == "Predicción Base de Datos":
        st.session_state['menu'] = 'base_datos'

# Lógica para manejar el flujo de la aplicación
if 'menu' not in st.session_state:
    st.session_state['menu'] = 'main'

if st.session_state['menu'] == 'main':
    menu_principal()
elif st.session_state['menu'] == 'individual':
    prediccion_individual()
elif st.session_state['menu'] == 'base_datos':
    prediccion_base_datos()
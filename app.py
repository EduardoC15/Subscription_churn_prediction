import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Charging the model
clf = joblib.load('modelo_churn.joblib')

st.title("Prueba de Modelo de Churn Prediction")

is_tv_subscriber = st.number_input("¿Es Suscriptor de TV?", min_value=0, max_value=1, value=1)
is_movie_subscriber = st.number_input("¿Es Suscriptor de Paquete de Películas?", min_value=0, max_value=1, value=1)
subscription_age = st.number_input("Edad de Suscripción", min_value=0, max_value=12, value=2)
bill_avg = st.number_input("Factura", min_value=0, max_value=406, value=13)
reamining_contract = st.number_input("Contrato Restante (meses)", min_value=0.0, max_value=2.9, value=0.7)
download_avg = st.number_input("Descarga de datos", min_value=0.0, max_value=4415.0, value=50.0)
upload_avg = st.number_input("Carga de datos", min_value=0.0, max_value=453.0, value=5.0)

if st.button("Predecir Churn"):
    user_data = pd.DataFrame({
        "is_tv_subscriber": [is_tv_subscriber],
        "is_movie_package_subscriber": [is_movie_subscriber],
        "subscription_age": [subscription_age],
        "bill_avg": [bill_avg],
        "reamining_contract": [reamining_contract],
        "download_avg": [download_avg],
        "upload_avg": [upload_avg]
    })

    prediction = clf.predict(user_data)

    if prediction[0] == 0:
        st.write("Resultado de la Predicción: No Churn")
    else:
        st.write("Resultado de la Predicción: Churn")

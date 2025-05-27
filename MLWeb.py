import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Load model
model = pickle.load(open('model_prediksi_harga_mobil.sav', 'rb'))

# Sidebar mode
mode = st.sidebar.radio("Pilih Mode Input", ["Cepat", "Lanjutan"])

st.title('🚗 Prediksi Harga Mobil')
st.caption("Aplikasi ini memprediksi harga mobil berdasarkan Highway MPG, Curb Weight, dan Horsepower.")

# Tabs layout
tab1, tab2, tab3 = st.tabs(["📊 Dataset", "📈 Grafik", "📥 Prediksi"])

with tab1:
    st.subheader("Data Mobil")
    df = pd.read_csv("CarPrice_Assignment.csv")
    st.dataframe(df)

    # Statistik deskriptif
    with st.expander("Lihat Statistik Deskriptif"):
        st.write(df.describe())

with tab2:
    st.subheader("Visualisasi Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Highway-mpg")
        st.line_chart(df["highwaympg"])

    with col2:
        st.write("Curb Weight")
        st.area_chart(df["curbweight"])

    with col3:
        st.write("Horsepower")
        st.bar_chart(df["horsepower"])

with tab3:
    st.subheader("Input Fitur Mobil untuk Prediksi")

    if mode == "Cepat":
        highwaympg = st.slider("Highway MPG", 10, 70, 30)
        curbweight = st.slider("Curb Weight (kg)", 1000, 5000, 2500)
        horsepower = st.slider("Horsepower", 50, 400, 150)
    else:
        highwaympg = st.number_input("Highway MPG", 10, 70, 30)
        curbweight = st.number_input("Curb Weight (kg)", 1000, 5000, 2500)
        horsepower = st.number_input("Horsepower", 50, 400, 150)

    if st.button("🔍 Prediksi Harga"):
        car_prediction = model.predict([[highwaympg, curbweight, horsepower]])
        harga = float(car_prediction[0])
        harga_formatted = f"${harga:,.2f}"

        st.metric(label="💰 Prediksi Harga Mobil", value=harga_formatted)
        st.success(f"Harga mobil diperkirakan sekitar **{harga_formatted}** berdasarkan input fitur yang diberikan.")

        st.caption("Model ini menggunakan Regresi Linear dari library scikit-learn.")
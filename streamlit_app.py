import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn import metrics
from pickle import dump
import joblib
import altair as alt
from streamlit_option_menu import option_menu
from sklearn.metrics import accuracy_score


st.title("Klasifikasi & Prediksi Indeks Standar Pencemaran Udara (ISPU)")
st.write("Yoga Tirta Permana | 200411100142")
# with st.sidebar:
selected = option_menu(
    menu_title=None,
    options=["View Data", "Preprocessing", "Modeling", "Predict"],
    icons=["data", "process", "model", "predict"],
    orientation="horizontal",
)

df_train = pd.read_csv("indeks-standar-pencemar-udara-tahun-2020.csv")
y = df_train['kategori']


# View Data
if (selected == "View Data"):
    st.write("# View Data")
    view_data, info_data = st.tabs(["View Data", "Info Data"])

    with view_data:
        st.write("## Menampilkan Dataset :")
        st.dataframe(df_train)
        st.write("""
                  ### Fitur:
                  - tanggal: Tanggal pengukuran kualitas udara
                  - pm10: Kadar Partikulat (PM10)
                  - so2: Kadar Sulfur Dioksida (SO2)
                  - co: Kadar Karbon Monoksida (CO)
                  - o3: Kadar Lapisan Ozon (O3)
                  - no2: Kadar Nitrogen Dioksida (NO2)
                  - max: Nilai ukur paling tinggi dari seluruh parameter yang diukur dalam waktu yang sama
                  - critical: Parameter yang hasil pengukurannya paling tinggi
                  - kategori: Kategori hasil perhitungan indeks standar pencemaran udara
                    > BAIK

                    > SEDANG

                    > TIDAK SEHAT
                """)

    with info_data:
        st.write("## Informasi Dataset :")
        st.info(f"""
                  - Jumlah Data : {df_train.shape[0]} data
                  - Jumlah Fitur : {df_train.shape[1]} fitur
                """)
        st.write("#### Tipe data", df_train.dtypes)
        st.write("#### Nilai maksimal data", df_train.max())
        st.write("#### Nilai minimal data", df_train.min())


# Preprocessing
elif (selected == 'Preprocessing'):
    st.write("# Preprocessing")
    data_asli, normalisasi = st.tabs(["View Data", "Normalisasi"])

    with data_asli:
        st.write('Data Sebelum Preprocessing')
        st.dataframe(df_train)
        st.write("""
                  #### Data yang di Drop:
                  - tanggal: Tanggal pengukuran kualitas udara
                  - max: Nilai ukur paling tinggi dari seluruh parameter yang diukur dalam waktu yang sama
                  - critical: Parameter yang hasil pengukurannya paling tinggi
                  - kategori: Kategori hasil perhitungan indeks standar pencemaran udara
                """)

    with normalisasi:
        st.write('Data Setelah Preprocessing dengan Min-Max Scaler')
        # st.write('Kecuali data gender, chest_pain, & output')
        scaler = MinMaxScaler()
        # df_train_pre = scaler.fit_transform(df_train.drop(columns=['gender', 'chest_pain', 'output']))
        df_train_pre = scaler.fit_transform(df_train.drop(
            columns=['tanggal', 'max', 'critical', 'kategori']))
        st.dataframe(df_train_pre)
        st.write("""
                  #### Data yang di Normalisasi:
                  - pm10: Kadar Partikulat (PM10)
                  - so2: Kadar Sulfur Dioksida (SO2)
                  - co: Kadar Karbon Monoksida (CO)
                  - o3: Kadar Lapisan Ozon (O3)
                  - no2: Kadar Nitrogen Dioksida (NO2)
                """)

    # Save Scaled
    joblib.dump(df_train_pre, 'model/df_train_pre.sav')
    joblib.dump(scaler, 'model/df_scaled.sav')


# Modeling
elif (selected == 'Modeling'):
    st.write("# Modeling")
    # st.caption("Splitting Data yang digunakan merupakan 70:30, 30\% untuk data test dan 70\% untuk data train\nIterasi K di lakukan sebanyak 20 Kali")
    nb, knn, dtc = st.tabs(['Naive-Bayes', 'SVM', 'Random Forest'])

    # Naive-Bayes Gaussian
    with nb:
        df_train_pre = joblib.load('model/df_train_pre.sav')
        x_train, x_test, y_train, y_test = train_test_split(
            df_train_pre, y, test_size=0.3, random_state=0)

        nb = GaussianNB()
        nb.fit(x_train, y_train)

        # Save Model
        # Menyimpan Model ke dalam folder model
        joblib.dump(nb, 'model/nb_model.sav')

        y_pred = nb.predict(x_test)
        akurasi = accuracy_score(y_test, y_pred)

        st.info(f'Akurasi yang dihasilkan Naive Bayes = {akurasi*100}%')
        st.write(df_train_pre)

    # K-Nearest Neighbour
    with knn:
        df_train_pre = joblib.load('model/df_train_pre.sav')
        x_train, x_test, y_train, y_test = train_test_split(
            df_train_pre, y, test_size=0.3, random_state=0)
        scores = {}
        for i in range(1, 20+1):
            KN = KNeighborsClassifier(n_neighbors=i)
            KN.fit(x_train, y_train)
            y_pred = KN.predict(x_test)
            scores[i] = accuracy_score(y_test, y_pred)

        best_k = max(scores, key=scores.get)
        st.info(f"Akurasi yang dihasilkan SVM = {max(scores.values())* 100}%")
        st.write(df_train_pre)
        st.success(f"Parameter Terbaik : {best_k} berada di Index : {best_k-1}")

        # Create Chart
        st.write('Grafik Akurasi K')
        accuration_k = np.array(list(scores.values()))
        chart_data = pd.DataFrame(accuration_k, columns=['Akurasi'])
        st.line_chart(chart_data)

        knn = KNeighborsClassifier(n_neighbors=best_k)
        knn.fit(x_train, y_train)

        # Save Model
        # Menyimpan Model ke dalam folder model
        joblib.dump(knn, 'model/knn_model.sav')

    # Decision Tree Classifier
    with dtc:
        df_train_pre = joblib.load('model/df_train_pre.sav')
        x_train, x_test, y_train, y_test = train_test_split(
            df_train_pre, y, test_size=0.3, random_state=0)

        dtc = DecisionTreeClassifier()
        dtc.fit(x_train, y_train)

        # Save Model
        # Menyimpan Model ke dalam folder model
        joblib.dump(dtc, 'model/dtc_model.sav')

        y_pred = dtc.predict(x_test)
        akurasi = accuracy_score(y_test, y_pred)

        st.info(f'Akurasi yang dihasilkan Random Forest = {akurasi*100}%')
        st.write(df_train_pre)


# Predict
elif (selected == 'Predict'):
    st.write("# Predict")
    st.write(
        "Prediksi menggunakan Model Random Forest dengan akurasi tertinggi sebesar 98%")

    pm10 = st.number_input("Masukkan Kadar Partikulat (PM10)")
    so2 = st.number_input("Masukkan Kadar Sulfur Dioksida (SO2)")
    co = st.number_input("Masukkan Kadar Karbon Monoksida (CO)")
    o3 = st.number_input("Masukkan Kadar Lapisan Ozon (O3)")
    no2 = st.number_input("Masukkan Kadar Nitrogen Dioksida (NO2)")

    st.write("Prediksi Indeks Standar Pencemaran Udara (ISPU)")
    cek_knn = st.button('Predict')
    inputan = [[pm10, so2, co, o3, no2]]

    scaler = joblib.load('model/df_scaled.sav')
    data_scaler = scaler.transform(inputan)

    FIRST_IDX = 0
    k_nn = joblib.load("model/knn_model.sav")
    if cek_knn:
        hasil_test = k_nn.predict(data_scaler)[FIRST_IDX]
        if hasil_test == 0:
            st.success(
                f'Status Udara BAIK')
        if hasil_test == 0:
            st.warning(
                f'Status Udara SEDANG')
        else:
            st.error(
                f'Status Udara TIDAK SEHAT')

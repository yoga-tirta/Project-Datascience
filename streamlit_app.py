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


st.title("Heart Attack Analysis & Prediction Application")
st.write("Yoga Tirta Permana | 200411100142")
# with st.sidebar:
selected = option_menu(
    menu_title  = None,
    options     = ["View Data","Preprocessing","Modelling","Implementation"],
    icons       = ["data","process","model","implemen"],
    orientation = "horizontal",
)

df_train = pd.read_csv("heart.csv")
y = df_train['output']


# View Data
if(selected == "View Data"):
    st.write("# View Data")
    view_data, info_data = st.tabs(["View Data","Info Data"])
    
    with view_data:
        st.write("## Menampilkan Dataset :")
        st.dataframe(df_train)
        st.write("""
                  ### Fitur:
                  - age: Umur dari pasien
                  - gender: Jenis kelamin pasien
                    > 1 - Pria

                    > 0 - Wanita
                  - chest_pain: Tipe nyeri dada pasien
                    > 0 - typical angina : Ketidaknyamanan pada dada dengan durasi tertentu

                    > 1 - atypical angina : Rasa sakit yang dipicu oleh aktifitas fisik berlebihan atau stres emosional

                    > 2 - non-anginal pain : Rasa sakit biasa yang bisa diredakan dengan istirahat

                    > 3 - asymtomatic : Tidak ada gejala
                  - blood_pressure: Tekanan darah pasien (mm/Hg) *
                  - cholestoral: Kadar kolesterol pasien (mm/dl) *
                  - heart_rate: Detak jantung maksimal pasien *
                  - oldpeak: Tingkat depresi pasien *
                  - output: Hasil
                    > 1 - True

                    > 0 - False
                """)
        st.error(f"Note : *Informasi bisa didapatkan melalui pemeriksaan dokter")

    with info_data:
        st.write("## Informasi Dataset :")
        st.info(f"""
                  - Jumlah Data : {df_train.shape[0]} data
                  - Jumlah Fitur : {df_train.shape[1]} fitur
                """)
        st.write("#### Tipe data",df_train.dtypes)
        st.write("#### Nilai maksimal data",df_train.max())
        st.write("#### Nilai minimal data",df_train.min())


# Preprocessing
elif(selected == 'Preprocessing'):
  st.write("# Preprocessing")
  data_asli, normalisasi = st.tabs(["View Data","Normalisasi"])

  with data_asli:
    st.write('Data Sebelum Preprocessing')
    st.dataframe(df_train)

  with normalisasi:
    st.write('Data Setelah Preprocessing dengan Min-Max Scaler')
    #st.write('Kecuali data gender, chest_pain, & output')
    scaler = MinMaxScaler()
    #df_train_pre = scaler.fit_transform(df_train.drop(columns=['gender', 'chest_pain', 'output']))
    df_train_pre = scaler.fit_transform(df_train.drop(columns=['output']))
    st.dataframe(df_train_pre)

  # Save Scaled
  joblib.dump(df_train_pre, 'model/df_train_pre.sav')
  joblib.dump(scaler,'model/df_scaled.sav')


# Modelling
elif(selected == 'Modelling'):
  st.write("# Modelling")
  #st.caption("Splitting Data yang digunakan merupakan 70:30, 30\% untuk data test dan 70\% untuk data train\nIterasi K di lakukan sebanyak 20 Kali")
  knn, nb, dtc = st.tabs(['K-NN', 'Naive-Bayes', 'Decission Tree'])
  
  # K-Nearest Neighbour
  with knn:
    df_train_pre = joblib.load('model/df_train_pre.sav')
    x_train, x_test, y_train, y_test = train_test_split(df_train_pre, y, test_size = 0.3, random_state = 0)
    scores = {}
    for i in range(1, 20+1):
        KN = KNeighborsClassifier(n_neighbors = i)
        KN.fit(x_train, y_train)
        y_pred = KN.predict(x_test)
        scores[i] = accuracy_score(y_test, y_pred)
        
    best_k = max(scores, key=scores.get)
    st.info(f"Akurasi yang dihasilkan K-NN = {max(scores.values())* 100}%")
    st.write(df_train_pre)
    st.success(f"K Terbaik : {best_k} berada di Index : {best_k-1}")
    
    # Create Chart 
    st.write('Grafik Akurasi K')
    accuration_k = np.array(list(scores.values()))
    chart_data = pd.DataFrame(accuration_k, columns=['Akurasi'])
    st.line_chart(chart_data)
    
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(x_train, y_train)

    # Save Model
    joblib.dump(knn, 'model/knn_model.sav') # Menyimpan Model ke dalam folder model
  
  # Naive-Bayes Gaussian
  with nb:
    df_train_pre = joblib.load('model/df_train_pre.sav')
    x_train, x_test, y_train, y_test = train_test_split(df_train_pre, y, test_size = 0.3, random_state = 0)
    
    nb = GaussianNB()
    nb.fit(x_train, y_train)

    # Save Model
    joblib.dump(nb, 'model/nb_model.sav') # Menyimpan Model ke dalam folder model
    
    y_pred = nb.predict(x_test)
    akurasi = accuracy_score(y_test,y_pred)
    
    st.info(f'Akurasi yang dihasilkan Naive-Bayes = {akurasi*100}%')
    st.write(df_train_pre)

  # Decision Tree Classifier
  with dtc:
    df_train_pre = joblib.load('model/df_train_pre.sav')
    x_train, x_test, y_train, y_test = train_test_split(df_train_pre, y, test_size = 0.3, random_state = 0)
    
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)

    # Save Model
    joblib.dump(dtc, 'model/dtc_model.sav') # Menyimpan Model ke dalam folder model
    
    y_pred = dtc.predict(x_test)
    akurasi = accuracy_score(y_test,y_pred)
    
    st.info(f'Akurasi yang dihasilkan Decision Tree = {akurasi*100}%')
    st.write(df_train_pre)


# Implementasi
elif(selected == 'Implementation'):
  st.write("# Implementation")
  st.write("Implementasi menggunakan Model K-NN dengan akurasi tertinggi sebesar 80.2%")

  nama_pasien = st.text_input("Masukkan Nama")
  age = st.number_input("Masukkan Umur (29 - 77)", min_value=29, max_value=77)
  gender = st.number_input("Masukkan Jenis Kelamin (1 = Pria, 0 = Wanita)", min_value=0, max_value=1)
  chest_pain = st.number_input("Masukkan Type Nyeri Dada (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymtomatic)", min_value=0, max_value=3)
  blood_pressure = st.number_input("Masukkan Tekanan Darah (mm/Hg) (94 - 200)", min_value=94, max_value=200)
  cholestoral = st.number_input("Masukkan Kadar Kolesterol (mm/dl) (126 - 564)", min_value=126, max_value=564)
  heart_rate = st.number_input("Masukkan Detak Jantung Maximal (71 - 202)", min_value=71, max_value=202)
  oldpeak = st.number_input("Masukkan Oldpeak (0 - 6.2)", min_value=0.0, max_value=6.2)

  st.write("Cek apakah pasien mengidap serangan jantung atau tidak")
  cek_knn = st.button('Cek Pasien')
  inputan = [[age, gender, chest_pain, blood_pressure, cholestoral, heart_rate, oldpeak]]
  
  scaler = joblib.load('model/df_scaled.sav')
  data_scaler = scaler.transform(inputan)

  FIRST_IDX = 0
  k_nn = joblib.load("model/knn_model.sav")
  if cek_knn:
    hasil_test = k_nn.predict(data_scaler)[FIRST_IDX]
    if hasil_test == 0:
      st.success(f'Nama Pasien {nama_pasien} Tidak Mengidap Serangan Jantung Berdasarkan Model K-NN')
    else:
      st.error(f'Nama Pasien {nama_pasien} Mengidap Serangan Jantung Berdasarkan Model K-NN')
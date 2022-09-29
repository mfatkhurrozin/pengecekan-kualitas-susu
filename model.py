#import package
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time

# import the data
data = pd.read_csv("milknewfix.csv")
image = Image.open("susu.jpg")
st.title("Selamat datang di Pengecekan Kulaitas Susu")
st.image(image, use_column_width=True)

# checking the data
st.write("Aplikasi ini bertujuan unutukMengecek Kualitas Susu")
check_data = st.checkbox("Lihat contoh data")
if check_data:
    st.write(data[1:10])
st.write("Sekarang masukan data parameter untuk melihat hasil pengecekan")

# input the numbers
ph = st.slider("PH Pada Susu", float(
    data.ph.min()), float(data.ph.max()), float(data.ph.mean()))
temprature = st.slider("Temperature", int(
    data.temprature.min()), int(data.temprature.max()), int(data.temprature.mean()))
# taste = st.slider("Apakah Rasa Susu Enak? (1=Ya|0=Tidak)", int(data.taste.min()), int(
#     data.taste.max()), int(data.taste.mean()))
odor = st.slider("Apakah Bau Susu Enak? (1=Ya|0=Tidak)", int(data.odor.min()), int(
    data.odor.max()), int(data.odor.mean()))
fat = st.slider("Kadar Lemak Pada Susu? (1=Tingggi|0=Rendah)", int(data.fat.min()), int(
    data.fat.max()), int(data.fat.mean()))
turbidity = st.slider("Kekeruhan Warna Pada Susu? (1=Tingggi|0=Rendah)", int(data.turbidity.min()), int(
    data.turbidity.max()), int(data.turbidity.mean()))
colour = st.slider("Tingkat Warna pada susu", float(data.colour.min()), float(
    data.colour.max()), float(data.colour.mean()))


# splitting your data
X = data.drop('grade', axis=1)
y = data['grade']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=20, random_state=42)

# # modelling step
# # Linear Regression model
# # import your model
# model = LinearRegression()
# # fitting and predict your model
# model.fit(X_train, y_train)
# model.predict(X_test)
# errors = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
# predictions = model.predict(
#     [[temp, humid, tv, odor, fat, turbidity]])[0]
# akurasi = np.sqrt(r2_score(y_test, model.predict(X_test)))

# =============================================================================
# #RandomForestModel
#model2 = RandomForestRegressor(random_state=0)
# model2.fit(X_train,y_train)
# model2.predict(X_test)
#errors = np.sqrt(mean_squared_error(y_test,model2.predict(X_test)))
#predictions = model2.predict([[temp,humid,tv,odor,fat,turbidity]])[0]
#akurasi= np.sqrt(r2_score(y_test,model2.predict(X_test)))
# =============================================================================

# =============================================================================
# DecissionTreeModel
model3 = DecisionTreeRegressor(random_state=42)
model3.fit(X_train, y_train)
model3.predict(X_test)
errors = np.sqrt(mean_squared_error(y_test, model3.predict(X_test)))
predictions = model3.predict(
    # [[ph, temprature, taste, odor, fat, turbidity, colour]])[0]
    [[ph, temprature, odor, fat, turbidity, colour]])[0]
akurasi = np.sqrt(r2_score(y_test, model3.predict(X_test)))
# =============================================================================

if predictions == 0:
    hasil = ("Kualitas Rendah")
elif predictions == 1:
    hasil = ("Kualitas Sedang")
else:
    hasil = ("Kualitas Tinggi")

# hasilakurasi = akurasi*100
# checking prediction house price
if st.button("Cek!"):
    st.header("Hasil Pengecekan : {}".format(hasil))
    # st.subheader("Akurasi : {}".format(hasilakurasi))

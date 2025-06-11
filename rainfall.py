import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_pacf
import xgboost as xgb
from sklearn.metrics import mean_squared_error
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

st.title("Observasi Data Curah Hujan")
upload_file = st.file_uploader("Pilih File Excel Anda", type=['xlsx', 'xls'])

if upload_file is not None:
    st.write('File telah diupload')
    data1 = pd.read_excel(upload_file)
    data1.replace({8888: np.nan, 9999: np.nan}, inplace=True)
    data1.set_index('DATA TIMESTAMP', inplace=True)
    data1.index = pd.to_datetime(data1.index)

    st.session_state.data1 = data1

    st.subheader('Data Preview')
    st.write(data1.head())

    st.subheader('Data Summary')
    st.write(data1.describe())

    st.subheader('Filter Data')
    columns = data1.columns.tolist()
    selected_columns = st.multiselect('Pilih kolom yang ingin difilter', columns)

    if selected_columns:
        filtered_columns = data1[selected_columns]
    else:
        filtered_columns = pd.DataFrame()  # kosong

    st.write(filtered_columns)

    st.subheader('Plot Data')

    # Ambil batas tanggal dari index
    min_date = data1.index.min().date()
    max_date = data1.index.max().date()

    # Input rentang tanggal dengan batasan tanggal sesuai index
    start_date, end_date = st.date_input(
        "Pilih rentang tanggal plot",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    # Validasi jika user input tuple
    if isinstance(start_date, tuple) or isinstance(start_date, list):
        start_date, end_date = start_date[0], start_date[1]

    # Pilihan kolom untuk diplot
    plot_options = ['[Gunakan kolom terfilter]'] + columns
    y_plot = st.selectbox('Pilih kolom yang akan diplot', plot_options)

    if st.button('Generate Plot'):
        # Filter data sesuai tanggal
        mask = (data1.index.date >= start_date) & (data1.index.date <= end_date)
        data_filtered_by_date = data1.loc[mask]

        if data_filtered_by_date.empty:
            st.warning("Tidak ada data pada rentang tanggal tersebut.")
        else:
            if y_plot == '[Gunakan kolom terfilter]':
                if not selected_columns:
                    st.warning("Anda belum memilih kolom mana pun di filter.")
                else:
                    st.line_chart(data_filtered_by_date[selected_columns])
            else:
                st.line_chart(data_filtered_by_date[[y_plot]])

    st.subheader('Cek Data')
    st.write(data1.tail())

else:
    st.stop()
    
st.title('Preprocessing Data')

# Ambil data yang sudah disimpan sebelumnya
if 'data1' in st.session_state:
    data1 = st.session_state.data1.copy()
else:
    st.error("Silakan upload file terlebih dahulu.")
    st.stop()

# Checkbox agar user bisa memilih apakah ingin menjalankan MICE + Outlier Handling
with st.spinner('Sedang mendeteksi dan mengoreksi outlier...'):
    data1.index = pd.to_datetime(data1.index)
    df_clean = data1.copy()
    fitur = df_clean.columns

    for kolom in fitur:
        Q1 = df_clean[kolom].quantile(0.25)
        Q3 = df_clean[kolom].quantile(0.75)
        IQR = Q3 - Q1
        batas_bawah = Q1 - 1.5 * IQR
        batas_atas = Q3 + 1.5 * IQR

        outlier_mask = (df_clean[kolom] < batas_bawah) | (df_clean[kolom] > batas_atas)
        outlier_data = df_clean.loc[outlier_mask].index

        for timestamp in outlier_data:
            tanggal = timestamp.date()
            minggu_ini_start = tanggal - pd.Timedelta(days=tanggal.weekday())
            minggu_ini_end = minggu_ini_start + pd.Timedelta(days=6)

            data_minggu_ini = df_clean.loc[(df_clean.index.date >= minggu_ini_start) & (df_clean.index.date <= minggu_ini_end), kolom]
            data_minggu_valid = data_minggu_ini[(data_minggu_ini >= batas_bawah) & (data_minggu_ini <= batas_atas)]

            if not data_minggu_valid.empty:
                rata2_mingguan = data_minggu_valid.mean()
            else:
                rata2_mingguan = df_clean.at[timestamp, kolom]

            df_clean.at[timestamp, kolom] = rata2_mingguan

    st.success('Outlier telah diganti dengan rata-rata mingguan.')

with st.spinner('Sedang melakukan imputasi data...'):
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    MiceImputed = df_clean.copy(deep=True)
    mice_imputer = IterativeImputer()
    MiceImputed.iloc[:, :] = mice_imputer.fit_transform(df_clean)

    st.success('Imputasi selesai. Tidak ada data kosong:')
    st.write(MiceImputed.isnull().sum())

    # Tampilkan hasil praproses
    st.subheader("Data Setelah Praproses")
    st.write(MiceImputed)
    st.subheader('Cek Data Kosong')
    st.write(MiceImputed.isnull().sum())
st.session_state.dataf = MiceImputed.copy()

st.title('Visualisasi Data per Fitur')
if 'dataf' in st.session_state:
    dataf = st.session_state.dataf.copy()
else:
    st.error("Data tidak terdeteksi.")
    st.stop()

st.subheader("Visualisasi Data Kasat Mata")
st.markdown("**Perhatikan tren masing-masing kolom. Jika ada yang tampak tidak wajar, lanjutkan ke pemeriksaan detail.**")

# Pilihan kolom untuk ditampilkan otomatis satu per satu
show_all = st.checkbox("Tampilkan semua kolom utama secara otomatis")
if show_all:
    for col in dataf.columns:
        fig, ax = plt.subplots(figsize=(15, 5))  # BUAT figure baru tiap loop
        ax.plot(dataf.index, dataf[col], '.', color='blue')
        ax.set_title(col)
        ax.set_xlabel('Tanggal')
        ax.set_ylabel(col)
        ax.grid(True)
        st.pyplot(fig)

# Pemeriksaan manual oleh pengguna
st.subheader("Cek Detail Kolom Tertentu")
st.markdown("Jika kamu merasa ada data aneh, silakan pilih kolom dan rentang waktu untuk diperiksa lebih dalam.")

# Pilih kolom
kolom_dicek = st.selectbox("Pilih kolom yang ingin diperiksa", dataf.columns)

# Ambil batas tanggal dari index
min_date = dataf.index.min().date()
max_date = dataf.index.max().date()

# Pilih tanggal
tgl_mulai, tgl_selesai = st.date_input(
    "Pilih rentang waktu untuk pengecekan",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Validasi
if isinstance(tgl_mulai, list) or isinstance(tgl_mulai, tuple):
    tgl_mulai, tgl_selesai = tgl_mulai[0], tgl_selesai[1]

# Filter data sesuai rentang waktu
mask = (dataf.index.date >= tgl_mulai) & (dataf.index.date <= tgl_selesai)
data_filtered = dataf.loc[mask]

# Plot hasil pengecekan
st.subheader(f"Hasil Pemeriksaan: {kolom_dicek}")
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(data_filtered.index, data_filtered[kolom_dicek], '.', color='blue')
ax.set_title(kolom_dicek)
ax.axhline(y=0, color='r', linestyle='--', label='Batas bawah (0)')
ax.legend()
ax.grid(True)
st.pyplot(fig)

dataf['RAINFALL 24H MM'] = dataf['RAINFALL 24H MM'].clip(lower=0)
df_bersih = dataf.copy()
st.session_state.df_bersih = df_bersih.copy()

st.title("Data Bersih")
st.write(df_bersih)

st.title("Analisis Partial Autocorrelation (PACF)")

# Pastikan kolom target tersedia
if 'df_bersih' in st.session_state:
    df_bersih = st.session_state.df_bersih.copy()
    if 'RAINFALL 24H MM' not in df_bersih.columns:
        st.error("Data tidak ditemukan.")
        st.stop()
else:
    st.error("Data praproses belum tersedia.")
    st.stop()

# Plot PACF
st.subheader("Partial Autocorrelation Function (PACF)")

fig, ax = plt.subplots(figsize=(8, 5))
plot_pacf(df_bersih['RAINFALL 24H MM'], lags=10, method='ywm', ax=ax)
ax.set_title('Partial Autocorrelation Function (PACF)')
ax.set_xlabel('Lag')
ax.set_ylabel('PACF')
ax.grid(True)
st.pyplot(fig)

st.title('Penambahan Feature dan Lag')

# Input jumlah lag yang ingin ditambahkan, langsung t-1 hingga t-n
lag_count = st.slider("Berapa banyak lag harian yang ingin ditambahkan?", min_value=1, max_value=7, value=4)

# Checkbox untuk memilih fitur mana saja yang ingin ditambahkan
st.subheader("Pilih Fitur Time Series yang akan Ditambahkan:")

pilih_semua = st.checkbox("✅ Pilih Semua Fitur", value=False)
# Checkbox individual, akan otomatis True jika 'Pilih Semua' dicentang
add_month = st.checkbox("Tambahkan Fitur Bulan & Tahun", value=pilih_semua)
add_rainy_season = st.checkbox("Tambahkan Fitur Musim Hujan (is_rainy_season)", value=pilih_semua)
add_week_of_month = st.checkbox("Tambahkan Fitur Minggu ke-berapa", value=pilih_semua)
add_rainfall_change = st.checkbox("Tambahkan Fitur Perubahan Curah Hujan Harian", value=pilih_semua)
add_rolling = st.checkbox("Tambahkan Rolling Mean / Std / Min / Max", value=pilih_semua)
add_temp_humidity = st.checkbox("Tambahkan Interaksi Suhu & Kelembapan", value=pilih_semua)

# Tombol untuk generate fitur
if st.button("Tambahkan Fitur dan Lag"):
    df_feat = df_bersih.copy()

    if add_month:
        df_feat['month'] = df_feat.index.month
        df_feat['year'] = df_feat.index.year

    if add_rainy_season:
        df_feat['is_rainy_season'] = df_feat['month'].apply(lambda x: 1 if x in [11, 12, 1, 2, 3] else 0)

    if add_week_of_month:
        df_feat['week_of_month'] = df_feat.index.isocalendar().week % 4

    if add_rainfall_change:
        df_feat['rainfall_change'] = df_feat['RAINFALL 24H MM'] - df_feat['RAINFALL 24H MM'].shift(1)

    if add_rolling:
        df_feat['rainfall_7d_mean'] = df_feat['RAINFALL 24H MM'].rolling(window=7).mean()
        df_feat['rainfall_30d_mean'] = df_feat['RAINFALL 24H MM'].rolling(window=30).mean()
        df_feat['rainfall_7d_std'] = df_feat['RAINFALL 24H MM'].rolling(window=7).std()
        df_feat['rainfall_7d_ma'] = df_feat['RAINFALL 24H MM'].rolling(window=7).mean()
        df_feat['rainfall_7d_min'] = df_feat['RAINFALL 24H MM'].rolling(window=7).min()
        df_feat['rainfall_7d_max'] = df_feat['RAINFALL 24H MM'].rolling(window=7).max()

    if add_temp_humidity:
        df_feat['temp_humidity_interaction'] = (
            df_feat['TEMPERATURE AVG C'] * df_feat['REL HUMIDITY AVG PC']
        )

    # Tambahkan lag menggunakan shift (t-1, t-2, ..., t-n)
    for i in range(1, lag_count + 1):
        df_feat[f'lag{i}'] = df_feat['RAINFALL 24H MM'].shift(i)

    # Simpan hasil ke session state
    st.session_state.df_feat = df_feat.copy()

    st.success(f"Fitur dan {lag_count} lag (t-1 hingga t-{lag_count}) berhasil ditambahkan!")
    st.write(df_feat.head())

else:
    st.info("Silakan centang fitur, serta pilih jumlah lag harian yang ingin ditambahkan.")

st.write()

st.title("Pembagian Data Training dan Testing")

# Input jumlah fold & panjang test set
# Cek apakah df_feat sudah tersedia
if 'df_feat' in st.session_state:
    df_feat = st.session_state.df_feat.copy()
else:
    st.warning("Silakan klik tombol 'Tambahkan Fitur dan Lag' terlebih dahulu.")
    st.stop()

# Misal kamu punya DataFrame bernama df
st.write("Kolom tersedia:", df_bersih.columns.tolist())

# Pilihan kolom target oleh user
target_col = st.selectbox("Pilih kolom yang akan diprediksi:", df_bersih.columns)

# Pisahkan fitur dan target
X = df_feat.drop(columns=target_col)
y = df_feat[target_col]

# Input rentang tanggal valid
min_date, max_date = df_feat.index.min(), df_feat.index.max()

start_date = st.date_input("Tanggal mulai test set", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.date_input("Tanggal akhir test set", value=max_date, min_value=min_date, max_value=max_date)

# Validasi tanggal
if start_date < end_date:
    # Konversi ke datetime (jika index df adalah datetime)
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Bagi data berdasarkan tanggal
    train = df_feat[(df_feat.index < start_date) | (df_feat.index > end_date)]
    test = df_feat[(df_feat.index >= start_date) & (df_feat.index <= end_date)]

    X_train = train.drop(columns=target_col)
    y_train = train[target_col]
    X_test = test.drop(columns=target_col)
    y_test = test[target_col]

    st.success(f"Data berhasil dibagi. Train: {len(train)} data, Test: {len(test)} data.")
else:
    st.error("Tanggal akhir harus setelah tanggal mulai.")

# Input manual dari pengguna untuk masing-masing parameter
base_score = st.number_input("Base Score", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
n_estimators = st.number_input("Jumlah Estimator (Trees)", min_value=10, max_value=5000, value=1000, step=10)
early_stopping_rounds = st.number_input("Early Stopping Rounds", min_value=1, max_value=200, value=50)
max_depth = st.slider("Max Depth", min_value=1, max_value=50, value=3)
learning_rate = st.number_input("Learning Rate", min_value=0.001, max_value=1.0, value=0.05, step=0.01)
colsample_bytree = st.slider("Colsample By Tree", min_value=0.1, max_value=1.0, value=0.8, step=0.05)
subsample = st.slider("Subsample", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
gamma = st.number_input("Gamma", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

# Inisialisasi model
reg = xgb.XGBRegressor(
    base_score=base_score,
    booster='gbtree',  # FIX
    n_estimators=n_estimators,
    early_stopping_rounds=early_stopping_rounds,
    objective='reg:squarederror',  # FIX
    max_depth=max_depth,
    learning_rate=learning_rate,
    colsample_bytree=colsample_bytree,
    gamma=gamma,
    subsample=subsample,
    random_state=42,
    verbosity=0
)

with st.spinner("Melatih model..."):
    reg.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )

# Prediksi
y_pred = reg.predict(X_test)

# MAAPE Function
def maape(y_true, y_pred):
    return np.mean(np.arctan(np.abs((y_true - y_pred) / (y_true + 1e-10))))

# Evaluasi
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
maape_score = maape(y_test, y_pred)

st.subheader("Evaluasi Model")
st.write(f"**RMSE**: {rmse:.4f}")
st.write(f"**R² Score**: {r2:.4f}")
st.write(f"**MAAPE**: {maape_score:.4f}")

# Tampilkan plot hasil prediksi vs aktual
st.subheader("Plot Prediksi vs Aktual")
fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(y_test.index, y_test.values, label='Aktual', marker='o')
ax.plot(y_test.index, y_pred, label='Prediksi', marker='x')
ax.legend()
ax.set_title("Perbandingan Aktual vs Prediksi")
ax.grid(True)
st.pyplot(fig)

# Buat DataFrame hasil prediksi vs aktual
hasil_prediksi_df = pd.DataFrame({
    'Timestamp': y_test.index,
    'Aktual': y_test.values,
    'Prediksi': y_pred
})
st.write('nilai terbesar selisih')
st.write(max(abs(y_pred - y_test)))

# Tampilkan tabel perbandingan
st.subheader("Tabel Perbandingan Aktual vs Prediksi")
st.dataframe(hasil_prediksi_df)


# # Pastikan model sudah tersedia
# if 'reg' not in st.session_state:
#     st.error("❌ Model belum tersedia. Silakan retrain terlebih dahulu.")
#     st.stop()

# # Load model
# regl = st.session_state.reg

# if 'df' in st.session_state:
#     df = st.session_state.df.copy()
# else:
#     st.error("Data tidak tersedia.")
#     st.stop()

# # 1. Tampilkan periode waktu dari DataFrame
# start_date = df.index.min().date()
# end_date = df.index.max().date()

# st.markdown(f"📆 **Periode Data:** {start_date} sampai {end_date}")

# # 2. Input tanggal oleh pengguna
# st.subheader("Pilih Rentang Tanggal untuk Data Testing")
# selected_start_date, selected_end_date = st.date_input(
#     "Pilih rentang waktu untuk prediksi",
#     [start_date, end_date],
#     min_value=start_date,
#     max_value=end_date
# )

# # Validasi input
# if isinstance(selected_start_date, list) or isinstance(selected_start_date, tuple):
#     selected_start_date, selected_end_date = selected_start_date[0], selected_end_date[1]

# # 3. Filter data berdasarkan rentang tanggal
# test_mask = (df.index.date >= selected_start_date) & (df.index.date <= selected_end_date)
# test = df.loc[test_mask]

# # 4. Tampilkan informasi
# st.info(f"Jumlah baris data untuk prediksi: {test.shape[0]}")
# st.write(f"Periode: {test.index.min().date()} hingga {test.index.max().date()}")

# # Prediksi pada data testing
# X_test_new = test[FEATURES]
# y_test_actual = test[TARGET]
# y_pred_new = reg_final.predict(X_test_new)

# # Gabungkan hasil ke dalam DataFrame
# hasil_prediksi = pd.DataFrame({
#     'Timestamp': test.index,
#     'Prediksi': y_pred_new,
#     'Aktual': y_test_actual
# })

# # Tampilkan tabel
# st.subheader("Hasil Prediksi vs Aktual")
# st.write(hasil_prediksi)

# # Tampilkan dimensi
# rows, cols = hasil_prediksi.shape
# st.write(f"📊 Dataframe hasil: {rows} baris dan {cols} kolom")

# # Plot
# st.subheader("Visualisasi Prediksi vs Aktual")
# fig, ax = plt.subplots(figsize=(15, 5))
# ax.plot(hasil_prediksi['Timestamp'], hasil_prediksi['Aktual'], label='Aktual', color='blue')
# ax.plot(hasil_prediksi['Timestamp'], hasil_prediksi['Prediksi'], label='Prediksi', color='orange')
# ax.set_xlabel('Tanggal')
# ax.set_ylabel('Curah Hujan (mm)')
# ax.set_title('Prediksi vs Aktual Curah Hujan')
# ax.legend()
# ax.grid(True)
# st.pyplot(fig)

# st.title('Hasil Prediksi Curah Hujan')
# y_pred_mod = np.where(y_pred_new < 0, 0, y_pred_new)
# hasil_prediksi_mod = pd.DataFrame({
#     'Timestamp': test.index,
#     'Prediksi': y_pred_mod,
#     'Aktual': y_test_actual})
# st.write(hasil_prediksi_mod)
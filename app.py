import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import glob

# Konfigurasi Halaman Web
st.set_page_config(page_title="Biznify AI - Pusat Visualisasi Bab 3", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size:28px; font-weight:bold; color:#1E3A8A; margin-bottom:20px; }
    .section-title { font-size:22px; font-weight:bold; color:#2563EB; margin-top:30px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">🎯 Pusat Visualisasi & Manajemen Aset Gambar - Bab III</div>', unsafe_allow_html=True)

# ==========================================
# DEFINISI DIREKTORI UTAMA (RIIL DATA)
# ==========================================
PATH_PREPROCESSED = r'E:\Semester 7\TA\code\preprocesing\standardized_data'
PATH_CLUSTERING   = r'E:\Semester 7\TA\code\clustering'
PATH_EVALUATION   = r'E:\Semester 7\TA\code\evaluation'

# Daftar 26 Ticker yang Lolos Seleksi Irisan Sektoral
TICKERS_BASELINE = ["ADRO", "AKRA", "APEX", "ARII", "BULL", "BUMI", "DEWA", "DOID", "ELSA", "ENRG", 
                    "FIRE", "HRUM", "INDY", "ITMA", "ITMG", "KKGI", "MBSS", "MEDC", "PGAS", "PTBA", 
                    "RUIS", "SOCI", "TCPI", "TEBE", "TPMA", "WINS"]

# ==========================================
# FUNCTION DOWNLOAD HIGH RES (300 DPI)
# ==========================================
def hitung_dan_export_300dpi(fig_plt, nama_file="plot.png"):
    buf = io.BytesIO()
    fig_plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    st.download_button(
        label=f"📥 Download {nama_file} (300 DPI)",
        data=buf,
        file_name=nama_file,
        mime="image/png"
    )

# ==========================================
# KONTROL PANEL (SIDEBAR)
# ==========================================
st.sidebar.header("⚙️ Kontrol Panel Data")
menu_select = st.sidebar.radio(
    "Pilih Sub-bab Analisis:",
    ["A. Standardisasi & Pivot Points", 
     "B. Implementasi K-Means & Centroid", 
     "C. Pemetaan Zona Rentang Dinamis", 
     "D. Evaluasi Performa Model"]
)

ticker_aktif = st.sidebar.selectbox("Pilih Emiten:", TICKERS_BASELINE)

# ==========================================
# SUB-BAB A: STANDARDIZATION & PIVOT POINTS
# ==========================================
if menu_select == "A. Standardisasi & Pivot Points":
    st.markdown('<div class="section-title">A. Analisis Data, Pivot Points, dan Standardisasi Z-Score</div>', unsafe_allow_html=True)
    
    # Cari file emiten aktif di folder preprocessing
    file_pattern = os.path.join(PATH_PREPROCESSED, f"{ticker_aktif}*.csv")
    matched_files = glob.glob(file_pattern)
    
    if matched_files:
        df_raw = pd.read_csv(matched_files[0])
        # Pastikan kolom Tanggal bertipe datetime untuk grafik
        df_raw['Tanggal'] = pd.to_datetime(df_raw['Tanggal'])
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader(f"📋 Sampel Data Terstandardisasi ({ticker_aktif})")
            st.dataframe(df_raw[['Tanggal', 'Level', 'Type', 'Z_Score']].head(15), use_container_width=True, hide_index=True)
            
        with col2:
            st.subheader("📈 Grafik Deteksi Pivot & Transformasi Z-Score")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            
            # Subplot 1: Harga Riil (Level) & Titik Pembalikan
            ax1.plot(df_raw["Tanggal"], df_raw["Level"], label="Harga Penutupan Harian", color="blue", alpha=0.6)
            
            df_high = df_raw[df_raw['Type'] == 'Resistance']
            df_low  = df_raw[df_raw['Type'] == 'Support']
            
            ax1.scatter(df_high["Tanggal"], df_high["Level"], color="red", s=40, label="Pivot High (S)", zorder=5)
            ax1.scatter(df_low["Tanggal"], df_low["Level"], color="green", s=40, label="Pivot Low (R)", zorder=5)
            ax1.set_title(f"Ekstraksi Titik Balik (Pivot Points) Riil - {ticker_aktif}")
            ax1.legend()
            ax1.grid(True, linestyle="--", alpha=0.5)
            
            # Subplot 2: Distribusi Nilai Skala Z-Score
            ax2.plot(df_raw["Tanggal"], df_raw["Z_Score"], label="Nilai Hasil Z-Score", color="purple", linestyle="-", alpha=0.7)
            ax2.axhline(0, color="black", linestyle="-", alpha=0.4)
            ax2.set_title("Hasil Transformasi Penyelarasan Skala (Z-Score)")
            ax2.legend()
            ax2.grid(True, linestyle="--", alpha=0.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            hitung_dan_export_300dpi(fig, nama_file=f"SubbabA_Pivot_ZScore_{ticker_aktif}.png")
    else:
        st.error(f"❌ Berkas CSV untuk emiten {ticker_aktif} tidak ditemukan di direktori: {PATH_PREPROCESSED}")

# ==========================================
# SUB-BAB B: IMPLEMENTASI K-MEANS
# ==========================================
elif menu_select == "B. Implementasi K-Means & Centroid":
    st.markdown('<div class="section-title">B. Implementasi Klasterisasi K-Means</div>', unsafe_allow_html=True)
    
    k_select = st.sidebar.slider("Pilih Nilai K untuk melihat Posisi Centroid:", 2, 10, 4)
    
    # Membaca data detail clustering hasil K-Means per nilai K yang dipilih
    file_pattern = os.path.join(PATH_CLUSTERING, f"k{k_select}", "detail", f"{ticker_aktif}*.csv")
    matched_files = glob.glob(file_pattern)
    
    if matched_files:
        df_cluster = pd.read_csv(matched_files[0])
        
        st.subheader(f"📍 Distribusi Data Hasil Klasterisasi K={k_select} pada Ruang Fitur Z-Score ({ticker_aktif})")
        
        fig_c, ax_c = plt.subplots(figsize=(10, 4))
        # Plot sebaran data asli yang sudah di-cluster oleh program kamu
        sns.stripplot(data=df_cluster, x="Z_Score", hue="Cluster", palette="Set1", ax=ax_c, size=7, alpha=0.7)
        
        ax_c.set_xlabel("Skala Nilai Fitur Z-Score")
        ax_c.set_title(f"Sebaran Keanggotaan Objek Data terhadap Klaster K={k_select}")
        ax_c.grid(True, linestyle=":", alpha=0.5)
        
        st.pyplot(fig_c)
        hitung_dan_export_300dpi(fig_c, nama_file=f"SubbabB_Sebaran_Klaster_K{k_select}_{ticker_aktif}.png")
    else:
        st.error(f"❌ Berkas detail klaster K={k_select} untuk {ticker_aktif} tidak ditemukan di: {PATH_CLUSTERING}")

# ==========================================
# SUB-BAB C: PEMETAAN ZONA RENTANG DINAMIS
# ==========================================
elif menu_select == "C. Pemetaan Zona Rentang Dinamis":
    st.markdown('<div class="section-title">C. Pemetaan Zona Rentang Dinamis (Batas Spasial Internal)</div>', unsafe_allow_html=True)
    
    k_select = st.sidebar.slider("Pilih Nilai K untuk Visualisasi Zona:", 2, 10, 4)
    
    # Membaca berkas summary_zona hasil perhitungan rumus Min/Max kamu
    file_pattern = os.path.join(PATH_CLUSTERING, f"k{k_select}", "summary_zona", f"{ticker_aktif}*.csv")
    matched_files = glob.glob(file_pattern)
    
    # Membaca data harga dasar sebagai background grafik
    file_raw_pattern = os.path.join(PATH_PREPROCESSED, f"{ticker_aktif}*.csv")
    matched_raw = glob.glob(file_raw_pattern)
    
    if matched_files and matched_raw:
        df_zona = pd.read_csv(matched_files[0])
        df_raw  = pd.read_csv(matched_raw[0])
        df_raw['Tanggal'] = pd.to_datetime(df_raw['Tanggal'])
        
        st.subheader(f"🛡️ Plot Batas Area Ruang Spasial Dinamis (Min/Max Area) - K={k_select}")
        
        fig_z, ax_z = plt.subplots(figsize=(12, 6))
        ax_z.plot(df_raw["Tanggal"], df_raw["Level"], color="black", linewidth=1.2, label="Harga Saham Aktual", alpha=0.8)
        
        # Gambar arsiran zona Support & Resistance secara otomatis dari baris summary_zona .csv kamu
        for index, row in df_zona.iterrows():
            if row['Type'] == 'Resistance':
                ax_z.axhspan(row['Min'], row['Max'], color='red', alpha=0.15, 
                             label="Zona Resistance" if "Zona Resistance" not in ax_z.get_legend_handles_labels()[1] else "")
                ax_z.axhline(row['Centroid_Price'], color='red', linestyle='--', alpha=0.5)
            elif row['Type'] == 'Support':
                ax_z.axhspan(row['Min'], row['Max'], color='green', alpha=0.15, 
                             label="Zona Support" if "Zona Support" not in ax_z.get_legend_handles_labels()[1] else "")
                ax_z.axhline(row['Centroid_Price'], color='green', linestyle='--', alpha=0.5)
                
        ax_z.set_title(f"Visualisasi Batas Spasial Dinamis pada Tren Pergerakan Saham {ticker_aktif} (K={k_select})")
        ax_z.set_xlabel("Periode Runut Waktu")
        ax_z.set_ylabel("Nominal Harga Saham Riil (Rupiah)")
        ax_z.legend(loc="upper left")
        ax_z.grid(True, linestyle="--", alpha=0.3)
        
        st.pyplot(fig_z)
        hitung_dan_export_300dpi(fig_z, nama_file=f"SubbabC_Zona_BatasDinamis_K{k_select}_{ticker_aktif}.png")
    else:
        st.error("❌ Kegagalan memuat file summary_zona atau file data dasar.")

# ==========================================
# SUB-BAB D: EVALUASI PERFORMA MODEL
# ==========================================
elif menu_select == "D. Evaluasi Performa Model":
    st.markdown('<div class="section-title">D. Evaluasi Performa Model (Hasil Kinerja Riil Trading Rules)</div>', unsafe_allow_html=True)
    
    # Kumpulkan semua file summary_metrics_k2.csv sampai k10 otomatis dari folder evaluation
    all_metrics = []
    for k in range(2, 11):
        metric_file = os.path.join(PATH_EVALUATION, f"summary_metrics_k{k}.csv")
        if os.path.exists(metric_file):
            df_m = pd.read_csv(metric_file)
            df_m['K_Value'] = f"K={k}"
            all_metrics.append(df_m)
            
    if all_metrics:
        df_all_perf = pd.concat(all_metrics)
        
        # Fitur Pilihan Mode Tampilan di dalam halaman
        mode_tampilan = st.radio("Pilih Mode Analisis Evaluasi:", ["Rata-rata Sektoral (26 Emiten)", "Per Emiten Individu"], horizontal=True)
        
        mul = 100 if df_all_perf['Accuracy'].max() <= 1.0 else 1
        
        if mode_tampilan == "Rata-rata Sektoral (26 Emiten)":
            st.subheader("📊 Rekapitulasi Rata-Rata Final Performa Sektoral Energi (K2 - K10)")
            
            # Hitung rata-rata grouping berdasarkan K_Value (Persis seperti check_best_k.py)
            df_sektoral = df_all_perf.groupby('K_Value')[['Accuracy', 'Precision', 'Recall', 'F1_Score']].mean().reset_index()
            
            # Urutkan berdasarkan index K agar rapi dari K=2 sampai K=10
            df_sektoral['sort_key'] = df_sektoral['K_Value'].str.extract('(\d+)').astype(int)
            df_sektoral = df_sektoral.sort_values('sort_key').drop(columns=['sort_key'])
            
            st.dataframe(df_sektoral, use_container_width=True, hide_index=True)
            
            df_plot = df_sektoral
            judul_grafik = "Grafik Komparasi Rata-Rata Performa Sektoral Energi"
            nama_file_out = "SubbabD_RataRata_Sektoral_Energi.png"
            
        else:
            st.subheader(f"📊 Tabel Perbandingan Nilai K2 s/d K10 - {ticker_aktif}")
            
            # Filter berdasarkan emiten aktif pilihan user
            df_ticker_perf = df_all_perf[df_all_perf['Emiten'] == ticker_aktif].copy()
            df_ticker_perf['sort_key'] = df_ticker_perf['K_Value'].str.extract('(\d+)').astype(int)
            df_ticker_perf = df_ticker_perf.sort_values('sort_key').drop(columns=['sort_key'])
            
            st.dataframe(df_ticker_perf[['K_Value', 'Accuracy', 'Precision', 'Recall', 'F1_Score']], use_container_width=True, hide_index=True)
            
            df_plot = df_ticker_perf
            judul_grafik = f"Grafik Komparasi Metrik Confusion Matrix ({ticker_aktif})"
            nama_file_out = f"SubbabD_Komparasi_Akurasi_{ticker_aktif}.png"
            
        # Gambar Grafik Batang Komparasi Berdasarkan Mode yang Dipilih
        fig_b, ax_b = plt.subplots(figsize=(10, 5))
        x_indices = np.arange(len(df_plot))
        width = 0.25
        
        ax_b.bar(x_indices - width, df_plot["Accuracy"] * mul, width, label="Accuracy", color="#1E3A8A")
        ax_b.bar(x_indices, df_plot["Precision"] * mul, width, label="Precision", color="#2563EB")
        ax_b.bar(x_indices + width, df_plot["F1_Score"] * mul, width, label="F1-Score", color="#3B82F6")
        
        ax_b.set_xticks(x_indices)
        ax_b.set_xticklabels(df_plot["K_Value"])
        ax_b.set_ylabel("Kinerja Prediksi Sinyal (%)")
        ax_b.set_title(judul_grafik)
        ax_b.legend()
        ax_b.grid(True, linestyle=":", alpha=0.5)
        
        st.pyplot(fig_b)
        hitung_dan_export_300dpi(fig_b, nama_file=nama_file_out)
        
    else:
        st.error(f"❌ Berkas metrik pengujian `summary_metrics_k*.csv` tidak ditemukan di folder: {PATH_EVALUATION}")
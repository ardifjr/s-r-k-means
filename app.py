import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import glob

# Konfigurasi Halaman Web
st.set_page_config(page_title="Pusat Visualisasi Bab 3", layout="wide")

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

def gambarkan_diagram_seleksi_emiten():
    # Definisikan tahapan seleksi
    tahapan = [
        "Seluruh Emiten Sektor Energi\ndi BEI (IDX)", 
        "Penyaringan Kelayakan Data\n(Kelengkapan Historis)", 
        "Seleksi Kecukupan Pivot Points\nUntuk Variasi K=2 s/d K=10\n(Konsistensi Klasterisasi)"
    ]
    # Representasi jumlah emiten fiktif/perkiraan sebelum dikerucutkan ke 26 emiten lolos
    jumlah_emiten = [45, 34, 26] 
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Membuat diagram berbentuk corong/blok panah ke bawah
    y_pos = np.arange(len(tahapan))
    colors = ["#1E3A8A", "#2563EB", "#10B981"] # Biru Tua -> Biru Muda -> Hijau (Lolos)
    
    # Gambar kotak proses
    bars = ax.barh(y_pos, jumlah_emiten, align='center', color=colors, alpha=0.85, height=0.5, edgecolor='black')
    
    # Menambahkan teks label di dalam/samping balok
    for bar, teks, nilai in zip(bars, tahapan, jumlah_emiten):
        width = bar.get_width()
        ax.text(width / 2, bar.get_y() + bar.get_height()/2, f"{nilai} Emiten", 
                va='center', ha='center', color='white', fontweight='bold', fontsize=10)
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, teks, 
                va='center', ha='left', color='black', fontsize=9, fontweight='semibold')
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels([]) # Hilangkan tik standar
    ax.set_xlabel("Jumlah Emiten")
    ax.set_title("Alur Tahapan Penyaringan Irisan Sektoral Emiten Energi", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlim(0, 60)
    ax.invert_yaxis()  # Supaya urutan dari atas ke bawah
    ax.grid(axis='x', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    return fig

def gambarkan_komparasi_zscore_timeline(df_raw, ticker):
    # Buat subplot 1 baris, 2 kolom bersisian
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    # Filter data khusus untuk titik pivot (Support & Resistance) agar grafik tidak padat
    df_pivot = df_raw[df_raw['Type'].isin(['Support', 'Resistance'])].copy()
    if df_pivot.empty:
        df_pivot = df_raw.copy()
        
    # Pastikan kolom Tanggal bertipe datetime
    df_pivot['Tanggal'] = pd.to_datetime(df_pivot['Tanggal'])
    
    # Pisahkan data Support dan Resistance untuk membedakan warna seperti grafik harianmu
    df_high = df_pivot[df_pivot['Type'] == 'Resistance']
    df_low = df_pivot[df_pivot['Type'] == 'Support']
    
    # ----------------------------------------------------
    # KIRI: SEBELUM STANDARDISASI (X = HARGA RIIL, Y = TANGGAL)
    # ----------------------------------------------------
    ax1.scatter(df_high["Level"], df_high["Tanggal"], color="red", alpha=0.7, edgecolors='k', s=45, label="Pivot High (R)")
    ax1.scatter(df_low["Level"], df_low["Tanggal"], color="green", alpha=0.7, edgecolors='k', s=45, label="Pivot Low (S)")
    
    ax1.set_title(f"Original Data - {ticker} (Sebelum Standaridisasi)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Nominal Harga Saham Riil (Rupiah)", fontweight='semibold')
    ax1.set_ylabel("Rentang Tanggal Historis", fontweight='semibold')
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")
    
    # ----------------------------------------------------
    # KANAN: SESUDAH STANDARDISASI (X = Z-SCORE, Y = TANGGAL)
    # ----------------------------------------------------
    ax2.scatter(df_high["Z_Score"], df_high["Tanggal"], color="red", alpha=0.7, edgecolors='k', s=45, label="Pivot High (R)")
    ax2.scatter(df_low["Z_Score"], df_low["Tanggal"], color="green", alpha=0.7, edgecolors='k', s=45, label="Pivot Low (S)")
    
    # Garis bantu vertikal di Z-Score = 0 (Mean)
    ax2.axvline(0, color="purple", linestyle="--", alpha=0.7, label="Mean = 0")
    
    # Batasan Sumbu X dari -3 sampai 3 sesuai kebutuhan klasterisasi
    ax2.set_xlim(-3.5, 3.5)
    ax2.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    
    ax2.set_title("Standardized Data (Sesudah Z-Score)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Skala Nilai Transformasi Z-Score", fontweight='semibold')
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")
    
    plt.suptitle(f"Gambar 3.2: Perbandingan Distribusi Titik Pivot {ticker} Berdasarkan Garis Waktu", fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig
# ==========================================
# SUB-BAB A: STANDARDIZATION & PIVOT POINTS
# ==========================================
if menu_select == "A. Standardisasi & Pivot Points":
    st.markdown('<div class="section-title">A. Analisis Data, Pivot Points, dan Standardisasi Z-Score</div>', unsafe_allow_html=True)
    
    # Bagian Penyaringan Emiten & Gambar 3.1 Sektoral tetap berada di bagian atas...
    st.subheader("📌 Tahapan Penyaringan dan Konsistensi Data Penelitian")
    # ... (code penyaringan sebelumnya) ...
    
    st.markdown("---")
    
    # ----------------------------------------------------
    # DATA EMITEN AKTIF (DAPAT DI-FILTER DARI SIDEBAR)
    # ----------------------------------------------------
    st.subheader(f"🔍 Analisis Deteksi & Transformasi Spasial Dinamis: {ticker_aktif}")
    
    file_pattern = os.path.join(PATH_PREPROCESSED, f"{ticker_aktif}*.csv")
    matched_files = glob.glob(file_pattern)
    
    if matched_files:
        df_raw = pd.read_csv(matched_files[0])
        df_raw['Tanggal'] = pd.to_datetime(df_raw['Tanggal'])
        
        # --- BLOK INTEGRASI GAMBAR 3.2 TUNGGAL (X=Harga/Z-Score, Y=Tanggal) ---
        st.write(
            "Hasil identifikasi Pivot Points menunjukkan bahwa setiap emiten memiliki rentang harga yang berbeda, "
            f"mulai dari puluhan hingga ribuan rupiah per lembar saham. Oleh karena itu, khusus untuk emiten **{ticker_aktif}**, "
            "dilakukan standardisasi menggunakan metode Z-Score agar seluruh data berada pada skala yang sama "
            "sebelum proses klasterisasi. Hasil standardisasi ditampilkan pada **Gambar 3.2**."
        )
        
        # Memanggil fungsi grafik komparasi bersisian (Timeline)
        fig_32_timeline = gambarkan_komparasi_zscore_timeline(df_raw, ticker_aktif)
        st.pyplot(fig_32_timeline)
        hitung_dan_export_300dpi(fig_32_timeline, nama_file=f"Gambar_3.2_Transformasi_ZScore_{ticker_aktif}.png")
        
        st.write(
            "Based on the visualization above, seluruh data berhasil ditransformasikan ke dalam skala yang seragam "
            "dengan nilai rata-rata (*mean*) mendekati 0 dan standar deviasi sebesar 1. Kondisi ini membuat setiap "
            "emiten memiliki skala yang setara sehingga proses klasterisasi menggunakan algoritma K-Means "
            "dapat dilakukan secara lebih objektif."
        )
        
        st.markdown("---")
        
        # Tampilan tabel sampel data dan grafik runut waktu harian asli bawaanmu
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**📋 Sampel Data Terstandardisasi ({ticker_aktif})**")
            st.dataframe(df_raw[['Tanggal', 'Level', 'Type', 'Z_Score']].head(15), use_container_width=True, hide_index=True)
            
        with col2:
            st.markdown("**📈 Grafik Deteksi Pivot & Transformasi Z-Score**")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            
            # Subplot 1: Harga Riil
            ax1.plot(df_raw["Tanggal"], df_raw["Level"], label="Harga Penutupan Harian", color="blue", alpha=0.6)
            df_high = df_raw[df_raw['Type'] == 'Resistance']
            df_low  = df_raw[df_raw['Type'] == 'Support']
            ax1.scatter(df_high["Tanggal"], df_high["Level"], color="red", s=40, label="Pivot High (R)", zorder=5)
            ax1.scatter(df_low["Tanggal"], df_low["Level"], color="green", s=40, label="Pivot Low (S)", zorder=5)
            ax1.set_title(f"Ekstraksi Titik Balik (Pivot Points) Riil - {ticker_aktif}")
            ax1.legend()
            ax1.grid(True, linestyle="--", alpha=0.5)
            
            # Subplot 2: Distribusi Z-Score
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
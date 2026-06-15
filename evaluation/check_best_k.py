import pandas as pd
import os
import glob

# ================= CONFIGURATION =================
path_evaluation = r'E:\Semester 7\TA\code\evaluation'

# Cari semua file summary_metrics_k*.csv di folder evaluation
csv_files = glob.glob(os.path.join(path_evaluation, "summary_metrics_k*.csv"))

if len(csv_files) == 0:
    print("❌ ERROR: Tidak ditemukan file 'summary_metrics_k*.csv' di folder evaluation.")
    print("Pastikan kamu sudah menjalankan script 'backtesting_loop.py' terlebih dahulu.")
    exit()

rekap_sektoral = []

# ================= LOOP READING DATA =================
for file_path in csv_files:
    file_name = os.path.basename(file_path)
    
    # Ekstrak nilai K dari nama file (misal: summary_metrics_k3.csv -> K = 3)
    try:
        k_val = int(file_name.split('_k')[-1].split('.csv')[0])
    except ValueError:
        continue
        
    df_metrics = pd.read_csv(file_path)
    
    if df_metrics.empty:
        continue
        
    # Hitung rata-rata sektoral untuk 40 emiten pada nilai K ini
    avg_accuracy = df_metrics['Accuracy'].mean()
    avg_precision = df_metrics['Precision'].mean()
    avg_recall = df_metrics['Recall'].mean()
    avg_f1 = df_metrics['F1_Score'].mean()
    total_emiten = len(df_metrics)
    
    rekap_sektoral.append({
        'K_Value': f"K = {k_val}",
        'Int_K': k_val,
        'Avg_Accuracy': avg_accuracy,
        'Avg_Precision': avg_precision,
        'Avg_Recall': avg_recall,
        'Avg_F1_Score': avg_f1,
        'Jumlah_Emiten': total_emiten
    })

# Ubah ke DataFrame dan urutkan berdasarkan nilai K terkecil ke terbesar untuk tampilan tabel
df_summary = pd.DataFrame(rekap_sektoral).sort_values('Int_K').reset_index(drop=True)

# ================= OUTPUT RESULTS =================
print("\n" + "="*85)
print("📊 REKAPITULASI RATA-RATA FINAL PERFORMA SEKTORAL ENERGI (K2 - K10)")
print("="*85)

# Cetak tabel ringkasan
print(f"{'Skenario':<10} | {'Emiten':<6} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-"*85)
for _, row in df_summary.iterrows():
    print(f"{row['K_Value']:<10} | {row['Jumlah_Emiten']:<6} | {row['Avg_Accuracy']:10.4f} | {row['Avg_Precision']:10.4f} | {row['Avg_Recall']:10.4f} | {row['Avg_F1_Score']:10.4f}")
print("-"*85)

# --- PENENTUAN MODEL TERBAIK BERDASARKAN F1-SCORE ---
# Cari baris yang memiliki Avg_F1_Score paling tinggi
best_model = df_summary.loc[df_summary['Avg_F1_Score'].idxmax()]

print("\n🏆 KESIMPULAN OPTIMALISASI NILAI K UNTUK BAB 5:")
print("-" * 55)
print(f" ✨ Model Klaster Paling Optimal : {best_model['K_Value']}")
print(f" ✨ Rata-Rata F1-Score Sektoral : {best_model['Avg_F1_Score']:.4f} ({round(best_model['Avg_F1_Score']*100, 2)}%)")
print(f" ✨ Rata-Rata Akurasi Sektoral  : {best_model['Avg_Accuracy']:.4f} ({round(best_model['Avg_Accuracy']*100, 2)}%)")
print("-" * 55)
print(f"💡 REKOMENDASI: Gunakan konfigurasi {best_model['K_Value']} sebagai parameter final")
print("   pada sistem Support & Resistance Algoritma K-Means kamu!")
print("="*85 + "\n")
import pandas as pd
import numpy as np
import os
import glob
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

warnings.filterwarnings("ignore", category=UserWarning)

# ================= CONFIGURATION =================
K_VALUE = 4  # Ganti menjadi 3 atau 4 untuk file eval_k3.py dan eval_k4.py
path_testing = r'E:\Semester 7\TA\code\testing'
path_scoring_zona = fr'E:\Semester 7\TA\code\clustering\scoring\k{K_VALUE}'
path_output_eval = fr'E:\Semester 7\TA\code\evaluation\results_k{K_VALUE}'

if not os.path.exists(path_output_eval): 
    os.makedirs(path_output_eval)

zona_files = glob.glob(os.path.join(path_scoring_zona, "*.csv"))

def evaluate_testing_data(df_test, df_zona):
    y_true = []  # 1: Reversal Riil, 0: Breakout/Tidak ada pantulan riil
    y_pred = []  # 1: Model Prediksi Reversal (Zona Aktif), 0: Model Prediksi Diam (No Zona)
    
    total_rows = len(df_test)
    
    for i in range(total_rows - 3):
        row = df_test.iloc[i]
        next_days = df_test.iloc[i+1 : i+4]
        
        signal_triggered = False
        
        for _, zone in df_zona.iterrows():
            # --- A. EVALUASI ZONA SUPPORT ---
            if zone['Type'] == 'Support':
                if zone['Min'] <= row['Low'] <= zone['Max']:
                    y_pred.append(1)
                    signal_triggered = True
                    jebol = any(next_days['Low'] < zone['Min'])
                    y_true.append(0 if jebol else 1)
                    break
                        
            # --- B. EVALUASI ZONA RESISTANCE ---
            elif zone['Type'] == 'Resistance':
                if zone['Min'] <= row['High'] <= zone['Max']:
                    y_pred.append(1)
                    signal_triggered = True
                    jebol = any(next_days['High'] > zone['Max'])
                    y_true.append(0 if jebol else 1)
                    break
        
        # --- C. DETEKSI DI LUAR JANGKAUAN ZONA ---
        if not signal_triggered:
            if i >= 2 and i < total_rows - 2:
                is_local_bottom = row['Low'] == df_test.iloc[i-2:i+3]['Low'].min()
                is_local_top = row['High'] == df_test.iloc[i-2:i+3]['High'].max()
                
                if is_local_bottom or is_local_top:
                    y_pred.append(0)  
                    y_true.append(1)  # Menjadi FN (Pantulan Terlewat)
                else:
                    y_pred.append(0)  
                    y_true.append(0)  # Menjadi TN (Abaikan Valid)

    return y_true, y_pred

# ================= EXECUTION =================
print(f"=== Menghitung Evaluasi Performa Model untuk K={K_VALUE} ===\n")
all_metrics = []

for file in zona_files:
    file_name = os.path.basename(file)
    ticker = file_name.split('_')[0]
    
    test_file_name = f"{ticker}_2023-12-26_to_2025-12-26.csv"
    test_file_path = os.path.join(path_testing, test_file_name)
    
    if not os.path.exists(test_file_path): continue
        
    df_zona = pd.read_csv(file)
    df_test = pd.read_csv(test_file_path)
    
    y_true, y_pred = evaluate_testing_data(df_test, df_zona)
    
    if len(y_pred) == 0: continue
        
    # --- HITUNG MANUAL KOMPONEN INTERAKTIF ---
    pantulan_tepat = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1) # TP
    zona_jebol = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)     # FP
    pantulan_terlewat = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0) # FN
    abaikan_valid = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)     # TN

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    all_metrics.append({
        'Emiten': ticker, 
        'Pantulan_Tepat': pantulan_tepat, 
        'Zona_Jebol': zona_jebol, 
        'Pantulan_Terlewat': pantulan_terlewat, 
        'Abaikan_Valid': abaikan_valid,
        'Accuracy': round(accuracy, 4), 
        'Precision': round(precision, 4),
        'Recall': round(recall, 4), 
        'F1_Score': round(f1, 4)
    })

# --- MONITORING OUTPUT METRIK ---
if len(all_metrics) > 0:
    df_results = pd.DataFrame(all_metrics)
    summary_save_path = os.path.join(path_output_eval, "summary_metrics.csv")
    df_results.to_csv(summary_save_path, index=False)
    
    # 1. Hitung Akumulasi Total Sektoral untuk Matriks Gambar Gambar
    final_tp = df_results['Pantulan_Tepat'].sum()
    final_fp = df_results['Zona_Jebol'].sum()
    final_fn = df_results['Pantulan_Terlewat'].sum()
    final_tn = df_results['Abaikan_Valid'].sum()
    
    cm_matrix = np.array([[final_tp, final_fp],
                          [final_fn, final_tn]])
    
    # 2. Proses Plot Gambar Matriks menggunakan Nama Parameter Baru
    plt.figure(figsize=(7, 6))
    labels = ['Reversal (Zona Aktif)', 'Breakout / No Sinyal']
    
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, cbar=True,
                annot_kws={"size": 13, "weight": "bold"})
    
    plt.title(f'Confusion Matrix Sektoral Energi (K={K_VALUE})', fontsize=14, pad=15)
    plt.ylabel('Realita Pergerakan Pasar', fontsize=12)
    plt.xlabel('Prediksi Model Zona K-Means', fontsize=12)
    plt.tight_layout()
    
    graph_save_path = os.path.join(path_output_eval, f"confusion_matrix_k{K_VALUE}.png")
    plt.savefig(graph_save_path, dpi=300)
    plt.close()
    
    # Cetak Tabel Rapi di Terminal
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    
    print(f"📋 TABEL EVALUASI PERFORMA MODEL PER EMITEN (K={K_VALUE}):")
    print("=" * 115)
    print(df_results.to_string(index=False))
    print("=" * 115)
    
    print(f"\n📊 RINGKASAN AKUMULASI MATRIKS SEKTORAL:")
    print("-" * 55)
    print(f" Total Pantulan Tepat Target (TP) : {final_tp} kali")
    print(f" Total Zona Jebol/Breakout (FP)   : {final_fp} kali")
    print(f" Total Pantulan Terlewat (FN)     : {final_fn} kali")
    print(f" Total Abaikan Valid (TN)         : {final_tn} kali")
    print("-" * 55)
    
    print(f"\n🎯 RATA-RATA FINAL SEKTOR ENERGI (K={K_VALUE}):")
    print("-" * 40)
    print(df_results[['Accuracy', 'Precision', 'Recall', 'F1_Score']].mean().to_string())
    print("-" * 40)
    print(f"✨ Hasil pembaruan tabel sukses disimpan di: {summary_save_path}")
else:
    print("\n❌ ERROR: Tidak ada satu pun emiten yang berhasil dievaluasi.")
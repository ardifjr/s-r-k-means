import pandas as pd
import numpy as np
import os
import glob
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ================= CONFIGURATION =================
K_VALUE = 2
path_testing = r'E:\Semester 7\TA\code\testing'
path_scoring_zona = fr'E:\Semester 7\TA\code\clustering\scoring\k{K_VALUE}'
path_output_eval = fr'E:\Semester 7\TA\code\evaluation\results_k{K_VALUE}'

if not os.path.exists(path_output_eval): os.makedirs(path_output_eval)

zona_files = glob.glob(os.path.join(path_scoring_zona, "*.csv"))

def evaluate_testing_data(df_test, df_zona):
    y_true = []  # Realita pasar (1: Pantul/Valid, 0: Jebol/Breakout)
    y_pred = []  # Prediksi model (Selalu 1 jika harga menyentuh zona)
    
    # Ambil zona dengan Ranking 1 (Zona Terkuat hasil scoring)
    s1_zone = df_zona[(df_zona['Type'] == 'Support') & (df_zona['Ranking'] == 1)].iloc[0]
    r1_zone = df_zona[(df_zona['Type'] == 'Resistance') & (df_zona['Ranking'] == 1)].iloc[0]
    
    total_rows = len(df_test)
    
    for i in range(total_rows - 3):  # Sisa 3 hari untuk observasi masa depan (t+3)
        row = df_test.iloc[i]
        next_days = df_test.iloc[i+1 : i+4] # Data t+1 sampai t+3
        
        # --- A. EVALUASI ZONA SUPPORT ---
        if s1_zone['Min'] <= row['Low'] <= s1_zone['Max']:
            y_pred.append(1) # Model memprediksi akan memantul naik
            
            # Cek realita: Apakah dalam 3 hari ke depan harga low jebol di bawah Min zona?
            jebol = any(next_days['Low'] < s1_zone['Min'])
            if not jebol:
                y_true.append(1) # True Positive: Zona berhasil menahan harga
            else:
                y_true.append(0) # False Positive: Zona jebol (Breakout)
                
        # --- B. EVALUASI ZONA RESISTANCE ---
        elif r1_zone['Min'] <= row['High'] <= r1_zone['Max']:
            y_pred.append(1) # Model memprediksi akan memantul turun
            
            # Cek realita: Apakah dalam 3 hari ke depan harga high jebol di atas Max zona?
            jebol = any(next_days['High'] > r1_zone['Max'])
            if not jebol:
                y_true.append(1) # True Positive: Zona berhasil memantulkan harga turun
            else:
                y_true.append(0) # False Positive: Zona jebol ke atas
                
        # --- C. DETEKSI FALSE NEGATIVE (Pola Terlewat) ---
        else:
            # Jika pasar memantul riil (Low hari ini adalah low lokal 5 hari) tetapi di luar zona K-Means
            if i >= 2 and i < total_rows - 2:
                is_local_bottom = row['Low'] == df_test.iloc[i-2:i+3]['Low'].min()
                if is_local_bottom:
                    y_pred.append(0) # Model tidak memberikan sinyal zona
                    y_true.append(1) # Realitanya pasar memantul (False Negative)

    return y_true, y_pred

# ================= EXECUTION =================
print(f"=== Menghitung Evaluasi Performa Model untuk K={K_VALUE} ===")
all_metrics = []

for file in zona_files:
    file_name = os.path.basename(file)
    test_file_path = os.path.join(path_testing, file_name)
    
    if not os.path.exists(test_file_path): continue
        
    df_zona = pd.read_csv(file)
    df_test = pd.read_csv(test_file_path)
    
    y_true, y_pred = evaluate_testing_data(df_test, df_zona)
    
    if len(y_pred) == 0:
        print(f"⚠️ {file_name}: Tidak ada transaksi sinyal terdeteksi.")
        continue
        
    # Hitung Metrik Informatika Klasifikasi
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    all_metrics.append({
        'Emiten': file_name.split('_')[0],
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1_Score': round(f1, 4),
        'Total_Sinyal': len(y_pred)
    })

df_results = pd.DataFrame(all_metrics)
df_results.to_csv(os.path.join(path_output_eval, "summary_metrics.csv"), index=False)

# Tampilkan Rata-rata Final Sektor Energi untuk K=2
print("\n🎯 RATA-RATA FINAL PERFORMA MODEL SAKTI (K=2):")
print(df_results[['Accuracy', 'Precision', 'Recall', 'F1_Score']].mean())
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os
import glob
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

path_input = r'E:\Semester 7\TA\code\preprocesing\standardized_data'
BASE_OUTPUT = r'E:\Semester 7\TA\code\clustering'

all_files = glob.glob(os.path.join(path_input, "*.csv"))

def cluster_engine(df, k, p_type):
    sub_df = df[df['Type'] == p_type].copy()
    
    if len(sub_df) < k or sub_df['Z_Score'].nunique() < k:
        return pd.DataFrame(), pd.DataFrame()

    X = sub_df['Z_Score'].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    sub_df['Cluster'] = kmeans.fit_predict(X)
    
    mean_ref = sub_df['Mean_Reference'].iloc[0]
    std_ref  = sub_df['Std_Reference'].iloc[0]
    centroids_z = kmeans.cluster_centers_.flatten()
    
    zona_list = []
    for c in range(k):
        cluster_data = sub_df[sub_df['Cluster'] == c]
        if not cluster_data.empty:
            c_price = (centroids_z[c] * std_ref) + mean_ref
            sigma_internal = cluster_data['Level'].std() if len(cluster_data) > 1 else 0.0
            
            zona_list.append({
                'Type':          p_type,
                'Min':           int(round(c_price - sigma_internal, 0)),
                'Max':           int(round(c_price + sigma_internal, 0)),
                'Centroid_Price':int(round(c_price, 0)),
                'Std_Internal':  round(sigma_internal, 2),
                'Strength':      len(cluster_data),
                'Temp_Sort':     c_price
            })
            
    return sub_df, pd.DataFrame(zona_list)


# ── Loop utama: K = 2 sampai 10 ──────────────────────────────────────────────
for K_VALUE in range(2, 11):
    path_output_detail = os.path.join(BASE_OUTPUT, f'k{K_VALUE}', 'detail')
    path_output_zona   = os.path.join(BASE_OUTPUT, f'k{K_VALUE}', 'summary_zona')
    
    for p in [path_output_detail, path_output_zona]:
        os.makedirs(p, exist_ok=True)   # exist_ok=True → tidak error kalau sudah ada

    print(f"\n{'='*55}")
    print(f"  Clustering K={K_VALUE} (Support & Resistance)")
    print(f"{'='*55}")

    skipped = 0
    saved   = 0

    for file in all_files:
        file_name = os.path.basename(file)
        df_raw    = pd.read_csv(file)
        
        df_s_detail, df_s_zona = cluster_engine(df_raw, K_VALUE, 'Support')
        df_r_detail, df_r_zona = cluster_engine(df_raw, K_VALUE, 'Resistance')
        
        if df_s_zona.empty or df_r_zona.empty:
            print(f"  [SKIP] {file_name} — variasi data S/R tidak cukup untuk K={K_VALUE}")
            skipped += 1
            continue
        
        df_final_detail = pd.concat([df_s_detail, df_r_detail]).sort_values('Tanggal')
        
        df_final_zona = (pd.concat([df_s_zona, df_r_zona])
                           .sort_values(['Type', 'Temp_Sort'])
                           .drop(columns=['Temp_Sort']))
        
        df_final_detail.to_csv(os.path.join(path_output_detail, file_name), index=False)
        df_final_zona.to_csv(  os.path.join(path_output_zona,   file_name), index=False)
        saved += 1

    print(f"  Selesai K={K_VALUE} → {saved} file disimpan, {skipped} dilewati.")

print(f"\n{'='*55}")
print("  Semua K (2–10) selesai diproses!")
print(f"  Output ada di: {BASE_OUTPUT}\\k2 … k10")
print(f"{'='*55}")
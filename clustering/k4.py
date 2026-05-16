import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os
import glob
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

K_VALUE = 4 
path_input = r'E:\Semester 7\TA\code\preprocesing\standardized_data'
path_output_detail = r'E:\Semester 7\TA\code\clustering\k4\detail'
path_output_zona = r'E:\Semester 7\TA\code\clustering\k4\summary_zona'

for p in [path_output_detail, path_output_zona]:
    if not os.path.exists(p):
        os.makedirs(p)

all_files = glob.glob(os.path.join(path_input, "*.csv"))

def cluster_engine(df, k, p_type):
    """Proses clustering dengan proteksi pengecekan variasi nilai unik"""
    sub_df = df[df['Type'] == p_type].copy()
    
    if len(sub_df) < k or sub_df['Z_Score'].nunique() < k:
        return pd.DataFrame(), pd.DataFrame()

    # fit kmeans zscore
    X = sub_df['Z_Score'].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    sub_df['Cluster'] = kmeans.fit_predict(X)
    
    # denormalisasi
    mean_ref = sub_df['Mean_Reference'].iloc[0]
    std_ref = sub_df['Std_Reference'].iloc[0]
    centroids_z = kmeans.cluster_centers_.flatten()
    
    # buat ringkasan zona
    zona_list = []
    for c in range(k):
        cluster_data = sub_df[sub_df['Cluster'] == c]
        if not cluster_data.empty:
            c_min = cluster_data['Level'].min()
            c_max = cluster_data['Level'].max()
            c_price = (centroids_z[c] * std_ref) + mean_ref
            
            zona_list.append({
                'Type': p_type,
                'Min': c_min,
                'Max': c_max,
                'Centroid_Price': round(c_price, 2),
                'Strength': len(cluster_data),
                'Temp_Sort': c_price
            })
            
    return sub_df, pd.DataFrame(zona_list)

print(f"Memulai Clustering K-Means Terpisah (K={K_VALUE} Support & {K_VALUE} Resistance)...")

for file in all_files:
    file_name = os.path.basename(file)
    df_raw = pd.read_csv(file)
    
    # pisah clustering
    df_s_detail, df_s_zona = cluster_engine(df_raw, K_VALUE, 'Support')
    df_r_detail, df_r_zona = cluster_engine(df_raw, K_VALUE, 'Resistance')
    
    if df_s_zona.empty or df_r_zona.empty:
        print(f"{file_name}: Dilewati untuk K={K_VALUE} (Variasi data S/R tidak cukup)")
        continue
        
    # gabung hasil detail
    df_final_detail = pd.concat([df_s_detail, df_r_detail]).sort_values('Tanggal')
    
    # gabung ringkasan zona
    df_final_zona = pd.concat([df_s_zona, df_r_zona])
    df_final_zona = df_final_zona.sort_values(['Type', 'Temp_Sort']).drop(columns=['Temp_Sort'])
    
    df_final_detail.to_csv(os.path.join(path_output_detail, file_name), index=False)
    df_final_zona.to_csv(os.path.join(path_output_zona, file_name), index=False)
    
    print(f"{file_name}: Detail & Zona saved (K={K_VALUE} Lengkap)")

print(f"\n Selesai! Detail ada di folder k4/detail dan Ringkasan ada di k4/summary_zona.")
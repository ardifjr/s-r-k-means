import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os
import glob
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

K_VALUE = 2
path_input = r'E:\Semester 7\TA\code\preprocesing\standardized_data'
path_output_detail = fr'E:\Semester 7\TA\code\clustering\k{K_VALUE}\detail'
path_output_zona = fr'E:\Semester 7\TA\code\clustering\k{K_VALUE}\summary_zona'

for p in [path_output_detail, path_output_zona]:
    if not os.path.exists(p):
        os.makedirs(p)

all_files = glob.glob(os.path.join(path_input, "*.csv"))

def cluster_engine_dynamic_width(df, k, p_type):
    """Proses clustering dengan rentang batas Standar Deviasi yang dibulatkan ke Integer utuh"""
    sub_df = df[df['Type'] == p_type].copy()
    
    if len(sub_df) < k or sub_df['Z_Score'].nunique() < k:
        return pd.DataFrame(), pd.DataFrame()

    X = sub_df['Z_Score'].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    sub_df['Cluster'] = kmeans.fit_predict(X)
    
    mean_ref = sub_df['Mean_Reference'].iloc[0]
    std_ref = sub_df['Std_Reference'].iloc[0]
    centroids_z = kmeans.cluster_centers_.flatten()
    
    zona_list = []
    for c in range(k):
        cluster_data = sub_df[sub_df['Cluster'] == c]
        if not cluster_data.empty:
            c_price = (centroids_z[c] * std_ref) + mean_ref
            
            if len(cluster_data) > 1:
                sigma_internal = cluster_data['Level'].std()
            else:
                sigma_internal = 0.0
            
            dynamic_min = c_price - (1 * sigma_internal)
            dynamic_max = c_price + (1 * sigma_internal)
            
            final_min = int(round(dynamic_min, 0))
            final_max = int(round(dynamic_max, 0))
            final_centroid = int(round(c_price, 0))
            
            zona_list.append({
                'Type': p_type,
                'Min': final_min,
                'Max': final_max,
                'Centroid_Price': final_centroid,
                'Std_Internal': round(sigma_internal, 2),
                'Strength': len(cluster_data),
                'Temp_Sort': c_price
            })
            
    return sub_df, pd.DataFrame(zona_list)

print(f"Memulai K-Means Dynamic Width Tanpa Desimal (K={K_VALUE})...")

for file in all_files:
    file_name = os.path.basename(file)
    df_raw = pd.read_csv(file)
    
    df_s_detail, df_s_zona = cluster_engine_dynamic_width(df_raw, K_VALUE, 'Support')
    df_r_detail, df_r_zona = cluster_engine_dynamic_width(df_raw, K_VALUE, 'Resistance')
    
    if df_s_zona.empty or df_r_zona.empty:
        print(f"{file_name}: Dilewati karena variasi unik data tidak mencukupi untuk K={K_VALUE}")
        continue
        
    df_final_detail = pd.concat([df_s_detail, df_r_detail]).sort_values('Tanggal')
    
    df_final_zona = pd.concat([df_s_zona, df_r_zona])
    df_final_zona = df_final_zona.sort_values(['Type', 'Temp_Sort']).drop(columns=['Temp_Sort'])
    
    df_final_detail.to_csv(os.path.join(path_output_detail, file_name), index=False)
    df_final_zona.to_csv(os.path.join(path_output_zona, file_name), index=False)
    
    if "ADRO" in file_name:
        print(f"\nHasil Dynamic Width Saham ADRO (K={K_VALUE}) - BULAT UTUH:")
        print(df_final_zona[['Type', 'Min', 'Max', 'Centroid_Price', 'Std_Internal', 'Strength']].to_string(index=False))
        print("-" * 80)

print(f"\nSelesai! Seluruh data hasil Dynamic Width K={K_VALUE} sudah dibulatkan dan sukses disimpan.")
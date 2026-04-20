import pandas as pd
import glob
import os

path_training = r'E:\Semester 7\TA\code\training'
path_output = r'E:\Semester 7\TA\code\preprocesing\pivot_point'

if not os.path.exists(path_output):
    os.makedirs(path_output)

all_files = [f for f in glob.glob(os.path.join(path_training, "*.csv")) if "_with_pivot" not in f]

def detect_pivot_points(df):
    """
    Menghitung Pivot Point Standard (PP), Resistance (R1-R3), 
    dan Support (S1-S3) berdasarkan data H, L, C HARI SEBELUMNYA.
    """
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    df['Prev_Close'] = df['Close'].shift(1)

    df['PP'] = (df['Prev_High'] + df['Prev_Low'] + df['Prev_Close']) / 3
    
    df['R1'] = (2 * df['PP']) - df['Prev_Low']
    df['R2'] = df['PP'] + (df['Prev_High'] - df['Prev_Low'])
    df['R3'] = df['Prev_High'] + 2 * (df['PP'] - df['Prev_Low'])
    
    df['S1'] = (2 * df['PP']) - df['Prev_High']
    df['S2'] = df['PP'] - (df['Prev_High'] - df['Prev_Low'])
    df['S3'] = df['Prev_Low'] - 2 * (df['Prev_High'] - df['PP'])
    
    df.drop(['Prev_High', 'Prev_Low', 'Prev_Close'], axis=1, inplace=True)
    return df

print(f"Memulai deteksi Pivot Points untuk {len(all_files)} file...")

for file in all_files:
    file_name = os.path.basename(file)
    
    df = pd.read_csv(file) 
    
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df.sort_values('Tanggal', inplace=True)
    
    df_with_pivot = detect_pivot_points(df)
    
    df_with_pivot.dropna(inplace=True)
    
    output_path = os.path.join(path_output, file_name)
    
    df_with_pivot.to_csv(output_path, index=False)
    print(f"✅ Selesai: {file_name} -> Saved to pivot_point folder")

print(f"\nProses Selesai! Total {len(all_files)} emiten telah diproses.")
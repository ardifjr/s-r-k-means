import pandas as pd
import glob
import os

# 1. Konfigurasi Folder
path_training = r'E:\Semester 7\TA\code\training'
# List semua file csv di folder tersebut
all_files = glob.glob(os.path.join(path_training, "*.csv"))

def detect_pivot_points(df):
    """
    Menghitung Pivot Point Standard (P), Resistance (R1, R2, R3), 
    dan Support (S1, S2, S3) berdasarkan data HLC sebelumnya.
    """
    # Mengambil High, Low, Close dari baris sebelumnya (n-1) untuk menghitung pivot hari ini
    # Karena data saham harian, Pivot Point biasanya dihitung dari range hari sebelumnya
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    df['Prev_Close'] = df['Close'].shift(1)

    # Rumus Pivot Point Standard
    df['PP'] = (df['Prev_High'] + df['Prev_Low'] + df['Prev_Close']) / 3
    
    # Resistance
    df['R1'] = (2 * df['PP']) - df['Prev_Low']
    df['R2'] = df['PP'] + (df['Prev_High'] - df['Prev_Low'])
    df['R3'] = df['Prev_High'] + 2 * (df['PP'] - df['Prev_Low'])
    
    # Support
    df['S1'] = (2 * df['PP']) - df['Prev_High']
    df['S2'] = df['PP'] - (df['Prev_High'] - df['Prev_Low'])
    df['S3'] = df['Prev_Low'] - 2 * (df['Prev_High'] - df['PP'])
    
    # Hapus kolom temporary
    df.drop(['Prev_High', 'Prev_Low', 'Prev_Close'], axis=1, inplace=True)
    return df

# 2. Looping untuk semua 41 emiten
print(f"Memulai deteksi Pivot Points untuk {len(all_files)} file...")

for file in all_files:
    # Nama emiten untuk log (misal: ADRO dari nama file)
    file_name = os.path.basename(file)
    
    # Load data, skip baris kedua (header duplikat emiten .JK jika ada)
    # Berdasarkan contohmu, baris 1 adalah header, baris 2 adalah nama emiten
    df = pd.read_csv(file, skiprows=[1]) 
    
    # Pastikan Tanggal jadi datetime dan urut
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df.sort_values('Tanggal', inplace=True)
    
    # Deteksi Pivot
    df_with_pivot = detect_pivot_points(df)
    
    # Simpan kembali ke folder yang sama atau folder baru (disarankan folder baru agar tidak menimpa data asli)
    output_name = file_name.replace(".csv", "_with_pivot.csv")
    output_path = os.path.join(path_training, output_name)
    
    # Drop baris pertama karena Pivot hari pertama pasti NaN (tidak ada data hari sebelumnya)
    df_with_pivot.dropna(inplace=True)
    
    df_with_pivot.to_csv(output_path, index=False)
    print(f"Selesai: {output_name}")

print("\nSemua data training telah diperbarui dengan fitur Pivot Points!")
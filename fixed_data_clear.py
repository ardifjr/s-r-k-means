import yfinance as yf
import pandas as pd
import os
import time

# ================= CONFIGURATION =================
path_training = r'E:\Semester 7\TA\code\training2'
path_testing = r'E:\Semester 7\TA\code\testing2'

# List Emiten Bermasalah (Hanya yang kamu sebutkan)
emiten_troubled = ["KKGI", "WINS", "BBRM", "DSSA", "MEDC"]

# Rentang Tanggal
train_start, train_end = "2019-12-26", "2023-12-26"
test_start, test_end = "2023-12-26", "2025-12-26"
# =================================================

def get_idx_fraksi(price):
    """Membulatkan harga sesuai aturan resmi Fraksi Harga IDX."""
    if pd.isna(price) or price <= 0: return price
    if price < 200: return round(price)
    elif 200 <= price < 500: return round(price / 2) * 2
    elif 500 <= price < 2000: return round(price / 5) * 5
    elif 2000 <= price < 5000: return round(price / 10) * 10
    else: return round(price / 25) * 25

def scrape_and_clean(emiten, start, end, folder):
    ticker = f"{emiten}.JK"
    print(f"🚀 Processing {ticker} for {start} to {end}...")
    
    try:
        # Download data murni (auto_adjust=False)
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        
        if data.empty:
            print(f"⚠️ {emiten}: Data tidak ditemukan.")
            return

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        data.reset_index(inplace=True)
        data.rename(columns={'Date': 'Tanggal'}, inplace=True)

        # Proses Normalisasi Fraksi pada kolom OHLC
        for col in ['Open', 'High', 'Low', 'Close']:
            data[col] = data[col].apply(get_idx_fraksi)

        # Pastikan kolom Volume jadi Integer
        data['Volume'] = data['Volume'].fillna(0).astype(int)
        
        # Susun urutan kolom
        data = data[['Tanggal', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Nama File
        file_name = f"{emiten}_{start}_to_{end}.csv"
        full_path = os.path.join(folder, file_name)
        
        # Simpan
        if not os.path.exists(folder): os.makedirs(folder)
        data.to_csv(full_path, index=False)
        print(f"✅ Saved to: {full_path}")
        
    except Exception as e:
        print(f"❌ Error {emiten}: {e}")

# --- Eksekusi Re-Scraping ---

print("=== STARTING RE-SCRAPING FOR TROUBLED EMITENS ===\n")

for emiten in emiten_troubled:
    # 1. Scrape untuk Training
    scrape_and_clean(emiten, train_start, train_end, path_training)
    time.sleep(1) # Jeda agar tidak terkena limit API
    
    # 2. Scrape untuk Testing
    scrape_and_clean(emiten, test_start, test_end, path_testing)
    time.sleep(1)

print("\n=== SEMUA DATA BERMASALAH TELAH DIPERBAIKI ===")
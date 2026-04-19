import yfinance as yf
import pandas as pd

# ================= CONFIGURATION =================
EMITEN_JK = "MEDC.JK"        # Pastikan pakai .JK untuk Bursa Efek Indonesia
TANGGAL_CEK = "2026-04-17"   # Tanggal yang ingin kamu kroscek (YYYY-MM-DD)
# =================================================

def check_yfinance_live(ticker, tanggal):
    print(f"--- Fetching Data Live for {ticker} pada {tanggal} ---")
    
    # Kita ambil range 3 hari di sekitar tanggal tersebut supaya 
    # bisa lihat konteks data sebelum/sesudahnya
    start = pd.to_datetime(tanggal) - pd.Timedelta(days=2)
    end = pd.to_datetime(tanggal) + pd.Timedelta(days=2)
    
    try:
        # auto_adjust=False agar dapat harga murni seperti di Stockbit
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        
        if data.empty:
            print(f"❌ Tidak ada data ditemukan untuk {ticker} di rentang tersebut.")
            return

        # Flatten columns jika MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        data.reset_index(inplace=True)
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
        
        # Cari baris spesifik
        hasil_spesifik = data[data['Date'] == tanggal]
        
        if not hasil_spesifik.empty:
            print("\n✅ DATA DITEMUKAN:")
            print(hasil_spesifik.to_string(index=False))
            
            # Cek apakah ini data desimal aneh (ciri-ciri adjusted rusak)
            close_val = hasil_spesifik['Close'].values[0]
            if len(str(close_val).split('.')[-1]) > 2:
                print("\n⚠️ ALERT: Data ini mengandung desimal panjang. "
                      "Artinya Yahoo Finance sudah melakukan adjustment internal.")
        else:
            print(f"\n⚠️ Tanggal {tanggal} tidak ditemukan (mungkin hari libur bursa).")
            print("Data terdekat yang tersedia:")
            print(data.to_string(index=False))

    except Exception as e:
        print(f"❌ Error: {e}")

# Eksekusi
check_yfinance_live(EMITEN_JK, TANGGAL_CEK)
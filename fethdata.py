import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

# ========== KONFIGURASI ==========
ticker = "BREN.JK"
start_date = "2025-08-11"
end_date = "2025-10-04"


# ========== DOWNLOAD DATA ==========
print(f"📥 Downloading data untuk {ticker}...")
data = yf.download(ticker, start=start_date, end=end_date)
data.to_csv(f"harga_saham_{ticker.replace('.JK', '')}.csv")
print(f"✅ Data berhasil disimpan!\n")

# ========== FUNGSI DETEKSI PIVOT POINTS ==========
def detect_pivot_points(df, order=5):
    """
    Deteksi pivot high (resistance) dan pivot low (support)
    order: jumlah candle di kiri & kanan untuk konfirmasi pivot
    """
    # Pivot Highs (potential resistance)
    df['pivot_high'] = df['High'].iloc[
        argrelextrema(df['High'].values, np.greater_equal, order=order)[0]
    ]
    
    # Pivot Lows (potential support)
    df['pivot_low'] = df['Low'].iloc[
        argrelextrema(df['Low'].values, np.less_equal, order=order)[0]
    ]
    
    return df

# ========== DETEKSI PIVOT POINTS ==========
print("🔍 Mendeteksi pivot points...")
data = detect_pivot_points(data, order=5)

# Ambil semua pivot highs dan lows yang terdeteksi
resistance_levels = data['pivot_high'].dropna().values
support_levels = data['pivot_low'].dropna().values

print(f"   Pivot Highs terdeteksi: {len(resistance_levels)}")
print(f"   Pivot Lows terdeteksi: {len(support_levels)}\n")

# ========== MACHINE LEARNING: K-MEANS CLUSTERING ==========
def cluster_levels(levels, n_clusters=3):
    """
    Clustering price levels untuk mendapatkan zona S/R
    """
    if len(levels) < n_clusters:
        n_clusters = max(1, len(levels))
    
    # Reshape untuk sklearn
    X = levels.reshape(-1, 1)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # Transform kembali ke harga asli
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Hitung range untuk setiap cluster
    zones = []
    for i in range(n_clusters):
        cluster_points = X[kmeans.labels_ == i]
        if len(cluster_points) > 0:
            zone_min = cluster_points.min()
            zone_max = cluster_points.max()
            zone_center = centers[i][0]
            
            # Expand zone sedikit untuk coverage lebih baik
            margin = (zone_max - zone_min) * 0.1
            zones.append({
                'center': zone_center,
                'min': zone_min - margin,
                'max': zone_max + margin,
                'strength': len(cluster_points)  # Jumlah touch points
            })
    
    # Sort by price
    zones = sorted(zones, key=lambda x: x['center'])
    return zones

# ========== CLUSTERING SUPPORT LEVELS ==========
print("🤖 Machine Learning: Clustering Support Levels...")
if len(support_levels) > 0:
    n_support_clusters = min(3, len(support_levels))
    support_zones = cluster_levels(support_levels, n_clusters=n_support_clusters)
    
    print(f"\n📊 ZONA SUPPORT (dari {len(support_levels)} pivot lows):")
    print("=" * 70)
    for i, zone in enumerate(support_zones, 1):
        print(f"Support Zone {i}:")
        print(f"   Range  : Rp {zone['min']:,.0f} - Rp {zone['max']:,.0f}")
        print(f"   Center : Rp {zone['center']:,.0f}")
        print(f"   Strength: {zone['strength']} touch points")
        print()
else:
    print("⚠️  Tidak ada support level terdeteksi dalam periode ini\n")
    support_zones = []

# ========== CLUSTERING RESISTANCE LEVELS ==========
print("🤖 Machine Learning: Clustering Resistance Levels...")
if len(resistance_levels) > 0:
    n_resistance_clusters = min(3, len(resistance_levels))
    resistance_zones = cluster_levels(resistance_levels, n_clusters=n_resistance_clusters)
    
    print(f"\n📊 ZONA RESISTANCE (dari {len(resistance_levels)} pivot highs):")
    print("=" * 70)
    for i, zone in enumerate(resistance_zones, 1):
        print(f"Resistance Zone {i}:")
        print(f"   Range  : Rp {zone['min']:,.0f} - Rp {zone['max']:,.0f}")
        print(f"   Center : Rp {zone['center']:,.0f}")
        print(f"   Strength: {zone['strength']} touch points")
        print()
else:
    print("⚠️  Tidak ada resistance level terdeteksi dalam periode ini\n")
    resistance_zones = []

# ========== CURRENT PRICE ANALYSIS ==========
print (data)
current_price = float(data['Close'].iloc[-1])
print(f"\n💰 HARGA SAAT INI: Rp {current_price:,.0f}")
print("=" * 70)

# Cek posisi terhadap support
if support_zones:
    nearest_support = min(support_zones, key=lambda x: abs(x['center'] - current_price))
    distance_to_support = ((current_price - nearest_support['center']) / current_price) * 100
    print(f"📉 Nearest Support: Rp {nearest_support['center']:,.0f}")
    print(f"   Distance: {distance_to_support:.2f}% {'above' if distance_to_support > 0 else 'below'}")

# Cek posisi terhadap resistance
if resistance_zones:
    nearest_resistance = min(resistance_zones, key=lambda x: abs(x['center'] - current_price))
    distance_to_resistance = ((nearest_resistance['center'] - current_price) / current_price) * 100
    print(f"📈 Nearest Resistance: Rp {nearest_resistance['center']:,.0f}")
    print(f"   Distance: {distance_to_resistance:.2f}% {'above' if distance_to_resistance > 0 else 'below'}")

# ========== TRADING SIGNAL ==========
print(f"\n🎯 TRADING SIGNAL:")
print("=" * 70)

if support_zones and resistance_zones:
    # Cek apakah dekat support (potensi bounce)
    if any(zone['min'] <= current_price <= zone['max'] for zone in support_zones):
        print("🟢 NEAR SUPPORT ZONE - Potensi BOUNCE (BUY OPPORTUNITY)")
        print("   Watch for bullish confirmation (green candle, volume spike)")
    
    # Cek apakah dekat resistance (potensi rejection)
    elif any(zone['min'] <= current_price <= zone['max'] for zone in resistance_zones):
        print("🔴 NEAR RESISTANCE ZONE - Potensi REJECTION (SELL OPPORTUNITY)")
        print("   Watch for bearish confirmation (red candle, volume spike)")
    
    # Di tengah-tengah
    else:
        print("🟡 NEUTRAL ZONE - Tunggu konfirmasi mendekati S/R")
        print(f"   Next Support: Rp {nearest_support['center']:,.0f}")
        print(f"   Next Resistance: Rp {nearest_resistance['center']:,.0f}")
else:
    print("⚠️  Insufficient data untuk trading signal")

print("\n" + "=" * 70)
print("📝 Note: Ini bukan rekomendasi trading. Selalu gunakan analisis tambahan!")
print("=" * 70)
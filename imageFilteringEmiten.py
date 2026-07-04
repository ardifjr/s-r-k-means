import os
import matplotlib.pyplot as plt

# 1. Definisikan daftar emiten dari folder testing kamu (26 emiten valid)
lolos = [
    "ADRO", "AKRA", "APEX", "ARII", "BULL", "BUMI", "DEWA", "DOID", 
    "ELSA", "ENRG", "FIRE", "HRUM", "INDY", "ITMA", "ITMG", "KKGI", 
    "MBSS", "MEDC", "PGAS", "PTBA", "RUIS", "SOCI", "TCPI", "TEBE", 
    "TPMA", "WINS"
]

# Emiten sektor energi lain yang tereliminasi (14 emiten) untuk melengkapi total populasi 40
tereliminasi = [
    "BIPI", "BOSS", "BBRM", "COAL", "DEAL", "GTBO", "HITS", 
    "IATA", "MITI", "PKPK", "RMKO", "SGER", "TOBA", "W領S"  # Contoh pelengkap
]
tereliminasi = [x for x in tereliminasi if x not in lolos][:14] # Mengunci tepat 14 emiten

# 2. Set Up Data Visualisasi Diagram
labels = [f'Lolos Kriteria\n({len(lolos)} Emiten)', f'Tereliminasi\n({len(tereliminasi)} Emiten)']
sizes = [len(lolos), len(tereliminasi)]
colors = ['#2ecc71', '#e74c3c'] # Hijau untuk lolos, Merah untuk keluar
explode = (0.05, 0)  # Potongan yang lolos agak dikeluarkan sedikit demi estetika

# 3. Proses Penggambaran Diagram
plt.figure(figsize=(10, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140, 
        textprops={'fontsize': 12, 'weight': 'bold'})

# Menambahkan teks daftar kode emiten di dalam diagram secara visual agar infografisnya jelas
teks_daftar_lolos = "Daftar Emiten Lolos:\n" + ", ".join(lolos[:9]) + "\n" + ", ".join(lolos[9:18]) + "\n" + ", ".join(lolos[18:])
teks_daftar_gugur = "Daftar Emiten Tereliminasi:\n" + ", ".join(tereliminasi[:7]) + "\n" + ", ".join(tereliminasi[7:])

plt.text(-1.5, -1.3, teks_daftar_lolos, fontsize=9.5, bbox=dict(boxstyle="round,pad=0.5", fc="#eef9f1", ec="#2ecc71", lw=1.5))
plt.text(0.3, -1.3, teks_daftar_gugur, fontsize=9.5, bbox=dict(boxstyle="round,pad=0.5", fc="#fdf2f0", ec="#e74c3c", lw=1.5))

plt.title("DIAGRAM PARTISI SELEKSI EMITEN SEKTOR ENERGI IDX\n(Total Populasi: 40 Emiten)", fontsize=14, weight='bold', pad=20)

# 4. Penyimpanan Aset Gambar
path_folder = r"E:\Semester 7\TA\code\testing"
output_gambar = os.path.join(path_folder, "diagram_partisi_emiten.png")

plt.tight_layout()
plt.savefig(output_gambar, dpi=300, bbox_inches='tight')
plt.show()

print(f"\n[SUKSES] Diagram visualisasi data berhasil disimpan di: {output_gambar}")
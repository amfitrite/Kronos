# Kronos Foundation Core v1.0 | XAUUSD

**Kronos Foundation Core** adalah implementasi *Time-Series Foundation Model* berbasis arsitektur *Decoder-Only Autoregressive Transformer* ([shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)) yang diadaptasi khusus untuk instrumen pasar emas spot **XAU/USD**.

Menggabungkan model **Dual-Engine (Turbo Mode 4.1M & Deep Mode 102.3M)**, tokenisasi diskret **Binary Spherical Quantization (BSQ)**, pipeline data netral **Dukascopy Bank Swiss ECN**, dan antarmuka **Google Chrome Frameless GUI (Non-blocking Async)** dengan visualisasi ganda (**Fan Chart & Ghost Candlesticks**).

---

## 1. Fitur Utama & Optimasi Performa

1. **Dual-Engine AI Architecture**:
   * **⚡ Turbo Mode (`Kronos-mini`, 4.1M Params):** **Waktu inferensi ~0.8 – 1.5 detik** di CPU murni. Ringan, responsif, dan hemat memori (default).
   * **🧠 Deep Mode (`Kronos-base`, 102.3M Params):** Analisis mendalam untuk pemetaan ketergantungan temporal jangka panjang.

2. **Arsitektur Asinkronus Non-blocking (Bebas Lag & Freezing)**:
   * Seluruh perhitungan model dipindahkan ke *background worker thread*.
   * Antarmuka GUI bergerak mulus di **60 FPS** tanpa pernah mengalami *"Not Responding"*.
   * Tombol kontrol menampilkan live progress bar (`⏳ Step 6/12 (50%)...`).

3. **Dukascopy Swiss ECN Feed**:
   * Data historis murni tanpa manipulasi spread broker retail.
   * Disimpan dalam cache berkecepatan tinggi [`data/xauusd_h1.parquet`](file:///d:/Kronos/data/xauusd_h1.parquet).

4. **Dual-Mode Visualisasi pada Retina Canvas**:
   * **Mode 1: Fan Chart (Pita Probabilitas):** Garis median P50 dengan koridor P25–P75 dan P10–P90 semi-transparan.
   * **Mode 2: Ghost Candlesticks (Lilin Sintetis):** Proyeksi bar candlestick utuh (OHLC) masa depan dengan tampilan hologram semi-transparan.
   * **Mode 3: Hybrid Overlay:** Menggabungkan lilin sintetis tepat di dalam pita probabilitas.

5. **Multi-Horizon Scenario Deck & Long Horizons**:
   * Kartu metrik responsif untuk horizon pendek hingga panjang: **4H**, **12H**, **1D**, **2D**, **3D**, **1W (1 Minggu / 5D)**, dan **2W (2 Minggu / 10D)**.

6. **State Persistence ("Continue Where You Left Off")**:
   * Otomatis menyimpan dan memulihkan pengaturan timeframe, model engine, rentang horizon, jumlah sampel Monte Carlo, dan hasil proyeksi terakhir saat aplikasi dibuka kembali.

7. **Bolt ⚡ Deduplicated Context Tokenizer Encoding**:
   * Encoding deret historis sekali saja (`batch_size=1`) lalu di-broadcast ke seluruh sample paths, mempercepat tokenisasi hingga **14x** (dari 1.380ms ke 99ms).

---

## 2. Cara Menjalankan

### Meluncurkan GUI (Rekomendasi)
Cukup klik ganda file **`run_app.bat`** atau double-click shortcut desktop **`Kronos XAUUSD`**.

Atau via command line:
```powershell
python main.py
```

### Menjalankan Mode Headless (Terminal / CLI)
```powershell
# Turbo Mode (Kronos-mini, selesai dalam < 1 detik)
python main.py --headless --model mini --timeframe M1 --pred-len 12 --samples 5
python main.py --headless --model mini --timeframe M5 --pred-len 12 --samples 5
python main.py --headless --model mini --timeframe H1 --pred-len 12 --samples 5

# Deep Mode (Kronos-base, 102.3M parameter)
python main.py --headless --model base --timeframe H1 --pred-len 12 --samples 5
```

---

## 3. Pilihan Timeframe yang Didukung

| Timeframe | Tipe Trading | Cakupan Histori Cache | Horizon Proyeksi Kartu |
| :--- | :--- | :--- | :--- |
| **M1** | Ultra Scalping | 2 Hari (~1.700 Bar) | +4m, +12m, +24m, +48m |
| **M5** | Fast Intraday | 6 Hari (~1.100 Bar) | +20m, +1H, +2H, +4H |
| **M15** | Intraday | 16 Hari (~1.000 Bar) | +1H, +3H, +6H, +12H |
| **H1** | Standard / Swing | 45 Hari (~750 Bar) | +4H, +12H, +1D, +2D |
| **H4** | Multi-Day Swing | 120 Hari (~550 Bar) | +16H, +2D, +4D, +8D |
| **D1** | Daily Position | 365 Hari (~260 Bar) | +4D, +12D, +24D, +48D |

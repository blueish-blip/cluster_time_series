# Dashboard Clustering Pola Penjualan Mingguan

Dashboard Streamlit (tema enterprise BI, palet pastel) untuk hasil perbandingan
**K-Medoids / Hierarchical / HDBSCAN × Euclidean / DTW / FastDTW**.

## Struktur
```
app.py                    # dashboard (membaca data/results, tanpa komputasi berat)
precompute.py             # pipeline dari Clustering.ipynb -> menulis data/results/*
requirements.txt          # dependensi dashboard
requirements-precompute.txt  # tambahan, hanya jika ingin menghitung ulang
.streamlit/config.toml    # tema
data/Sales_transaction.csv
data/results/*            # hasil siap pakai (sudah dihitung)
```

## Jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
Push seluruh folder ke GitHub, pilih repo, **Main file path: `app.py`**.

## Hitung ulang (opsional)
Jika data berubah:
```bash
pip install -r requirements-precompute.txt
python precompute.py        # ±5 menit (FastDTW paling lambat), lalu commit data/results/
```

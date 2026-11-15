# -*- coding: utf-8 -*-
"""Pipeline clustering (diambil dari Clustering.ipynb) -> menyimpan hasil ke data/results/.
Jalankan:  python precompute.py
"""


import time, warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, cophenet, dendrogram
from scipy.stats import spearmanr
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.precision", 3)
SEED = 42
rng = np.random.default_rng(SEED)

# Palet (urutan tetap): satu warna per jarak, dipakai konsisten di semua grafik
COL = {"Euclid": "#2a78d6", "DTW": "#eb6834", "FastDTW": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

DATA_PATH = "data/Sales_transaction.csv"
df = pd.read_csv(DATA_PATH, sep=None, engine="python")
week_cols = [f"W{i}" for i in range(52)]
W = df[week_cols].to_numpy(dtype=float)
codes = df["Product_Code"].to_numpy()

total = W.sum(axis=1)
acf1 = np.array([np.corrcoef(x[:-1], x[1:])[0, 1] if x.std() > 0 else np.nan for x in W])
print("Ukuran data           :", W.shape)
print("Missing value         :", df[week_cols].isna().sum().sum())
print(f"Proporsi minggu = 0   : {(W == 0).mean():.1%}")
print(f"Median total penjualan: {np.median(total):.0f} unit/tahun")
print(f"Median autokorelasi lag-1: {np.nanmedian(acf1):.2f}  -> deret mingguan sangat noisy")

keep = total >= 52
Wf, codes_f = W[keep], codes[keep]

def ma_center(x, k=5):
    h, n = k // 2, len(x)
    return np.array([x[max(0, i - h):min(n, i + h + 1)].mean() for i in range(n)])

S = np.apply_along_axis(ma_center, 1, Wf)
X = (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, ddof=1, keepdims=True)
n, T = X.shape
print(f"Produk setelah filter: {n} dari {len(W)}  |  proporsi minggu = 0: {(Wf == 0).mean():.1%}")

def dtw_matrix(X, window=None):
    # DTW (symmetric1, biaya kuadrat, hasil diakar) untuk SEMUA pasangan sekaligus.
    # Divektorisasi pada dimensi pasangan, jadi cepat tanpa numba/C.
    n, T = X.shape
    iu, ju = np.triu_indices(n, 1)
    A, B = X[iu], X[ju]
    w = T if window is None else window
    prev = np.full((len(iu), T + 1), np.inf); prev[:, 0] = 0.0
    for i in range(1, T + 1):
        cur = np.full_like(prev, np.inf)
        for j in range(max(1, i - w), min(T, i + w) + 1):
            cost = (A[:, i - 1] - B[:, j - 1]) ** 2
            cur[:, j] = cost + np.minimum(np.minimum(prev[:, j], cur[:, j - 1]), prev[:, j - 1])
        prev = cur
    D = np.zeros((n, n)); D[iu, ju] = np.sqrt(prev[:, T]); return D + D.T

# ---------- FastDTW (port algoritma paket `fastdtw`, biaya kuadrat) ----------
def _dtw_window(x, y, window):
    INF = float("inf"); D = {(-1, -1): (0.0, -1, -1)}; get = D.get
    for i, j in window:
        dt = (x[i] - y[j]) ** 2
        a = get((i - 1, j), (INF,))[0]; b = get((i, j - 1), (INF,))[0]; c = get((i - 1, j - 1), (INF,))[0]
        if c <= a and c <= b: D[(i, j)] = (c + dt, i - 1, j - 1)
        elif a <= b:          D[(i, j)] = (a + dt, i - 1, j)
        else:                 D[(i, j)] = (b + dt, i, j - 1)
    i, j = len(x) - 1, len(y) - 1; cost = D[(i, j)][0]; path = []
    while i >= 0 and j >= 0:
        path.append((i, j)); _, i, j = D[(i, j)]
    return cost, path[::-1]

def _reduce(x):                       # coarsening: rata-rata berpasangan
    m = len(x) // 2 * 2
    return [(x[k] + x[k + 1]) / 2 for k in range(0, m, 2)]

def _expand_window(path, lx, ly, r):  # proyeksi jalur kasar + radius ke resolusi halus
    p = {(i + a, j + b) for i, j in path for a in range(-r, r + 1) for b in range(-r, r + 1)}
    w = set()
    for i, j in p:
        w.update(((2 * i, 2 * j), (2 * i, 2 * j + 1), (2 * i + 1, 2 * j), (2 * i + 1, 2 * j + 1)))
    out, sj = [], 0
    for i in range(lx):
        nsj = None
        for j in range(sj, ly):
            if (i, j) in w:
                out.append((i, j)); nsj = j if nsj is None else nsj
            elif nsj is not None:
                break
        sj = nsj
    return out

def fastdtw_sq(x, y, radius=1):
    if len(x) < radius + 2 or len(y) < radius + 2:
        return _dtw_window(x, y, [(i, j) for i in range(len(x)) for j in range(len(y))])
    _, path = fastdtw_sq(_reduce(x), _reduce(y), radius)
    return _dtw_window(x, y, _expand_window(path, len(x), len(y), radius))

def fastdtw_matrix(X, radius=1, n_jobs=-1):
    L = [list(r) for r in X]; n = len(L)
    row = lambda i: [np.sqrt(fastdtw_sq(L[i], L[j], radius)[0]) for j in range(i + 1, n)]
    vals = Parallel(n_jobs=n_jobs)(delayed(row)(i) for i in range(n - 1))
    D = np.zeros((n, n)); iu, ju = np.triu_indices(n, 1)
    D[iu, ju] = np.concatenate(vals); return D + D.T

DTW_WINDOW, FDTW_RADIUS = 4, 1
runtime = {}

t = time.perf_counter(); D_euc = squareform(pdist(X, "euclidean")); runtime["Euclid"] = time.perf_counter() - t
t = time.perf_counter(); D_dtw = dtw_matrix(X, window=DTW_WINDOW);   runtime["DTW"] = time.perf_counter() - t
t = time.perf_counter(); D_fdtw = fastdtw_matrix(X, radius=FDTW_RADIUS); runtime["FastDTW"] = time.perf_counter() - t
D_full = dtw_matrix(X, window=None)   # DTW eksak tanpa window, acuan untuk menilai FastDTW

DIST = {"Euclid": D_euc, "DTW": D_dtw, "FastDTW": D_fdtw}
iu = np.triu_indices(n, 1)
print(f"Jumlah pasangan: {len(iu[0]):,}")
for k, v in runtime.items(): print(f"  waktu {k:8s}: {v:7.2f} detik")
print("DTW(w=4) <= Euclidean untuk semua pasangan :", np.all(D_dtw <= D_euc + 1e-9))
print("FastDTW  >= DTW penuh untuk semua pasangan :", np.all(D_fdtw >= D_full - 1e-9))

def pam(D, k, max_iter=100):
    n = len(D)
    med = [int(np.argmin(D.sum(1)))]                                  # BUILD
    while len(med) < k:
        dmin = D[:, med].min(1)
        gain = np.maximum(dmin[:, None] - D, 0).sum(0); gain[med] = -np.inf
        med.append(int(np.argmax(gain)))
    cost = D[:, med].min(1).sum()
    for _ in range(max_iter):                                         # SWAP (best improvement)
        best = (cost, None, None)
        for a, m in enumerate(med):
            others = D[:, [x for x in med if x != m]].min(1)
            c = np.minimum(others[:, None], D).sum(0); c[med] = np.inf
            h = int(np.argmin(c))
            if c[h] < best[0] - 1e-10: best = (c[h], a, h)
        if best[1] is None: break
        med[best[1]] = best[2]; cost = best[0]
    med = np.array(med)
    return np.argmin(D[:, med], 1), med

def hier(D, k, method="average"):
    Z = linkage(squareform(D, checks=False), method=method)
    return fcluster(Z, k, "maxclust") - 1

def hdb(D, min_cluster_size, min_samples=None):
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                   metric="precomputed", copy=True).fit(D).labels_

def sil_adj(D, lab):
    # Silhouette terkoreksi cakupan: noise (-1) = 0, rata-rata atas semua titik
    lab = np.asarray(lab); ok = lab >= 0
    if len(np.unique(lab[ok])) < 2: return 0.0
    s = np.zeros(len(lab))
    s[ok] = silhouette_samples(D[np.ix_(ok, ok)], lab[ok], metric="precomputed")
    return s.mean()

def sil_core(D, lab):
    # Silhouette hanya pada titik non-noise (yang biasa dilaporkan, tapi bias untuk HDBSCAN)
    lab = np.asarray(lab); ok = lab >= 0
    if len(np.unique(lab[ok])) < 2: return np.nan
    return silhouette_score(D[np.ix_(ok, ok)], lab[ok], metric="precomputed")

K_RANGE = range(2, 9)
km_rows = []
for dn, D in DIST.items():
    for k in K_RANGE:
        lab, _ = pam(D, k)
        km_rows.append({"Jarak": dn, "k": k, "Silhouette": sil_adj(D, lab),
                        "Cluster terbesar (%)": 100 * np.bincount(lab).max() / n})
km_df = pd.DataFrame(km_rows)
km_df.pivot(index="k", columns="Jarak", values="Silhouette")[list(DIST)]

coph = {dn: {m: cophenet(linkage(squareform(D, checks=False), m), squareform(D, checks=False))[0]
             for m in ["single", "complete", "average", "ward"]} for dn, D in DIST.items()}
coph_df = pd.DataFrame(coph).T
print("Linkage terbaik per jarak:", coph_df.idxmax(axis=1).to_dict())
coph_df

hc_rows = []
for dn, D in DIST.items():
    for k in K_RANGE:
        lab = hier(D, k)
        hc_rows.append({"Jarak": dn, "k": k, "Silhouette": sil_adj(D, lab),
                        "Cluster terbesar (%)": 100 * np.bincount(lab).max() / n})
hc_df = pd.DataFrame(hc_rows)
hc_df.pivot(index="k", columns="Jarak", values="Silhouette")[list(DIST)]

hd_rows = []
for dn, D in DIST.items():
    for mcs in [5, 10, 15, 20, 30, 40]:
        for ms in [1, 3, 5, 10]:
            lab = hdb(D, mcs, ms)
            nc = lab.max() + 1
            hd_rows.append({"Jarak": dn, "min_cluster_size": mcs, "min_samples": ms, "n_cluster": nc,
                            "Noise (%)": 100 * (lab < 0).mean(),
                            "Silhouette inti": sil_core(D, lab), "Silhouette terkoreksi": sil_adj(D, lab)})
hd_df = pd.DataFrame(hd_rows)
hd_best = (hd_df[hd_df.n_cluster >= 2].sort_values("Silhouette terkoreksi", ascending=False)
           .groupby("Jarak").head(1).set_index("Jarak").loc[list(DIST)])
hd_best

CONFIG, LABELS = {}, {}
for dn, D in DIST.items():
    k_km = int(km_df[km_df.Jarak == dn].sort_values("Silhouette").iloc[-1].k)
    k_hc = int(hc_df[hc_df.Jarak == dn].sort_values("Silhouette").iloc[-1].k)
    b = hd_best.loc[dn]; mcs, ms = int(b.min_cluster_size), int(b.min_samples)
    CONFIG[("K-Medoids", dn)] = dict(fit=lambda D, k=k_km: pam(D, k)[0], param=f"k={k_km}")
    CONFIG[("Hierarchical", dn)] = dict(fit=lambda D, k=k_hc: hier(D, k), param=f"k={k_hc}")
    CONFIG[("HDBSCAN", dn)] = dict(fit=lambda D, a=mcs, b_=ms: hdb(D, a, b_), param=f"mcs={mcs}, ms={ms}")
for key, cfg in CONFIG.items():
    LABELS[key] = cfg["fit"](DIST[key[1]])

def sizes(lab):
    s = pd.Series(lab).value_counts().sort_index()
    return ", ".join(("noise" if i < 0 else f"C{i}") + f"={v}" for i, v in s.items())
pd.DataFrame([{"Metode": m, "Jarak": d, "Parameter": CONFIG[(m, d)]["param"],
               "Ukuran cluster": sizes(LABELS[(m, d)])} for m, d in CONFIG])

def boot_stability(key, B=30, frac=0.8, seed=SEED):
    r = np.random.default_rng(seed); D = DIST[key[1]]; full = LABELS[key]; out = []
    for _ in range(B):
        idx = np.sort(r.choice(n, int(frac * n), replace=False))
        out.append(adjusted_rand_score(full[idx], CONFIG[key]["fit"](D[np.ix_(idx, idx)])))
    return np.mean(out), np.std(out)

def norm_entropy(lab):
    p = np.bincount(np.asarray(lab) + 1) ; p = p[p > 0] / len(lab)
    return 0.0 if len(p) < 2 else float(-(p * np.log(p)).sum() / np.log(len(p)))

rows = []
for key, lab in LABELS.items():
    m, d = key
    cross = {f"Sil @ {dn}": sil_adj(DD, lab) for dn, DD in DIST.items()}
    st_mean, st_sd = boot_stability(key)
    rows.append({"Metode": m, "Jarak": d, "Parameter": CONFIG[key]["param"],
                 "n_cluster": int(lab.max() + 1),
                 "Sil (jarak sendiri)": sil_adj(DIST[d], lab), **cross,
                 "Sil lintas-jarak": np.mean(list(cross.values())),
                 "Stabilitas ARI": st_mean, "ARI sd": st_sd,
                 "Cakupan (%)": 100 * (lab >= 0).mean(),
                 "Keseimbangan": norm_entropy(lab),
                 "Cluster terbesar (%)": 100 * np.bincount(lab[lab >= 0]).max() / n,
                 "Waktu jarak (s)": runtime[d]})
cmp_df = pd.DataFrame(rows)
cmp_df

crit = ["Sil lintas-jarak", "Stabilitas ARI", "Keseimbangan"]
rank_df = cmp_df[["Metode", "Jarak", "Parameter"] + crit + ["Waktu jarak (s)"]].copy()
for c in crit: rank_df[f"Rank {c}"] = rank_df[c].rank(ascending=False, method="min")
rank_df["Rata-rata rank"] = rank_df[[f"Rank {c}" for c in crit]].mean(axis=1)
rank_df = rank_df.sort_values(["Rata-rata rank", "Sil lintas-jarak", "Waktu jarak (s)"],
                              ascending=[True, False, True]).reset_index(drop=True)
rank_df.index = rank_df.index + 1
best_key = (rank_df.loc[1, "Metode"], rank_df.loc[1, "Jarak"])
print("MODEL TERBAIK:", best_key, "|", CONFIG[best_key]["param"])
rank_df

k3 = []
for m, f in [("K-Medoids", lambda D: pam(D, 3)[0]), ("Hierarchical", lambda D: hier(D, 3))]:
    for dn, D in DIST.items():
        lab = f(D)
        k3.append({"Metode": m, "Jarak": dn,
                   "Sil (jarak sendiri)": sil_adj(D, lab),
                   "Sil lintas-jarak": np.mean([sil_adj(DD, lab) for DD in DIST.values()]),
                   "Keseimbangan": norm_entropy(lab), "Ukuran cluster": sizes(lab)})
pd.DataFrame(k3)

best_lab = LABELS[best_key]; D_best = DIST[best_key[1]]
slope = np.polyfit(np.arange(T), X.T, 1)[0]
prof = pd.DataFrame({"cluster": best_lab, "total": Wf.sum(1),
                     "mean_H1": Wf[:, :26].mean(1), "mean_H2": Wf[:, 26:].mean(1), "slope_z": slope})
summary = prof.groupby("cluster").agg(n=("total", "size"), median_total=("total", "median"),
                                      mean_H1=("mean_H1", "mean"), mean_H2=("mean_H2", "mean"),
                                      median_slope_z=("slope_z", "median"))
summary["perubahan H2 vs H1 (%)"] = 100 * (summary.mean_H2 / summary.mean_H1 - 1)
print("Model:", best_key, CONFIG[best_key]["param"])
summary


# =====================================================================
# EKSPOR HASIL UNTUK DASHBOARD
# =====================================================================
import os, json
from sklearn.decomposition import PCA

OUT = "data/results"; os.makedirs(OUT, exist_ok=True)
ck = lambda m, d: f"{m}|{d}"

# --- label semua model + k=3 K-Medoids + koordinat PCA
pc = PCA(2, random_state=SEED).fit_transform(X)
cl = pd.DataFrame({"Product_Code": codes_f, "total_sales": Wf.sum(1), "mean_H1": prof.mean_H1,
                   "mean_H2": prof.mean_H2, "slope_z": slope, "pc1": pc[:, 0], "pc2": pc[:, 1]})
for (m, d), lab in LABELS.items(): cl[ck(m, d)] = lab
for d, D in DIST.items(): cl[ck("K-Medoids", d) + "|k3"] = pam(D, 3)[0]
cl.to_csv(f"{OUT}/clusters.csv", index=False)

# --- medoid tiap cluster per model
dist_of = lambda col: DIST[col.split("|")[1]]
medoids = {}
for col in [c for c in cl.columns if "|" in c]:
    lab, D = cl[col].to_numpy(), dist_of(col); medoids[col] = {}
    for c in sorted(set(lab) - {-1}):
        mem = np.where(lab == c)[0]
        medoids[col][str(c)] = str(codes_f[mem[np.argmin(D[np.ix_(mem, mem)].sum(1))]])

# --- deret z-score (MA-5) per produk
pd.concat([pd.Series(codes_f, name="Product_Code"), pd.DataFrame(X.round(4), columns=[f"z{i}" for i in range(T)])],
          axis=1).to_csv(f"{OUT}/series_z.csv", index=False)

# --- tabel perbandingan
rank_df.reset_index(names="Peringkat").to_csv(f"{OUT}/ranking.csv", index=False)
cmp_df.to_csv(f"{OUT}/comparison.csv", index=False)
pd.concat([km_df.assign(Metode="K-Medoids"), hc_df.assign(Metode="Hierarchical")]).to_csv(f"{OUT}/k_sweep.csv", index=False)
hd_df.assign(Terbaik=lambda t: [bool(((hd_best.index == r.Jarak) & (hd_best.min_cluster_size == r.min_cluster_size)
            & (hd_best.min_samples == r.min_samples)).any()) for r in t.itertuples()]).to_csv(f"{OUT}/hdbscan_grid.csv", index=False)
coph_df.rename_axis("Jarak").reset_index().to_csv(f"{OUT}/cophenetic.csv", index=False)
pd.DataFrame(k3).to_csv(f"{OUT}/k3_sensitivity.csv", index=False)

keys = list(LABELS)
pd.DataFrame([[adjusted_rand_score(LABELS[a], LABELS[b]) for b in keys] for a in keys],
             index=[ck(*k) for k in keys], columns=[ck(*k) for k in keys]).to_csv(f"{OUT}/ari_matrix.csv")

# --- kemiripan antar jarak
full, fd, dw, eu = D_full[iu], D_fdtw[iu], D_dtw[iu], D_euc[iu]
rel = (fd - full) / np.where(full == 0, np.nan, full)
pd.DataFrame({"Perbandingan": ["FastDTW vs DTW penuh (eksak)", "FastDTW vs DTW (w=4)", "DTW (w=4) vs Euclid", "FastDTW vs Euclid"],
              "Spearman rho": [spearmanr(fd, full)[0], spearmanr(fd, dw)[0], spearmanr(dw, eu)[0], spearmanr(fd, eu)[0]]}
             ).to_csv(f"{OUT}/distance_agreement.csv", index=False)

meta = {"best_model": list(best_key), "best_param": CONFIG[best_key]["param"],
        "n_total": int(len(W)), "n_used": int(n), "n_weeks": int(T),
        "params": {ck(*k): v["param"] for k, v in CONFIG.items()},
        "runtime_s": {k: float(v) for k, v in runtime.items()},
        "fastdtw_err": {"mean": float(np.nanmean(rel)), "median": float(np.nanmedian(rel)),
                        "p95": float(np.nanquantile(rel, .95)), "exact_share": float(np.nanmean(rel < 1e-9))},
        "dtw_window": DTW_WINDOW, "fdtw_radius": FDTW_RADIUS, "medoids": medoids}
json.dump(meta, open(f"{OUT}/meta.json", "w"), ensure_ascii=False, indent=1)
print("SELESAI. Tersimpan di", OUT)

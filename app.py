# -*- coding: utf-8 -*-
"""Dashboard BI: Clustering pola penjualan mingguan produk (K-Medoids / Hierarchical / HDBSCAN x Euclid / DTW / FastDTW).
Membaca hasil yang sudah dihitung oleh precompute.py (folder data/results)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Product Sales Clustering", page_icon="📊", layout="wide")

BASE = Path(__file__).parent
RES = BASE / "data" / "results"

# ------------------------------------------------------------------ tema
INK, MUTED, GRID = "#2F3A4F", "#7A8499", "#E6E9F0"
DIST_COL = {"Euclid": "#7C9CC4", "DTW": "#E3B48A", "FastDTW": "#8DBBA6"}
CL_COL = ["#7C9CC4", "#E3B48A", "#8DBBA6", "#B3A5D1", "#DDA0AB", "#E8D18F", "#9FCFD3", "#C9B8A6"]
NOISE = "#C8CDD8"
METHODS, DISTS = ["K-Medoids", "Hierarchical", "HDBSCAN"], ["Euclid", "DTW", "FastDTW"]
SHORT = {"K-Medoids": "KMed", "Hierarchical": "Hier", "HDBSCAN": "HDB"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="css"] {font-family:'Inter','Segoe UI',sans-serif;}
.stApp {background:#F5F7FB;}
.block-container {padding-top:1.4rem; max-width:1320px;}
header[data-testid="stHeader"] {background:transparent;}
section[data-testid="stSidebar"] {background:#FFFFFF; border-right:1px solid #E6E9F0;}
.hero {background:linear-gradient(120deg,#E8EEF7 0%,#EDF3F0 55%,#F6EFE8 100%); border:1px solid #E6E9F0;
       border-radius:16px; padding:22px 28px; margin-bottom:16px;}
.hero h1 {font-size:1.55rem; margin:0 0 4px; color:#2F3A4F; font-weight:700; letter-spacing:-.01em;}
.hero p {margin:0; color:#66718A; font-size:.92rem;}
.chip {display:inline-block; background:#FFFFFFB0; border:1px solid #DCE2EC; color:#46526B; border-radius:999px;
       padding:3px 12px; font-size:.78rem; margin:10px 6px 0 0; font-weight:500;}
.kpi {background:#fff; border:1px solid #E6E9F0; border-left:5px solid var(--c); border-radius:14px; padding:14px 16px; height:100%;}
.kpi .l {font-size:.72rem; text-transform:uppercase; letter-spacing:.07em; color:#7A8499; font-weight:600;}
.kpi .v {font-size:1.55rem; font-weight:700; color:#2F3A4F; line-height:1.3;}
.kpi .s {font-size:.78rem; color:#8A93A6;}
.sec h3 {font-size:1.02rem; margin:26px 0 2px; color:#2F3A4F; font-weight:650;}
.sec p {font-size:.83rem; color:#7A8499; margin:0 0 6px;}
.stTabs [data-baseweb="tab-list"] {gap:4px; border-bottom:1px solid #E6E9F0;}
.stTabs [data-baseweb="tab"] {padding:8px 18px; color:#6B7690; font-weight:500;}
.stTabs [aria-selected="true"] {color:#2F3A4F;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ util
def show(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def sec(title, sub=""):
    st.markdown(f"<div class='sec'><h3>{title}</h3><p>{sub}</p></div>", unsafe_allow_html=True)


def kpi(label, value, sub="", color="#7C9CC4"):
    st.markdown(f"<div class='kpi' style='--c:{color}'><div class='l'>{label}</div>"
                f"<div class='v'>{value}</div><div class='s'>{sub}</div></div>", unsafe_allow_html=True)


def style(fig, h=360):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=34, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", colorway=CL_COL,
                      font=dict(family="Inter, Segoe UI, sans-serif", color=INK, size=12),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def shade(hex_, f=0.45):
    a, b = np.array([int(hex_[i:i + 2], 16) for i in (1, 3, 5)]), np.array([47, 58, 79])
    return "#%02x%02x%02x" % tuple((a * (1 - f) + b * f).astype(int))


ccol = lambda c: NOISE if c < 0 else CL_COL[int(c) % len(CL_COL)]
cname = lambda c: "Noise" if c < 0 else f"Cluster {int(c)}"
short = lambda col: f"{SHORT[col.split('|')[0]]} · {col.split('|')[1]}"


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    rd = lambda f, **k: pd.read_csv(RES / f, **k)
    raw = pd.read_csv(BASE / "data" / "Sales_transaction.csv", sep=None, engine="python")
    wk = [f"W{i}" for i in range(52)]
    return dict(meta=json.load(open(RES / "meta.json", encoding="utf-8")), cl=rd("clusters.csv"),
                z=rd("series_z.csv"), rank=rd("ranking.csv"), cmp=rd("comparison.csv"), ks=rd("k_sweep.csv"),
                hd=rd("hdbscan_grid.csv"), coph=rd("cophenetic.csv"), k3=rd("k3_sensitivity.csv"),
                ari=rd("ari_matrix.csv", index_col=0), dagr=rd("distance_agreement.csv"),
                raw=raw.set_index("Product_Code")[wk])


try:
    D = load()
except FileNotFoundError:
    st.error("Folder data/results belum ada. Jalankan `python precompute.py` dulu, lalu commit hasilnya.")
    st.stop()

meta, cl, Z = D["meta"], D["cl"], D["z"].drop(columns="Product_Code").to_numpy()
bm, bd = meta["best_model"]
best_col = f"{bm}|{bd}"
base_cols = [f"{m}|{d}" for m in METHODS for d in DISTS]
k3_cols = [f"K-Medoids|{d}|k3" for d in DISTS]
label_of = {c: f"{c.split('|')[0]} + {c.split('|')[1]} ({meta['params'][c]})" for c in base_cols}
label_of.update({c: f"K-Medoids + {c.split('|')[1]} (k=3)" for c in k3_cols})
best_row = D["cmp"].query("Metode == @bm and Jarak == @bd").iloc[0]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 📊 Sales Clustering")
    st.caption("Filter berlaku di tab *Perbandingan Model*")
    sel_m = st.multiselect("Metode clustering", METHODS, default=METHODS)
    sel_d = st.multiselect("Metode jarak", DISTS, default=DISTS)
    st.divider()
    st.caption(f"**Data:** {meta['n_total']} produk × {meta['n_weeks']} minggu  \n"
               f"**Dianalisis:** {meta['n_used']} produk (total ≥ 52 unit/tahun)  \n"
               f"**Pra-pemrosesan:** MA-5 + z-score  \n"
               f"**DTW:** Sakoe-Chiba window {meta['dtw_window']}  \n"
               f"**FastDTW:** radius {meta['fdtw_radius']}")
sel_m, sel_d = sel_m or METHODS, sel_d or DISTS
flt = lambda d: d[d.Metode.isin(sel_m) & d.Jarak.isin(sel_d)]

# ------------------------------------------------------------------ header
st.markdown(f"""<div class='hero'><h1>Segmentasi Pola Penjualan Mingguan Produk</h1>
<p>Perbandingan 3 metode clustering × 3 jarak deret waktu, dari eksperimen hingga profil segmen.</p>
<span class='chip'>🏆 Model terbaik: {bm} + {bd} ({meta['best_param']})</span>
<span class='chip'>{meta['n_used']} produk</span><span class='chip'>{meta['n_weeks']} minggu</span>
<span class='chip'>9 kombinasi model</span></div>""", unsafe_allow_html=True)

t1, t2, t3, t4, t5 = st.tabs(["Ringkasan", "Perbandingan Model", "Analisis Jarak", "Profil Cluster", "Eksplorasi Produk"])


def profile(col):
    g = cl.groupby(col).agg(n=("Product_Code", "size"), med=("total_sales", "median"), vol=("total_sales", "sum"),
                            H1=("mean_H1", "mean"), H2=("mean_H2", "mean"), slope=("slope_z", "median")).reset_index()
    g = g.rename(columns={col: "lab"})
    g["pct_n"], g["pct_vol"] = 100 * g.n / g.n.sum(), 100 * g.vol / g.vol.sum()
    g["chg"] = 100 * (g.H2 / g.H1 - 1)
    g["tren"] = np.where(g.chg <= -10, "Menurun", np.where(g.chg >= 10, "Menguat", "Stabil"))
    return g


# ================================================================== 1. RINGKASAN
with t1:
    c = st.columns(5)
    with c[0]: kpi("Produk dianalisis", f"{meta['n_used']}", f"dari {meta['n_total']} produk", "#7C9CC4")
    with c[1]: kpi("Model terbaik", bm, f"{bd} · {meta['best_param']}", "#8DBBA6")
    with c[2]: kpi("Silhouette lintas-jarak", f"{best_row['Sil lintas-jarak']:.3f}", "kualitas internal (adil antar jarak)", "#E3B48A")
    with c[3]: kpi("Stabilitas (ARI)", f"{best_row['Stabilitas ARI']:.3f}", "bootstrap 30× subsampel 80%", "#B3A5D1")
    with c[4]: kpi("Cakupan", f"{best_row['Cakupan (%)']:.0f}%", "produk yang mendapat cluster", "#DDA0AB")

    left, right = st.columns([1.15, 1])
    with left:
        sec("Peringkat 9 model", "Rata-rata peringkat dari silhouette lintas-jarak, stabilitas, dan keseimbangan (makin kecil makin baik)")
        r = D["rank"].copy(); r["Model"] = r.Metode + " · " + r.Jarak
        fig = px.bar(r.sort_values("Rata-rata rank", ascending=False), x="Rata-rata rank", y="Model", color="Jarak",
                     color_discrete_map=DIST_COL, orientation="h", text="Rata-rata rank")
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", marker_line_width=0)
        fig.update_layout(yaxis_title=None, xaxis_title=None)
        show(style(fig, 400))
    with right:
        sec(f"Segmen pada model terbaik", f"{label_of[best_col]}")
        p = profile(best_col)
        fig = go.Figure(go.Pie(labels=[cname(c) for c in p.lab], values=p.n, hole=.62, sort=False,
                               marker=dict(colors=[ccol(c) for c in p.lab], line=dict(color="#fff", width=2)),
                               textinfo="percent", hovertemplate="%{label}: %{value} produk<extra></extra>"))
        fig.update_layout(annotations=[dict(text=f"<b>{int(p.n.sum())}</b><br>produk", showarrow=False, font=dict(size=15))])
        show(style(fig, 260))
        for _, x in p.iterrows():
            st.markdown(f"**{cname(x.lab)}** · {x.tren} — {int(x.n)} produk ({x.pct_n:.0f}%), median {x.med:,.0f} unit/tahun, "
                        f"H2 vs H1 **{x.chg:+.0f}%**")

    cm = D["cmp"]; top3 = D["rank"].head(3); km = cm[cm.Metode == "K-Medoids"]; hd_ = cm[cm.Metode == "HDBSCAN"]
    rt = meta["runtime_s"]; fe = meta["fastdtw_err"]
    sec("Temuan utama")
    st.markdown(
        f"- **{bm} + {bd}** peringkat 1; K-Medoids menempati **{(top3.Metode == 'K-Medoids').sum()} dari 3** peringkat teratas "
        f"(ARI stabilitas {km['Stabilitas ARI'].min():.2f}–{km['Stabilitas ARI'].max():.2f}).\n"
        f"- **HDBSCAN** membuang **{100 - hd_['Cakupan (%)'].max():.0f}–{100 - hd_['Cakupan (%)'].min():.0f}%** produk sebagai noise; "
        f"pola penjualan lebih menyerupai kontinum daripada kelompok padat.\n"
        f"- **Pilihan jarak berpengaruh kecil**: selisih silhouette lintas-jarak antar jarak pada K-Medoids hanya "
        f"{km['Sil lintas-jarak'].max() - km['Sil lintas-jarak'].min():.3f}, jauh di bawah simpangan baku ARI bootstrap "
        f"({km['ARI sd'].min():.2f}–{km['ARI sd'].max():.2f}).\n"
        f"- **FastDTW tidak efisien untuk deret {meta['n_weeks']} minggu**: {rt['FastDTW'] / max(rt['DTW'], 1e-9):.0f}× lebih lambat "
        f"dari DTW eksak ber-window, galat rata-rata {fe['mean']:.1%} terhadap DTW penuh.\n"
        f"- **Keterbatasan**: silhouette tertinggi hanya {cm['Sil (jarak sendiri)'].max():.2f}, jadi segmen dibaca sebagai "
        f"*kecenderungan pola*, bukan kelompok terpisah tegas.")

# ================================================================== 2. PERBANDINGAN
with t2:
    sec("Tabel peringkat", "Urut berdasarkan rata-rata peringkat; waktu komputasi hanya pemecah seri")
    rk = flt(D["rank"])
    table(rk[["Peringkat", "Metode", "Jarak", "Parameter", "Sil lintas-jarak", "Stabilitas ARI", "Keseimbangan", "Rata-rata rank", "Waktu jarak (s)"]],
          column_config={"Sil lintas-jarak": st.column_config.ProgressColumn(min_value=0, max_value=0.4, format="%.3f"),
                         "Stabilitas ARI": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
                         "Keseimbangan": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                         "Rata-rata rank": st.column_config.NumberColumn(format="%.2f"),
                         "Waktu jarak (s)": st.column_config.NumberColumn(format="%.2f")})

    sec("Empat kriteria perbandingan", "Silhouette di jarak sendiri bias oleh skala jarak; silhouette lintas-jarak adalah pembanding utama")
    panels = [("Sil (jarak sendiri)", "Silhouette (jarak sendiri)"), ("Sil lintas-jarak", "Silhouette lintas-jarak"),
              ("Stabilitas ARI", "Stabilitas bootstrap"), ("Keseimbangan", "Keseimbangan cluster")]
    cmf = flt(D["cmp"])
    fig = make_subplots(rows=1, cols=4, subplot_titles=[p[1] for p in panels], horizontal_spacing=.06)
    for i, (col, _) in enumerate(panels, 1):
        for d in DISTS:
            s = cmf[cmf.Jarak == d].set_index("Metode").reindex([m for m in METHODS if m in cmf.Metode.values])
            fig.add_trace(go.Bar(x=s.index, y=s[col], name=d, marker_color=DIST_COL[d], showlegend=(i == 1)), row=1, col=i)
    fig.update_layout(barmode="group", bargap=.25)
    show(style(fig, 340))

    a, b = st.columns(2)
    with a:
        sec("Peta trade-off", "Kanan-atas = kualitas dan stabilitas tinggi; ukuran titik = cakupan")
        fig = px.scatter(cmf, x="Sil lintas-jarak", y="Stabilitas ARI", color="Metode", symbol="Jarak", size="Cakupan (%)",
                         size_max=22, color_discrete_sequence=CL_COL[:3], hover_data=["Parameter", "Keseimbangan"])
        fig.update_traces(marker_line_width=1, marker_line_color="#fff")
        show(style(fig, 380))
    with b:
        sec("Kesepakatan antar partisi (ARI)", "Seberapa mirip pembagian produk antar model")
        A = D["ari"]; nm = [short(c) for c in A.columns]
        fig = go.Figure(go.Heatmap(z=A.values, x=nm, y=nm, zmin=0, zmax=1, text=np.round(A.values, 2), texttemplate="%{text}",
                                   colorscale=[[0, "#F5F7FB"], [1, "#7C9CC4"]], colorbar=dict(thickness=10, title="ARI"),
                                   textfont=dict(size=10)))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 380))

    sec("Pemilihan jumlah cluster (k)", "Silhouette untuk k = 2…8 pada K-Medoids dan Hierarchical (average linkage)")
    ks = flt(D["ks"])
    fig = px.line(ks, x="k", y="Silhouette", color="Jarak", facet_col="Metode", markers=True, color_discrete_map=DIST_COL)
    fig.for_each_annotation(lambda x: x.update(text=x.text.split("=")[-1]))
    show(style(fig, 320))

    with st.expander("HDBSCAN: trade-off silhouette vs noise"):
        h = D["hd"].query("n_cluster >= 2").dropna(subset=["Silhouette inti"])
        h = h[h.Jarak.isin(sel_d)].melt(id_vars=["Jarak", "Noise (%)", "Terbaik"],
                                         value_vars=["Silhouette inti", "Silhouette terkoreksi"], var_name="Jenis", value_name="Nilai")
        fig = px.scatter(h, x="Noise (%)", y="Nilai", color="Jenis", facet_col="Jarak", color_discrete_sequence=["#C9D6E8", "#7C9CC4"],
                         opacity=.9)
        fig.for_each_annotation(lambda x: x.update(text=x.text.split("=")[-1]))
        st.caption("Silhouette inti terlihat tinggi karena titik sulit dibuang sebagai noise; silhouette terkoreksi memberi nilai 0 pada noise.")
        show(style(fig, 330))
    with st.expander("Uji sensitivitas k = 3 dan cophenetic correlation"):
        st.caption("K-Medoids / Hierarchical dengan k dipaksa 3")
        table(D["k3"])
        st.caption("Cophenetic correlation per linkage (Hierarchical)")
        table(D["coph"], column_config={c: st.column_config.NumberColumn(format="%.3f") for c in D["coph"].columns[1:]})

# ================================================================== 3. JARAK
with t3:
    fe, rt = meta["fastdtw_err"], meta["runtime_s"]
    c = st.columns(4)
    with c[0]: kpi("Galat rata-rata FastDTW", f"{fe['mean']:.1%}", "terhadap DTW penuh (eksak)", "#8DBBA6")
    with c[1]: kpi("Galat P95", f"{fe['p95']:.1%}", f"median {fe['median']:.1%}", "#E3B48A")
    with c[2]: kpi("Pasangan persis sama", f"{fe['exact_share']:.0%}", "FastDTW = DTW penuh", "#7C9CC4")
    with c[3]: kpi("FastDTW vs DTW", f"{rt['FastDTW'] / max(rt['DTW'], 1e-9):.0f}× lebih lambat", "waktu matriks jarak", "#DDA0AB")
    a, b = st.columns(2)
    with a:
        sec("Waktu komputasi matriks jarak", "Skala logaritmik, detik (hasil satu kali jalan precompute.py pada 1 core CPU)")
        fig = go.Figure(go.Bar(x=list(rt.keys()), y=list(rt.values()), marker_color=[DIST_COL[k] for k in rt],
                               text=[f"{v:.2f}s" for v in rt.values()], textposition="outside"))
        fig.update_yaxes(type="log")
        show(style(fig, 340))
    with b:
        sec("Kemiripan urutan jarak antar pasangan produk", "Korelasi Spearman; 1 = urutan kedekatan identik")
        dg = D["dagr"]
        fig = px.bar(dg.sort_values("Spearman rho"), x="Spearman rho", y="Perbandingan", orientation="h", text="Spearman rho",
                     color_discrete_sequence=["#B3A5D1"])
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(xaxis_range=[0, 1.1], yaxis_title=None)
        show(style(fig, 340))
    st.info(f"**Sifat yang diverifikasi:** DTW (window {meta['dtw_window']}) ≤ Euclidean dan FastDTW ≥ DTW penuh untuk semua pasangan. "
            "Silhouette DTW di jaraknya sendiri lebih tinggi sebagian besar karena efek skala (jarak DTW selalu lebih rapat), "
            "sehingga perbandingan antar jarak memakai silhouette lintas-jarak.")

# ================================================================== 4. PROFIL CLUSTER
with t4:
    opts = base_cols + k3_cols
    col = st.selectbox("Model", opts, index=opts.index(best_col), format_func=lambda c: label_of[c] + ("  ★ terbaik" if c == best_col else ""))
    lab = cl[col].to_numpy(); cls = sorted(c for c in np.unique(lab) if c >= 0); meds = meta["medoids"].get(col, {})
    codes = cl.Product_Code.tolist(); T = Z.shape[1]; rng = np.random.default_rng(42)

    sec("Pola deret waktu per cluster", "Garis tipis = anggota (sampel), putus-putus = rata-rata, tebal = medoid (produk representatif); z-score MA-5")
    ncol = min(len(cls), 3); nrow = -(-len(cls) // ncol)
    titles = [f"Cluster {c} · n={int((lab == c).sum())}" + (f" · medoid {meds[str(c)]}" if str(c) in meds else "") for c in cls]
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=titles, shared_yaxes=True, horizontal_spacing=.04, vertical_spacing=.16)
    for i, c in enumerate(cls):
        r_, c_ = divmod(i, ncol); mem = np.where(lab == c)[0]
        samp = rng.choice(mem, min(60, len(mem)), replace=False); xs, ys = [], []
        for j in samp: xs += list(range(T)) + [None]; ys += list(Z[j]) + [None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=ccol(c), width=.8), opacity=.2, hoverinfo="skip", showlegend=False), row=r_ + 1, col=c_ + 1)
        fig.add_trace(go.Scatter(x=list(range(T)), y=Z[mem].mean(0), mode="lines", line=dict(color=INK, width=2, dash="dash"),
                                 name="Rata-rata", showlegend=(i == 0)), row=r_ + 1, col=c_ + 1)
        if str(c) in meds:
            fig.add_trace(go.Scatter(x=list(range(T)), y=Z[codes.index(meds[str(c)])], mode="lines", line=dict(color=shade(ccol(c)), width=3),
                                     name="Medoid", showlegend=(i == 0)), row=r_ + 1, col=c_ + 1)
    fig.update_xaxes(title_text="Minggu"); fig.update_annotations(font_size=12)
    show(style(fig, 330 * nrow))

    p = profile(col); p.insert(0, "Cluster", [cname(c) for c in p.lab])
    sec("Profil segmen", "H1 = minggu 0–25, H2 = minggu 26–51 (rata-rata unit/minggu)")
    table(p[["Cluster", "tren", "n", "pct_n", "med", "pct_vol", "H1", "H2", "chg"]].rename(columns={
        "tren": "Tren", "n": "Produk", "pct_n": "% Produk", "med": "Median unit/tahun", "pct_vol": "% Volume",
        "H1": "Rata-rata H1", "H2": "Rata-rata H2", "chg": "Δ H2 vs H1 (%)"}),
        column_config={c: st.column_config.NumberColumn(format="%.1f") for c in ["% Produk", "% Volume", "Rata-rata H1", "Rata-rata H2", "Δ H2 vs H1 (%)"]}
        | {"Median unit/tahun": st.column_config.NumberColumn(format="%d")})

    a, b = st.columns(2)
    with a:
        sec("Peta produk (PCA 2D)", "Setiap titik = satu produk; arahkan kursor untuk kode produk")
        d = cl.assign(Cluster=[cname(x) for x in lab])
        fig = px.scatter(d, x="pc1", y="pc2", color="Cluster", hover_data=["Product_Code", "total_sales"],
                         color_discrete_map={cname(c): ccol(c) for c in np.unique(lab)}, opacity=.8)
        fig.update_traces(marker=dict(size=7, line=dict(width=.6, color="#fff")))
        show(style(fig, 380))
    with b:
        sec("Rata-rata penjualan mingguan H1 vs H2", "Menunjukkan arah perubahan tiap segmen")
        fig = go.Figure()
        fig.add_bar(x=p.Cluster, y=p.H1, name="H1", marker_color="#C9D6E8")
        fig.add_bar(x=p.Cluster, y=p.H2, name="H2", marker_color="#7C9CC4")
        fig.update_layout(barmode="group", bargap=.35, yaxis_title="Unit / minggu")
        show(style(fig, 380))
    st.download_button("⬇️ Unduh label cluster (model terpilih)", cl[["Product_Code", "total_sales", col]].rename(columns={col: "cluster"}).to_csv(index=False),
                       file_name="label_cluster.csv", mime="text/csv")

# ================================================================== 5. EKSPLORASI PRODUK
with t5:
    code = st.selectbox("Pilih produk", cl.Product_Code.tolist())
    i = cl.index[cl.Product_Code == code][0]; row = cl.loc[i]
    c = st.columns(4)
    with c[0]: kpi("Total setahun", f"{row.total_sales:,.0f}", "unit", "#7C9CC4")
    with c[1]: kpi("Cluster (model terbaik)", cname(row[best_col]), label_of[best_col], ccol(row[best_col]))
    with c[2]: kpi("Rata-rata H1 → H2", f"{row.mean_H1:.1f} → {row.mean_H2:.1f}", "unit / minggu", "#E3B48A")
    with c[3]: kpi("Perubahan H2 vs H1", f"{100 * (row.mean_H2 / row.mean_H1 - 1):+.0f}%", "", "#B3A5D1")

    sec("Deret mingguan", "Batang = unit terjual asli; garis = hasil penghalusan MA-5 (z-score, sumbu kanan)")
    wk = list(range(52)); fig = go.Figure()
    fig.add_bar(x=wk, y=D["raw"].loc[code].to_numpy(), name="Unit terjual", marker_color="#D5DEEC")
    fig.add_scatter(x=wk, y=Z[i], name="MA-5 (z-score)", yaxis="y2", line=dict(color=ccol(row[best_col]), width=3))
    fig.update_layout(yaxis=dict(title="Unit"), yaxis2=dict(overlaying="y", side="right", showgrid=False, title="z-score"), xaxis_title="Minggu")
    show(style(fig, 340))

    a, b = st.columns([1, 1.3])
    with a:
        sec("Keanggotaan di 9 model")
        table(pd.DataFrame([{"Metode": c.split("|")[0], "Jarak": c.split("|")[1], "Parameter": meta["params"][c], "Cluster": cname(row[c])} for c in base_cols]))
    with b:
        sec("Daftar produk", "Filter berdasarkan cluster model terbaik")
        pick = st.multiselect("Cluster", sorted(cl[best_col].unique()), default=sorted(cl[best_col].unique()), format_func=cname)
        t = cl[cl[best_col].isin(pick)][["Product_Code", "total_sales", "mean_H1", "mean_H2", best_col]].rename(
            columns={"total_sales": "Total unit", "mean_H1": "H1", "mean_H2": "H2", best_col: "Cluster"})
        table(t.sort_values("Total unit", ascending=False), height=300)
        st.download_button("⬇️ Unduh tabel (CSV)", t.to_csv(index=False), file_name="produk_cluster.csv", mime="text/csv")

st.caption("Sumber: Sales_transaction.csv · hasil dari precompute.py (identik dengan Clustering.ipynb) · dibangun dengan Streamlit & Plotly")

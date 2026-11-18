"""Gabungkan hasil clustering menjadi satu index.html mandiri untuk GitHub Pages.
Jalankan dari folder repo:  python build_site.py"""
import json
from pathlib import Path
import pandas as pd

R = Path("data/results")
rd = lambda f, **k: pd.read_csv(R / f, **k)
cl, z, ari = rd("clusters.csv"), rd("series_z.csv"), rd("ari_matrix.csv", index_col=0)
raw = pd.read_csv("data/Sales_transaction.csv", sep=None, engine="python").set_index("Product_Code")
codes = cl.Product_Code.tolist()
recs = lambda f: json.loads(rd(f).round(4).to_json(orient="records"))
D = dict(meta=json.load(open(R / "meta.json", encoding="utf-8")), cmp=recs("comparison.csv"), rank=recs("ranking.csv"),
         ks=recs("k_sweep.csv"), dagr=recs("distance_agreement.csv"), ari=dict(n=list(ari.columns), m=ari.values.round(3).tolist()),
         hd=recs("hdbscan_grid.csv"), coph=recs("cophenetic.csv"), k3=recs("k3_sensitivity.csv"),
         prod=dict(codes=codes, pc1=cl.pc1.round(3).tolist(), pc2=cl.pc2.round(3).tolist(), tot=cl.total_sales.astype(int).tolist(), h1=cl.mean_H1.round(2).tolist(), h2=cl.mean_H2.round(2).tolist(),
                   lab={c: cl[c].astype(int).tolist() for c in cl.columns if "|" in c},
                   z=z.drop(columns="Product_Code").round(2).values.tolist(),
                   raw=raw.loc[codes, [f"W{i}" for i in range(52)]].astype(int).values.tolist()))
html = Path("template.html").read_text(encoding="utf-8").replace("__DATA__", json.dumps(D, ensure_ascii=False, separators=(",", ":")))
Path("index.html").write_text(html, encoding="utf-8")
print("index.html:", round(len(html) / 1024), "KB")

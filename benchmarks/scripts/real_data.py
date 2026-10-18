"""Real-data analysis: every pairwise-test method on same-section spatial transcriptomics + metabolomics data.

There is no ground truth on real tissue, so methods are compared on
- calibration: the metabolite map is rotated/reflected onto the gene spots (true gene–metabolite correspondence is
  destroyed, spatial autocorrelation of both sides is kept); the fraction of pairs with p < 0.05 is the empirical
  false-positive rate at nominal 5%;
- discoveries: pairs significant at BH FDR 5% on the real data;
- reproducibility: overlap of the top-ranked pairs between sections of the same group (same features in all of them).

Data sets (see the project data list): SpatialMETA ccRCC (Tian et al., Nat Commun 2025; Zenodo 14986870), five
tumours already aligned to Visium spots by the authors.

Usage: python benchmarks/scripts/real_data.py <data dir> [--group ccrcc] [--methods A,B,...]
Writes results/real_<group>.json and per-section p-value matrices to benchmarks/cache/real/ (not committed).
"""
import argparse, glob, json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # benchmarks/
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from scipy.spatial import cKDTree
from respair import core as sj
from respair.full import fit_full
from compare import RUNNERS, _neglog, _prep

CACHE = os.path.join(ROOT, "cache", "real")
N_GENES, N_METS, MIN_DETECT, TOP_K = 1000, 300, 0.10, 1000


def _respair_full(d):
    fit = fit_full(d["G"], d["M"], d["coords"], Z=d["Z"], neg_genes=d["neg_genes"], seed=0)
    return {"ReSPAIR (full)": _neglog(fit["p"])}


ALL_RUNNERS = RUNNERS + [(["ReSPAIR (full)"], _respair_full)]
P_METHODS = [m for names, _ in ALL_RUNNERS for m in names if m != "Lee's L"]  # Lee's L has no test


def hex_coords(row, col):
    """Visium array indices -> coordinates in units of the spot spacing."""
    return np.column_stack([np.asarray(col, float) / 2, np.asarray(row, float) * np.sqrt(3) / 2])


def load_ccrcc(data_dir):
    import anndata as ad
    from scipy.sparse import issparse
    out = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "adata_joint_*_T_raw.h5ad"))):
        a = ad.read_h5ad(f)
        X = a.X.toarray() if issparse(a.X) else np.asarray(a.X)
        st = (a.var["type"] == "ST").to_numpy()
        name = os.path.basename(f).split("_")[2]
        out[name] = {"G": X[:, st], "M": X[:, ~st], "genes": a.var_names[st].to_numpy(),
                     "mz": a.var_names[~st].to_numpy(), "coords": hex_coords(a.obs["array_row"], a.obs["array_col"])}
    return out


def tic_normalise(M):
    tic = M.sum(1, keepdims=True)
    return M / np.where(tic > 0, tic, 1) * np.median(tic[tic > 0])


def select_shared(sections, key, names, n_top, transform):
    """Features detected in >= MIN_DETECT of spots in every section, ranked by mean variance rank of transform(X)."""
    common = set(sections[0][names])
    for s in sections[1:]:
        common &= set(s[names])
    common = np.array(sorted(common))
    ranks, ok = [], np.ones(len(common), bool)
    for s in sections:
        idx = {v: i for i, v in enumerate(s[names])}
        X = s[key][:, [idx[v] for v in common]]
        ok &= (X > 0).mean(0) >= MIN_DETECT
        v = transform(X).var(0)
        ranks.append(np.argsort(np.argsort(-v)))
    score = np.mean(ranks, 0)
    score[~ok] = np.inf
    return common[np.argsort(score)[:min(n_top, ok.sum())]]


def log_cpm(G):
    return np.log1p(G / np.maximum(G.sum(1, keepdims=True), 1) * 1e4)


def subset(s, genes, mz):
    gi = {v: i for i, v in enumerate(s["genes"])}
    mi = {v: i for i, v in enumerate(s["mz"])}
    G = s["G"][:, [gi[g] for g in genes]]
    M = tic_normalise(s["M"])[:, [mi[m] for m in mz]]
    keep = (G.sum(1) > 0) & (M.sum(1) > 0)
    return {"G": G[keep], "M": M[keep], "coords": s["coords"][keep], "Z": None, "neg_genes": None}


def rotated_null(d, min_overlap=0.5):
    """Map the metabolite map, rotated/reflected about the tissue centre, onto the gene spots.

    Tries the 7 non-identity symmetries of the square and keeps the one with the largest overlap; spots without a
    rotated metabolite spot within half a spacing are dropped.
    """
    c = d["coords"] - d["coords"].mean(0)
    best = None
    for refl in (1, -1):
        for ang in (0, 90, 180, 270):
            if refl == 1 and ang == 0:
                continue
            t = np.deg2rad(ang)
            R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) @ np.diag([refl, 1])
            dist, idx = cKDTree(c @ R.T).query(c)
            ok = dist < 0.5
            if best is None or ok.sum() > best[0].sum():
                best = (ok, idx, ang, refl)
    ok, idx, ang, refl = best
    if ok.mean() < min_overlap:
        return None
    return {"G": d["G"][ok], "M": d["M"][idx[ok]], "coords": d["coords"][ok], "Z": None, "neg_genes": None,
            "transform": f"rotate {ang} deg, reflect {refl == -1}", "overlap": float(ok.mean())}


def run_methods(d, methods, tag):
    out, secs = {}, {}
    for names, fn in ALL_RUNNERS:
        names = [m for m in names if m in methods]
        if not names:
            continue
        paths = {m: os.path.join(CACHE, "".join(c if c.isalnum() else "_" for c in m) + tag + ".npy") for m in names}
        if not all(os.path.exists(p) for p in paths.values()):
            t = time.time()
            res = fn(d)
            el = time.time() - t
            print(f"  {tag} {'/'.join(names)}: {el:.0f}s", flush=True)
            os.makedirs(CACHE, exist_ok=True)
            for m in names:
                np.save(paths[m], res[m])
                np.save(paths[m].replace(".npy", ".sec.npy"), el)
        for m in names:
            out[m] = np.load(paths[m])
            sp = paths[m].replace(".npy", ".sec.npy")
            secs[m] = float(np.load(sp)) if os.path.exists(sp) else float("nan")
    return out, secs


def bh_count(neglogp, alpha=0.05):
    return int(sj.bh(10 ** -neglogp, alpha).sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--group", default="ccrcc")
    ap.add_argument("--methods", default=",".join(m for names, _ in ALL_RUNNERS for m in names))
    a = ap.parse_args()
    methods = a.methods.split(",")
    secs_all = {"ccrcc": load_ccrcc}[a.group](a.data_dir)
    names = list(secs_all)
    genes = select_shared([secs_all[n] for n in names], "G", "genes", N_GENES, log_cpm)
    mz = select_shared([secs_all[n] for n in names], "M", "mz", N_METS, lambda M: np.log1p(tic_normalise(M)))
    print(f"{a.group}: {len(names)} sections, {len(genes)} genes, {len(mz)} metabolites", flush=True)

    result = {"group": a.group, "genes": len(genes), "metabolites": len(mz), "sections": {}}
    ranks = {}
    for n in names:
        d = subset(secs_all[n], genes, mz)
        null = rotated_null(d)
        print(n, f"{len(d['coords'])} spots; null: {null and null['transform']}, overlap {null and null['overlap']:.2f}",
              flush=True)
        S, T = run_methods(d, methods, f"_{a.group}_{n}")
        row = {"spots": len(d["coords"]), "null transform": null and null["transform"],
               "null overlap": null and null["overlap"], "methods": {}}
        if null is not None:
            Sn, _ = run_methods(null, methods, f"_{a.group}_{n}_null")
        for m in S:
            r = {"seconds": T[m]}
            if m in P_METHODS:
                r["BH5% hits"] = bh_count(S[m])
                if null is not None:
                    r["null FPR at p<0.05"] = float((Sn[m] > -np.log10(0.05)).mean())
                    r["null BH5% hits"] = bh_count(Sn[m])
            row["methods"][m] = r
            ranks.setdefault(m, {})[n] = np.argsort(-S[m].ravel(), kind="stable")[:TOP_K]
        result["sections"][n] = row
        print(" ", {m: (v.get("null FPR at p<0.05"), v.get("BH5% hits")) for m, v in row["methods"].items()}, flush=True)

    # reproducibility: mean overlap of the top-K pairs over all section pairs
    rep = {}
    for m, r in ranks.items():
        ov = [len(np.intersect1d(r[x], r[y])) / TOP_K for i, x in enumerate(names) for y in names[i + 1:]]
        rep[m] = {"mean top-%d overlap" % TOP_K: float(np.mean(ov)), "min": float(np.min(ov)), "max": float(np.max(ov))}
    result["reproducibility"] = rep
    result["chance top-K overlap"] = TOP_K / (len(genes) * len(mz))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", f"real_{a.group}.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(rep, indent=1))

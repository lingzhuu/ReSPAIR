"""Full vs simplified ReSPAIR on the scenarios of misspec.py and the semi-synthetic data of semisim.py.

Usage:
  python benchmarks/scripts/full_vs_simple_external.py mis  [n_seeds] [link strength]
  python benchmarks/scripts/full_vs_simple_external.py semi [n_seeds] <aligned.npz>
Writes results/full_vs_simple_misspec.json or results/full_vs_simple_semisim.json; each metric pair is
[AUC, power at 5% false-positive rate].
"""
import json, os, sys, time
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import respair
from respair.full import fit_full
from respair.simulate import simulate, semi_synthetic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # benchmarks/
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from misspec import SCENARIOS as MIS
from semisim import SCENARIOS as SEMI

def rs(p, pos, neg):
    s = -np.log10(np.clip(p, 1e-300, 1)); y = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
    sc = np.r_[s[pos], s[neg]]; f, t, _ = roc_curve(y, sc)
    return [float(roc_auc_score(y, sc)), float(np.interp(0.05, f, t))]

which = sys.argv[1]
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
link = float(sys.argv[3]) if which == "mis" and len(sys.argv) > 3 else 0.5
out = os.path.join(ROOT, "results", "full_vs_simple_misspec.json" if which == "mis" else "full_vs_simple_semisim.json")
res = {}
src = np.load(sys.argv[3]) if which == "semi" else None
for name, kw in (MIS if which == "mis" else SEMI).items():
    rows = {"simplified": [], "full": []}
    for seed in range(n_seeds):
        if which == "mis":
            d = simulate(seed=seed, link_strength=link, **kw)
            conf = d["conf_pair"] & ~d["truth"]
        else:
            d = semi_synthetic(src["G"], src["M"], src["coords"], seed=seed, **kw)
            conf = d["conf_pair"]
        t = time.time()
        a = respair.fit(d["G"], d["M"], d["coords"], Z=d["Z"], neg_genes=d["neg_genes"])
        b = fit_full(d["G"], d["M"], d["coords"], Z=d["Z"], neg_genes=d["neg_genes"])
        for m, r in [("simplified", a), ("full", b)]:
            rows[m].append({"all pairs": rs(r["p"], d["truth"], ~d["truth"]), "confounded pairs": rs(r["p"], d["truth"], conf),
                            "FPR at p<0.05": float((r["p"][~d["truth"]] < 0.05).mean()), "K": int(r["K"])})
        print(name, seed, f"{time.time()-t:.0f}s", rows["simplified"][-1], rows["full"][-1], flush=True)
    res[name] = {m: {k: (np.mean([x[k] for x in v], 0).tolist()) for k in v[0]} for m, v in rows.items()}
    res[name]["per_seed"] = rows
    json.dump({"link": link if which == "mis" else "per scenario", "seeds": n_seeds, "results": res}, open(out, "w"), indent=1)

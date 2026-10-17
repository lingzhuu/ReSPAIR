"""Sweep confounder strength and link strength; writes results/sweep.json.

Usage: python benchmarks/scripts/sweep_sim.py [n_seeds per setting]
"""
import sys, os, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # benchmarks/
from respair.simulate import simulate
from run_sim import evaluate, all_pvals, MAIN

CONF = [0.0, 0.3, 0.6, 1.0]
LINK = [0.1, 0.15, 0.2, 0.3, 0.5]


def one(seed, **kw):
    d = simulate(seed=seed, **kw)
    P, _, _ = all_pvals(d)
    return {m: evaluate(truth=d["truth"], conf_pair=d["conf_pair"], p=P[m]) for m in MAIN + ["Z-only + spectral test"]}


def avg(runs):
    return {m: {k: float(np.mean([r[m][k] for r in runs])) for k in runs[0][m]} for m in runs[0]}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    out = {"confounder strength": {}, "link strength": {}}
    for c in CONF:
        out["confounder strength"][str(c)] = avg([one(s, conf_strength=c) for s in range(n)])
        print("conf", c, flush=True)
    for w in LINK:
        out["link strength"][str(w)] = avg([one(s, link_strength=w) for s in range(n)])
        print("link", w, flush=True)
    with open(os.path.join(ROOT, "results", "sweep.json"), "w") as f:
        json.dump(out, f, indent=1)

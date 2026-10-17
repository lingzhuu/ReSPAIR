"""扫描混杂强度和联系强度，结果存到 results/sweep.json。

用法：python3 scripts/sweep_sim.py [每个设置的种子数]
"""
import sys, os, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from respair.simulate import simulate
from run_sim import evaluate, all_pvals, MAIN

CONF = [0.0, 0.3, 0.6, 1.0]
LINK = [0.1, 0.15, 0.2, 0.3, 0.5]


def one(seed, **kw):
    d = simulate(seed=seed, **kw)
    P, _, _ = all_pvals(d)
    return {m: evaluate(truth=d["truth"], conf_pair=d["conf_pair"], p=P[m]) for m in MAIN + ["只扣已知结构+空间检验"]}


def avg(runs):
    return {m: {k: float(np.mean([r[m][k] for r in runs])) for k in runs[0][m]} for m in runs[0]}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    out = {"混杂强度": {}, "联系强度": {}}
    for c in CONF:
        out["混杂强度"][str(c)] = avg([one(s, conf_strength=c) for s in range(n)])
        print("conf", c, flush=True)
    for w in LINK:
        out["联系强度"][str(w)] = avg([one(s, link_strength=w) for s in range(n)])
        print("link", w, flush=True)
    with open(os.path.join(ROOT, "results", "sweep.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

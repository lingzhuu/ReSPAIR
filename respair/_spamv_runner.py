"""在 SpaMV 的虚拟环境里运行：读入 npz，训练 SpaMV，输出 基因 × 代谢物 的共享主题载荷乘积。"""
import sys
import numpy as np
import anndata as ad
import torch
from SpaMV.spamv import SpaMV

inp, out, epochs, zs_dim, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
torch.manual_seed(seed); np.random.seed(seed)
d = np.load(inp)
G, M, coords = d["G"], d["M"], d["coords"]
n = len(coords)
a1, a2 = ad.AnnData(G), ad.AnnData(M - M.min())
for a in (a1, a2):
    a.obs_names = [f"s{i}" for i in range(n)]
    a.obsm["spatial"] = coords
a1.var_names = [f"g{j}" for j in range(G.shape[1])]
a2.var_names = [f"m{k}" for k in range(M.shape[1])]
m = SpaMV([a1, a2], interpretable=True, zs_dim=zs_dim, zp_dims=[5, 5], recon_types=["nb", "gauss"],
          omics_names=["RNA", "MSI"], device=torch.device("cpu"), max_epochs_stage1=epochs,
          max_epochs_stage2=epochs, early_stopping=True, patience=100)
m.train()
B = m.model.get_feature_by_topic()
Br = np.asarray(B["RNA"])[:, :zs_dim]
Bm = np.asarray(B["MSI"])[:, :zs_dim]
np.save(out, Br @ Bm.T)

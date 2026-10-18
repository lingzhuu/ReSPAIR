"""Pack the authors' aligned Vicari et al. 2024 sections (exported by export_vicari_seurat.R) into one .npz each.

Spot coordinates come from the Space Ranger tissue_positions_list.csv of each section (hex grid, unit = spot spacing),
annotations from its lesion.csv / region.csv / dopamine.csv when present.

Usage: python benchmarks/scripts/prepare_vicari.py <export dir> <vicari2024/sma dir> <out dir>
"""
import glob, os, sys
import numpy as np
import pandas as pd
from scipy.io import mmread

if __name__ == "__main__":
    exp, sma, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    for d in sorted(glob.glob(os.path.join(exp, "*", "*"))):
        sec = os.path.basename(d)
        outs = [p for p in glob.glob(os.path.join(sma, "*", sec, "output_data", sec + "*", "outs"))][0]
        spots = pd.read_csv(os.path.join(d, "spots.csv"))
        bc = spots["barcode"].str.replace(r"_\d+$", "", regex=True).to_numpy()
        pos = pd.read_csv(os.path.join(outs, "spatial", "tissue_positions_list.csv"), header=None,
                          names=["barcode", "in_tissue", "row", "col", "pxl_row", "pxl_col"]).set_index("barcode").loc[bc]
        coords = np.column_stack([pos["col"] / 2, pos["row"] * np.sqrt(3) / 2])
        G = mmread(os.path.join(d, "rna_counts.mtx")).T.toarray()
        M = mmread(os.path.join(d, "msi_counts.mtx")).T.toarray()
        ann = {}
        for name in ["lesion", "region", "dopamine"]:
            f = os.path.join(outs, name + ".csv")
            if os.path.exists(f):
                a = pd.read_csv(f, index_col=0).iloc[:, 0]
                ann[name] = np.asarray(a.reindex(bc).astype(str).to_numpy(), dtype=str)
        np.savez_compressed(os.path.join(out, sec + ".npz"), G=G.astype(np.int32), M=M.astype(np.float32),
                            coords=coords, genes=np.loadtxt(os.path.join(d, "genes.txt"), dtype=str, delimiter="\t"),
                            mz=np.loadtxt(os.path.join(d, "mz.txt"), dtype=str), barcodes=bc, **ann)
        print(sec, G.shape, M.shape, {k: dict(zip(*np.unique(v, return_counts=True))) for k, v in ann.items()})

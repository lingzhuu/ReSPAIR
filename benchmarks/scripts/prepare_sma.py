"""Align one SMA sample (Vicari et al., Nat Biotechnol 2024; Visium + MALDI-MSI on the same section) to Visium spots.

Input: the copy of sample V11L12-038_D1 in the MAGPIE tutorial (github.com/Core-Bioinformatics/MAGPIE,
docs/tutorial/input/V11L12-038_D1): Space Ranger output, MSI intensities and pixel coordinates, and 10 landmark pairs.
Steps: affine map MSI → Visium hires image from the landmarks (least squares), scale to full resolution, assign each MSI
pixel to the nearest in-tissue Visium spot within half the spot spacing, and average the pixels of each spot.

Usage: python benchmarks/scripts/prepare_sma.py <input dir> <output .npz>
"""
import json, os, sys
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.spatial import cKDTree


def read_10x_h5(path):
    with h5py.File(path) as f:
        g = f["matrix"]
        X = csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=g["shape"][:]).T.tocsr()  # spots × genes
        barcodes = g["barcodes"][:].astype(str)
        genes = g["features"]["name"][:].astype(str)
        ftype = g["features"]["feature_type"][:].astype(str)
    keep = ftype == "Gene Expression"
    return X[:, keep], barcodes, genes[keep]


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    X, barcodes, genes = read_10x_h5(os.path.join(src, "visium", "filtered_feature_bc_matrix.h5"))
    pos = pd.read_csv(os.path.join(src, "visium", "spatial", "tissue_positions_list.csv"), header=None,
                      names=["barcode", "in_tissue", "row", "col", "pxl_row", "pxl_col"]).set_index("barcode")
    pos = pos.loc[barcodes]
    sf = json.load(open(os.path.join(src, "visium", "spatial", "scalefactors_json.json")))
    spot_xy = pos[["pxl_col", "pxl_row"]].to_numpy(float)  # full-resolution pixels

    lm = pd.read_csv(os.path.join(src, "landmarks_noHE.csv"))
    A = np.column_stack([lm.X_left, lm.Y_left, np.ones(len(lm))])
    T, *_ = np.linalg.lstsq(A, lm[["X_right", "Y_right"]].to_numpy(), rcond=None)
    resid = A @ T - lm[["X_right", "Y_right"]].to_numpy()
    print("landmark RMS error (hires px):", np.sqrt((resid ** 2).sum(1).mean()).round(2))

    meta = pd.read_csv(os.path.join(src, "msi", "MSI_metadata.csv")).set_index("spot_id")
    msi = pd.read_csv(os.path.join(src, "msi", "MSI_intensities.csv")).set_index("spot_id")
    meta = meta.loc[msi.index]
    msi_xy = np.column_stack([meta.x, meta.y, np.ones(len(meta))]) @ T / sf["tissue_hires_scalef"]

    tree = cKDTree(spot_xy)
    spacing = np.sort(tree.query(spot_xy, 2)[0][:, 1])[len(spot_xy) // 2]  # median nearest-neighbour distance
    dist, idx = tree.query(msi_xy)
    ok = dist < spacing / 2
    print(f"spot spacing {spacing:.0f} px; MSI pixels assigned {ok.sum()} / {len(ok)}")
    n_spot = len(spot_xy)
    sums = np.zeros((n_spot, msi.shape[1])); cnt = np.bincount(idx[ok], minlength=n_spot)
    np.add.at(sums, idx[ok], msi.to_numpy(float)[ok])
    both = cnt > 0
    M = sums[both] / cnt[both][:, None]
    G = X[both].toarray()
    grid = pos[["col", "row"]].to_numpy(float)[both]
    coords = np.column_stack([grid[:, 0] / 2, grid[:, 1] * np.sqrt(3) / 2])  # Visium hex grid, unit = spot spacing
    print(f"spots with both modalities: {both.sum()}; genes {G.shape[1]}; m/z features {M.shape[1]}")
    np.savez_compressed(out, G=G.astype(np.int32), M=M.astype(np.float32), coords=coords, genes=genes,
                        mz=np.array([c.replace("mz-", "") for c in msi.columns]), barcodes=barcodes[both],
                        pixels_per_spot=cnt[both])

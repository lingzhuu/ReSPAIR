# Export the authors' aligned Visium + MALDI-MSI objects of Vicari et al. (Nat Biotechnol 2024) to plain files.
# Input: se.multi.list (9 mouse sections) or hPDStr.multi.list (4 human striatum sections) from misc.zip in
# Mendeley Data 10.17632/w7nw4km7xd.1. In these Seurat objects the MSI assay is already on the Visium spots
# (the authors' manual STUtility alignment, one nearest MSI pixel per spot).
# Usage: Rscript export_vicari_seurat.R <list.rds> <out dir>   (needs R package SeuratObject)
suppressMessages({library(SeuratObject); library(Matrix)})
a <- commandArgs(TRUE); x <- readRDS(a[1]); out <- a[2]
for (i in seq_along(x)) {
  o <- x[[i]]; sid <- unique(o@meta.data$sample_id); cat(sid, ncol(o), "\n")
  d <- file.path(out, sid); dir.create(d, recursive=TRUE, showWarnings=FALSE)
  g <- LayerData(o, assay="RNA", layer="counts"); m <- LayerData(o, assay="MSI", layer="counts")
  writeMM(as(g, "CsparseMatrix"), file.path(d, "rna_counts.mtx")); writeLines(rownames(g), file.path(d, "genes.txt"))
  writeMM(as(m, "CsparseMatrix"), file.path(d, "msi_counts.mtx")); writeLines(rownames(m), file.path(d, "mz.txt"))
  md <- o@meta.data; md$barcode <- rownames(md)
  st <- o@tools$Staffli@meta.data; md <- cbind(md, st[rownames(md), intersect(c("x","y","pixel_x","pixel_y","warped_x","warped_y","sample"), colnames(st)), drop=FALSE])
  write.csv(md, file.path(d, "spots.csv"), row.names=FALSE)
}

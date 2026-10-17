"""ReSPAIR: pair-specific gene–metabolite link test for spatial transcriptomics + metabolomics (simplified and full estimators)."""
from importlib.metadata import PackageNotFoundError, version

from .core import bh, fit, lambda_gc, naive_z, pearson_residuals, pvals, spatial_basis, spectral_z
from .full import fit_full

try:
    __version__ = version("respair")
except PackageNotFoundError:  # imported from a source tree without installing
    __version__ = "0.0.0"

__all__ = ["fit", "fit_full", "spectral_z", "naive_z", "pvals", "bh", "lambda_gc", "spatial_basis", "pearson_residuals"]

"""ReSPAIR: pair-specific gene–metabolite link test for spatial transcriptomics + metabolomics."""
from importlib.metadata import PackageNotFoundError, version

from .core import bh, pvals
from .estimator import fit

try:
    __version__ = version("respair")
except PackageNotFoundError:  # imported from a source tree without installing
    __version__ = "0.0.0"

__all__ = ["fit", "pvals", "bh"]

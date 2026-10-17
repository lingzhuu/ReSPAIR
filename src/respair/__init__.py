"""ReSPAIR：空间转录组与空间代谢组的成对联系检验（简化版）。"""
from importlib.metadata import PackageNotFoundError, version

from .core import bh, fit, lambda_gc, naive_z, pearson_residuals, pvals, spatial_basis, spectral_z

try:
    __version__ = version("respair")
except PackageNotFoundError:  # 未安装、直接从源码目录导入
    __version__ = "0.0.0"

__all__ = ["fit", "spectral_z", "naive_z", "pvals", "bh", "lambda_gc", "spatial_basis", "pearson_residuals"]

"""
XOKDE++: Online Kernel Density Estimation Module

This subpackage provides the core classes for performing online kernel
density estimation, ported from the C++ XOKDE++ library.

J. Ferreira, D. M. de Matos, and R. Ribeiro,
“Fast and Extensible Online Multivariate Kernel Density Estimation,”
2016, arXiv. doi: 10.48550/ARXIV.1606.02608.
"""

from .okde_full import OKDEFull
from .okde_diagonal import OKDEDiagonal
from .gaussian_full import GaussianFull
from .gaussian_diagonal import GaussianDiagonal
from .explanation_full import ExplanationFull
from .explanation_diagonal import ExplanationDiagonal
from .mixture import Mixture

__all__ = [
    "OKDEFull",
    "OKDEDiagonal",
    "GaussianFull",
    "GaussianDiagonal",
    "ExplanationFull",
    "ExplanationDiagonal",
    "Mixture",
]

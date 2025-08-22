"""
XOKDE++: Online Kernel Density Estimation Module

This subpackage is a derivative work of XOKDE++ (C++), originally developed by:
(C) Copyright 2014-2016, Jaime Ferreira, David Martins de Matos, Ricardo Ribeiro
Spoken Language Systems Lab, INESC ID, IST/Universidade de Lisboa

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

See the LICENSE file for more details.
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

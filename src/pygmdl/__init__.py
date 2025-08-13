"""
pygmdl: A Python implementation of the Gaussian Mixture Descriptors Learner (GMDL).

This library provides an online classifier based on the Minimum Description
Length (MDL) principle, using Online Kernel Density Estimators
for density modeling.

FREITAS, B. L. de. Aprendiz de Descritores de Mistura Gaussiana. 2017. 90 f.
Dissertacao (Mestrado em Ciencia da Computacao) — Universidade Federal de Sao
Carlos, Sorocaba, 2017.

Paper: https://repositorio.ufscar.br/items/09ccfb95-d5dd-47ae-a364-af8641de6e2d
C++ code from Freitas: https://github.com/brenolf/gmdl/tree/master
"""

from .core import GMDL
from . import kde
from . import dataset_utils
from .dataset_utils import (
    load_from_file,
    load_from_stream,
    load_online_stream,
    SampleType,
)

__all__ = [
    "GMDL",
    "kde",
    "dataset_utils",
    "load_from_file",
    "load_from_stream",
    "load_online_stream",
    "SampleType",
]

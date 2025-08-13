"""
pygmdl: A Python implementation of the Gaussian Mixture Descriptors Learner (GMDL).

This library provides an online classifier based on the Minimum Description
Length (MDL) principle, using Online Kernel Density Estimators
for density modeling.
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

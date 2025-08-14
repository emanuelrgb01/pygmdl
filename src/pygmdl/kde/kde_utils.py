import numpy as np
from typing import List

from . import constants as constants

VectorType = np.ndarray
MatrixType = np.ndarray
DataType = np.float64


def allclose(
    a: MatrixType,
    b: MatrixType,
    rtol: float = constants.EPSILON,
    atol: float = constants.EPSILON12,
) -> bool:
    """
    Checks if two matrices are approximately equal, element-wise.
    This is equivalent to numpy's allclose function.
    """
    return np.allclose(a, b, rtol=rtol, atol=atol)


def select_subset_matrix(src: MatrixType, ok_id: List[int]) -> MatrixType:
    """
    Selects a subset of a matrix based on a list of indices.
    """
    return src[np.ix_(ok_id, ok_id)]


def select_subset_vector(src: VectorType, ok_id: List[int]) -> VectorType:
    """
    Selects a subset of a vector based on a list of indices.
    """
    return src[ok_id]


def vector_max_val_and_idx(vector: List[DataType]) -> tuple[DataType, int]:
    """
    Finds the maximum value and its index in a list.
    """
    if not vector:
        return (None, -1)
    max_val = max(vector)
    idx = vector.index(max_val)
    return max_val, idx


def double_equals(a: float, b: float, precision: float = constants.PRECISION) -> bool:
    """
    Checks if two floating-point numbers are approximately equal.
    """
    return abs(a - b) < precision  #


def ft_c_f_diag_vec(C: VectorType, F: VectorType) -> DataType:
    """
    Calculates F' * C * F, where C is a diagonal matrix represented by a vector.
    """
    # Equivalent to (F.array() * C.array() * F.array()).sum()
    return np.sum(F * C * F)


def ft_c_f_full_vec(cov: MatrixType, F: VectorType) -> DataType:
    """
    Calculates F' * cov * F, where cov is a full matrix.
    """
    return F.T @ cov @ F


def ft_c_f_diag_mat(C: VectorType, F: MatrixType) -> VectorType:
    """
    Calculates the diagonal of F' * C * F, where C is a diagonal matrix represented by a vector.
    """
    return (F**2).T @ C


def ft_c_f_full_mat(cov: MatrixType, F: MatrixType) -> MatrixType:
    """
    Calculates F' * cov * F, where cov and F are full matrices.
    """
    return F.T @ cov @ F


def f_c_ft(cov: MatrixType, F: MatrixType) -> MatrixType:
    """
    Calculates F * cov * F'.
    """
    return F @ cov @ F.T

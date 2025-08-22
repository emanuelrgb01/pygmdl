import numpy as np
from typing import Tuple, List, Union

from ._gaussian_base import GaussianBase
from .mixture import Mixture
from .kde_utils import VectorType, MatrixType
from .gaussian_full import GaussianFull
from . import constants as constants


class GaussianDiagonal(GaussianBase[VectorType]):
    """Gaussian distribution with a diagonal covariance matrix."""

    def __init__(
        self, dims: int, mean: VectorType = None, covariance: VectorType = None
    ):
        super().__init__(dims)
        self.mean = mean if mean is not None else np.zeros(dims)
        self.covariance = covariance if covariance is not None else np.ones(dims)

    def _mahalanobis_distance_sq(self, delta: VectorType) -> float:
        """Calculates delta' * inv(cov) * delta for a diagonal covariance matrix."""
        return np.sum(delta * self.covariance_inverse() * delta)

    def _compute_determinant_and_inverse(self) -> None:
        """
        Computes the inverse and log-determinant of a diagonal covariance matrix.
        """
        if not self._inverse_is_dirty:
            return

        cov = self._covariance.copy()

        cov[cov <= constants.MIN_BANDWIDTH] = constants.MIN_BANDWIDTH

        self._is_invertible = True
        self._covariance_log_determinant = np.sum(np.log(cov))
        self._covariance_inverse = 1.0 / cov

        self._covariance = cov
        self._inverse_is_dirty = False

    @classmethod
    def from_mixture(cls, mix: Mixture, is_flat: bool = False) -> "GaussianDiagonal":
        """Creates a diagonal Gaussian by moment-matching a mixture."""
        if len(mix) == 0:
            raise ValueError("Cannot create Gaussian from an empty mixture.")

        dims = mix.dims
        total_weight = mix.sum_of_weights()
        if np.isclose(total_weight, 0):
            raise ValueError("Total weight of mixture is zero.")

        new_mean = (
            np.sum([c.mean * w for c, w in zip(mix.components, mix.weights)], axis=0)
            / total_weight
        )

        new_cov_diag = np.zeros(dims)
        for c, w in zip(mix.components, mix.weights):
            cov_diag = c.covariance if c.covariance.ndim == 1 else np.diag(c.covariance)
            second_moment_diag = cov_diag + c.mean**2
            new_cov_diag += w * second_moment_diag
        new_cov_diag /= total_weight

        new_cov_diag -= new_mean**2

        return cls(dims, new_mean, new_cov_diag)

    @classmethod
    def from_merge(cls, mix: Mixture, bandwidth: MatrixType) -> "GaussianDiagonal":
        """
        Creates a Gaussian by merging a mixture of other Gaussians.
        The bandwidth parameter is ignored, mirroring the C++ implementation.
        """
        return cls.from_mixture(mix, is_flat=False)

    def KL_divergence(self, other: GaussianBase) -> float:
        if not isinstance(other, GaussianDiagonal):
            raise TypeError("KL Divergence requires another GaussianDiagonal instance.")

        log_det_ratio = (
            other.covariance_log_determinant() - self.covariance_log_determinant()
        )
        trace_term = np.sum(other.covariance_inverse() * self.covariance)
        mean_diff = self.mean - other.mean
        mahalanobis_term = np.sum(mean_diff**2 * other.covariance_inverse())

        kl_div = 0.5 * (log_det_ratio + trace_term + mahalanobis_term - self._dims)
        return kl_div if not np.isnan(kl_div) else np.inf

    def decompose_in_two(self) -> Tuple["GaussianDiagonal", "GaussianDiagonal"]:
        """Decomposes this Gaussian into two, preserving the first two moments."""
        full_gaussian = GaussianFull(self.dims, self.mean, np.diag(self.covariance))
        g1_full, g2_full = full_gaussian.decompose_in_two()

        g1 = GaussianDiagonal(self.dims, g1_full.mean, np.diag(g1_full.covariance))
        g2 = GaussianDiagonal(self.dims, g2_full.mean, np.diag(g2_full.covariance))
        return g1, g2

    def get_whitening_parameters_eigen(
        self,
    ) -> Tuple[List[int], List[int], MatrixType, MatrixType]:
        """
        Computes whitening matrices by delegating to the full covariance implementation.
        """
        temp_full_gauss = GaussianFull(self.dims, self.mean, np.diag(self.covariance))
        return temp_full_gauss.get_whitening_parameters_eigen()

    def get_whitening_parameters_svd(self) -> Tuple[List[int], List[int], MatrixType]:
        """
        Computes whitening matrix via SVD by delegating to the full covariance implementation.
        """
        temp_full_gauss = GaussianFull(self.dims, self.mean, np.diag(self.covariance))
        return temp_full_gauss.get_whitening_parameters_svd()

    def convolve(self, bandwidth: Union[VectorType, MatrixType]) -> None:
        """
        Adds a bandwidth to the covariance.
        If the bandwidth is a full matrix, its diagonal is extracted.
        """
        bw_to_add = np.diag(bandwidth) if bandwidth.ndim == 2 else bandwidth

        if self._covariance.shape != bw_to_add.shape:
            raise ValueError(
                f"Shape mismatch in convolve: self._covariance has shape {self._covariance.shape}, but bandwidth to add has shape {bw_to_add.shape}"
            )

        self._covariance += bw_to_add
        self._inverse_is_dirty = True

    def deconvolve(self, bandwidth: Union[VectorType, MatrixType]) -> None:
        """
        Subtracts a bandwidth from the covariance.
        If the bandwidth is a full matrix, its diagonal is extracted.
        """
        bw_to_subtract = np.diag(bandwidth) if bandwidth.ndim == 2 else bandwidth

        if self._covariance.shape != bw_to_subtract.shape:
            raise ValueError(
                f"Shape mismatch in deconvolve: self._covariance has shape {self._covariance.shape}, but bandwidth to subtract has shape {bw_to_subtract.shape}"
            )

        self._covariance -= bw_to_subtract
        self._inverse_is_dirty = True

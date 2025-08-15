import numpy as np
from typing import Tuple, List
import warnings

from ._gaussian_base import GaussianBase
from .mixture import Mixture
from .kde_utils import VectorType, MatrixType, ft_c_f_full_mat, f_c_ft
from . import constants as constants


class GaussianFull(GaussianBase[MatrixType]):
    """Gaussian distribution with a full covariance matrix."""

    def __init__(
        self, dims: int, mean: VectorType = None, covariance: MatrixType = None
    ):
        super().__init__(dims)
        self.mean = mean if mean is not None else np.zeros(dims)
        self.covariance = covariance if covariance is not None else np.identity(dims)

    def _mahalanobis_distance_sq(self, delta: VectorType) -> float:
        """Calculates delta' * inv(cov) * delta for a full covariance matrix."""
        return ft_c_f_full_mat(self.covariance_inverse(), delta)

    def _compute_determinant_and_inverse(self) -> None:
        """
        Computes the inverse and log-determinant of a full covariance matrix.
        Includes regularization logic to ensure the matrix is positive-definite.
        """
        if not self._inverse_is_dirty:
            return

        cov = self._covariance.copy()

        while True:
            try:
                eig_vals, _ = np.linalg.eigh(cov)
                if np.all(eig_vals > constants.MIN_BANDWIDTH):
                    break
            except np.linalg.LinAlgError:
                pass

            cov += np.identity(self._dims) * constants.EPSILON_REGULARIZE

        sign, log_det = np.linalg.slogdet(cov)

        if sign <= 0:
            self._is_invertible = False
            self._covariance_log_determinant = -np.inf
            self._covariance_inverse = np.zeros_like(cov)
        else:
            self._is_invertible = True
            self._covariance_log_determinant = log_det
            self._covariance_inverse = np.linalg.inv(cov)

        self._covariance = cov
        self._inverse_is_dirty = False

    @classmethod
    def from_mixture(cls, mix: Mixture, is_flat: bool = False) -> "GaussianFull":
        """Creates a Gaussian by moment-matching a mixture."""
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

        new_cov = np.zeros((dims, dims))
        for c, w in zip(mix.components, mix.weights):
            second_moment = c.covariance + np.outer(c.mean, c.mean)
            new_cov += w * second_moment
        new_cov /= total_weight

        new_cov -= np.outer(new_mean, new_mean)

        return cls(dims, new_mean, new_cov)

    @classmethod
    def from_merge(cls, mix: Mixture, bandwidth: MatrixType) -> "GaussianFull":
        """
        Creates a Gaussian by merging a mixture of other Gaussians.
        The bandwidth parameter is ignored, mirroring the C++ implementation.
        """
        return cls.from_mixture(mix, is_flat=False)

    def KL_divergence(self, other: GaussianBase) -> float:
        if not isinstance(other, GaussianFull):
            raise TypeError("KL Divergence requires another GaussianFull instance.")

        log_det_ratio = (
            other.covariance_log_determinant() - self.covariance_log_determinant()
        )
        trace_term = np.trace(other.covariance_inverse() @ self.covariance)
        mean_diff = self.mean - other.mean
        mahalanobis_term = mean_diff.T @ other.covariance_inverse() @ mean_diff

        kl_div = 0.5 * (log_det_ratio + trace_term + mahalanobis_term - self._dims)
        return kl_div if not np.isnan(kl_div) else np.inf

    def decompose_in_two(self) -> Tuple["GaussianFull", "GaussianFull"]:
        """Decomposes this Gaussian into two, preserving the first two moments."""
        U, s, _ = np.linalg.svd(self.covariance)
        sqrt_s = np.sqrt(s)

        M = np.zeros(self._dims)
        M[0] = 0.5

        FM = U @ np.diag(sqrt_s) @ M

        mu1 = self.mean + FM
        mu2 = self.mean - FM

        sigma = (
            self.covariance
            + np.outer(self.mean, self.mean)
            - 0.5 * (np.outer(mu1, mu1) + np.outer(mu2, mu2))
        )

        return GaussianFull(self._dims, mu1, sigma), GaussianFull(
            self._dims, mu2, sigma
        )

    def get_whitening_parameters_eigen(
        self,
    ) -> Tuple[List[int], List[int], MatrixType, MatrixType]:
        """
        Computes whitening transformation matrices F and iF using eigenvalue decomposition.

        Returns:
            A tuple (eig_ok, eig_ko, F, iF) where:
            - eig_ok: Indices of valid (non-degenerate) dimensions.
            - eig_ko: Indices of invalid (degenerate) dimensions.
            - F: The whitening matrix.
            - iF: The inverse whitening matrix (de-whitening).
        """
        eig_vals, V = np.linalg.eigh(self.covariance)

        eig_ok = [i for i, val in enumerate(eig_vals) if val > constants.MIN_BANDWIDTH]
        eig_ko = [i for i, val in enumerate(eig_vals) if val <= constants.MIN_BANDWIDTH]

        if not eig_ok:
            return (
                [],
                list(range(self.dims)),
                np.empty((self.dims, 0)),
                np.zeros_like(self.covariance),
            )

        ok_eig_vals = eig_vals[eig_ok]
        eig_vals_sqrt_inv = 1.0 / np.sqrt(ok_eig_vals)
        F = V[:, eig_ok] * eig_vals_sqrt_inv

        mean_ok_eig = np.mean(ok_eig_vals)
        sq_min_eig = np.sqrt(mean_ok_eig * 0.01)

        i_eig_vals_sqrt = np.sqrt(np.maximum(eig_vals, 0))
        i_eig_vals_sqrt[eig_ko] = sq_min_eig

        iF = V * i_eig_vals_sqrt
        iF = f_c_ft(np.identity(self.dims), V * i_eig_vals_sqrt)

        return eig_ok, eig_ko, F, iF

    def get_whitening_parameters_svd(self) -> Tuple[List[int], List[int], MatrixType]:
        """
        Computes a whitening transformation matrix using Singular Value Decomposition (SVD).

        Returns:
            A tuple (sv_ok, sv_ko, F_trns) where:
            - sv_ok: Indices of significant singular values.
            - sv_ko: Indices of non-significant singular values.
            - F_trns: The whitening transformation matrix.
        """
        C = self.covariance + np.identity(self.dims) * constants.EPSILON_WHITENING

        U, s, _ = np.linalg.svd(C)

        s_norm = s / s.max()

        sv_ok = [
            i for i, val in enumerate(s_norm) if val > constants.MIN_COEF_WHITENING
        ]
        sv_ko = [
            i for i, val in enumerate(s_norm) if val <= constants.MIN_COEF_WHITENING
        ]

        if not sv_ok:
            warnings.warn("Covariance matrix is completely singular in SVD whitening.")
            return [], list(range(self.dims)), np.zeros_like(self.covariance)

        S_inv_diag = np.zeros(self.dims)
        S_inv_diag[sv_ok] = 1.0 / s[sv_ok]

        min_s_val = s[sv_ok].min()
        for idx in sv_ko:
            S_inv_diag[idx] = 1.0 / (min_s_val * 1e-3)

        S_inv_sqrt = np.sqrt(S_inv_diag)

        F_trns = np.diag(S_inv_sqrt) @ U.T

        return sv_ok, sv_ko, F_trns

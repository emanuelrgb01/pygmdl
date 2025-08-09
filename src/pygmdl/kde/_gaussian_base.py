import numpy as np
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Tuple

from .kde_utils import VectorType, MatrixType, DataType, ft_c_f_full_mat, allclose
from .mixture import Mixture
from . import constants as constants

CovarianceType = TypeVar("CovarianceType", VectorType, MatrixType)


class GaussianBase(ABC, Generic[CovarianceType]):
    """
    Abstract Base Class for a Gaussian distribution.

    It defines the common interface and implements methods that do not depend
    on the specific covariance type (full or diagonal).
    """

    def __init__(self, dims: int):
        if dims <= 0:
            raise ValueError("Dimensions must be a positive integer.")
        self._dims: int = dims
        self._mean: VectorType = np.zeros(dims)

        self._log_pow_two_pi: float = -np.log(2.0 * np.pi) * (dims / 2.0)

        self._covariance: CovarianceType
        self._covariance_inverse: CovarianceType
        self._covariance_log_determinant: float = 0.0
        self._is_invertible: bool = False
        self._inverse_is_dirty: bool = True

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def mean(self) -> VectorType:
        return self._mean

    @mean.setter
    def mean(self, value: VectorType):
        if value.shape != self._mean.shape:
            raise ValueError("Dimension mismatch for mean vector.")
        self._mean = value

    @property
    def covariance(self) -> CovarianceType:
        return self._covariance

    @covariance.setter
    def covariance(self, value: CovarianceType):
        self._covariance = value
        self._inverse_is_dirty = True

    def covariance_is_invertible(self) -> bool:
        """Checks if the covariance matrix is invertible, computing it if necessary."""
        if self._inverse_is_dirty:
            self._compute_determinant_and_inverse()
        return self._is_invertible

    def covariance_log_determinant(self) -> float:
        """Returns the log determinant of the covariance, computing it if necessary."""
        if self._inverse_is_dirty:
            self._compute_determinant_and_inverse()
        return self._covariance_log_determinant

    def covariance_inverse(self) -> CovarianceType:
        """Returns the inverse of the covariance, computing it if necessary."""
        if self._inverse_is_dirty:
            self._compute_determinant_and_inverse()
        return self._covariance_inverse

    def convolve(self, bandwidth: CovarianceType) -> None:
        """Adds a bandwidth matrix to the covariance."""
        self._covariance += bandwidth
        self._inverse_is_dirty = True

    def deconvolve(self, bandwidth: CovarianceType) -> None:
        """Subtracts a bandwidth matrix from the covariance."""
        self._covariance -= bandwidth
        self._inverse_is_dirty = True

    def _log_probability(self, delta: VectorType) -> float:
        """Helper to calculate the log probability from a delta vector (x - mu)."""
        result = self._mahalanobis_distance_sq(delta)  # delta' * inv(cov) * delta
        exponent = -0.5 * result

        log_prob = (
            self._log_pow_two_pi - 0.5 * self.covariance_log_determinant() + exponent
        )
        return log_prob

    def log_likelihood(self, observation: VectorType) -> float:
        """Calculates the log-likelihood of a given observation."""
        delta = self.mean - observation
        return self._log_probability(delta)

    def likelihood(self, observation: VectorType) -> float:
        """Calculates the likelihood of a given observation."""
        return np.exp(self.log_likelihood(observation))

    def __str__(self) -> str:
        return f"Mean:\n{self.mean}\nCovariance:\n{self.covariance}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, GaussianBase):
            return NotImplemented
        return allclose(self.mean, other.mean) and allclose(
            self.covariance, other.covariance
        )

    @abstractmethod
    def _compute_determinant_and_inverse(self) -> None:
        """
        Computes the inverse and log-determinant of the covariance matrix.
        This method must handle regularization to ensure the matrix is invertible.
        It should set `_inverse_is_dirty` to False.
        """
        raise NotImplementedError

    @abstractmethod
    def _mahalanobis_distance_sq(self, delta: VectorType) -> float:
        """Calculates the squared Mahalanobis distance delta' * inv(cov) * delta."""
        raise NotImplementedError

    @abstractmethod
    def KL_divergence(self, other: "GaussianBase") -> float:
        """Computes the KL-Divergence to another Gaussian distribution."""
        raise NotImplementedError

    @abstractmethod
    def decompose_in_two(self) -> Tuple["GaussianBase", "GaussianBase"]:
        """Decomposes the Gaussian into two new Gaussians for K-Means initialization."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_mixture(cls, mix: Mixture, is_flat: bool = False) -> "GaussianBase":
        """Creates a Gaussian by moment-matching a mixture."""
        raise NotImplementedError

    @abstractmethod
    def get_whitening_parameters_eigen(
        self,
    ) -> Tuple[List[int], List[int], MatrixType, MatrixType]:
        """
        Computes whitening transformation matrices F and iF using eigenvalue decomposition.
        This must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def get_whitening_parameters_svd(self) -> Tuple[List[int], List[int], MatrixType]:
        """
        Computes a whitening transformation matrix using Singular Value Decomposition (SVD).
        This must be implemented by subclasses.
        """
        raise NotImplementedError

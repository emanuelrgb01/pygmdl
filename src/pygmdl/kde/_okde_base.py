import numpy as np
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List

from .mixture import Mixture
from .explanation_full import ExplanationFull
from .explanation_diagonal import ExplanationDiagonal
from .kde_utils import VectorType, MatrixType
from .hellinger_distance import hellinger_distance

SamplePDF_T = TypeVar("SamplePDF_T")
CovarianceType = TypeVar("CovarianceType", VectorType, MatrixType)
ExplanationType = TypeVar("ExplanationType", ExplanationFull, ExplanationDiagonal)


class OKDEBase(Mixture[ExplanationType], ABC, Generic[SamplePDF_T, CovarianceType]):
    """
    Abstract Base Class for an Online Kernel Density Estimator (oKDE).

    This class manages the online learning process, including adding new samples,
    deciding when to compress the model, and updating internal state variables.
    It inherits from Mixture, as an oKDE is a mixture of "explanation" components.
    """

    def __init__(self, dims: int, d_th: float = 0.1, forgetting_factor: float = 1.0):
        super().__init__(dims)

        self.d_th: float = d_th
        self.forgetting_factor: float = forgetting_factor

        self.nt: float = 0.0
        self.n_alpha: float = 1.0

        dm = ((dims * dims - dims) / 2) + 2 * dims + 1
        self.m_thc: float = min(15.0, dm)

        self.optimal_bandwidth: CovarianceType = None

    def _update_m_thc(self) -> None:
        """Dynamically updates the compression threshold."""
        if len(self) > self.m_thc:
            self.m_thc *= 1.5
        elif len(self) < self.m_thc / 2:
            self.m_thc *= 0.6

    def _update_n_alpha(self) -> None:
        """Updates the N_alpha parameter based on the effective number of samples."""
        if np.isclose(self.nt, 0):
            return
        w0 = 1.0 / self.nt
        self.n_alpha = 1.0 / ((1 - w0) ** 2 / self.n_alpha + w0**2)

    def _compression_routine(self) -> None:
        """Performs the full compression cycle: bandwidth estimation and clustering."""
        self.estimate_kernel_density()
        self.hierarchical_clustering()
        self._update_m_thc()

    def estimate_kernel_density(self) -> None:
        """Computes the optimal bandwidth and convolves the mixture components."""
        self.estimate_bandwidth()
        if self.optimal_bandwidth is not None:
            self.convolve(self.optimal_bandwidth)

    def add_sample(self, features: VectorType) -> None:
        """
        Adds a new data sample to the KDE, updating the model online.
        """
        self.nt = self.nt * self.forgetting_factor + 1
        self._update_n_alpha()

        w0 = 1.0 / self.nt
        self.scale_weights(1 - w0)

        ExplType = self._get_explanation_type()
        self.add(ExplType(self.dims, mean=features), w0)

        if len(self) > self.m_thc:
            self._compression_routine()

    def add_samples(self, samples: List[VectorType]) -> None:
        """Adds a batch of samples to the KDE."""
        for sample in samples:
            self.add_sample(sample)

    def revitalize(self) -> List[int]:
        """
        Expands components whose simplified representation has too high an error
        compared to their detailed underlying model.
        """
        revitalization_list = []
        for i in range(len(self)):
            component = self.component(i)
            if len(component.underlying_model) < 2:
                continue

            p0 = Mixture[ExplanationType](self.dims)
            p0.add(component, self.weight(i))

            detailed_model_as_mixture = Mixture[ExplanationType](self.dims)
            for j in range(len(component.underlying_model)):
                ExplType = self._get_explanation_type()
                detailed_expl = ExplType(self.dims)
                detailed_expl.mean = component.underlying_model.component(j).mean
                detailed_expl.covariance = component.underlying_model.component(
                    j
                ).covariance
                detailed_model_as_mixture.add(
                    detailed_expl, self.weight(i) * component.underlying_model.weight(j)
                )

            detailed_model_as_mixture.convolve(self.optimal_bandwidth)

            local_error = hellinger_distance(detailed_model_as_mixture, p0)

            if local_error > self.d_th:
                splitted_components = component.revitalize()
                splitted_components.convolve(self.optimal_bandwidth)

                original_weight = self.weight(i)
                self.replace(
                    i,
                    splitted_components.component(0),
                    original_weight * splitted_components.weight(0),
                )
                self.add(
                    splitted_components.component(1),
                    original_weight * splitted_components.weight(1),
                )
                revitalization_list.append(i)

        return revitalization_list

    @abstractmethod
    def estimate_bandwidth(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def hierarchical_clustering(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _get_explanation_type(self) -> type:
        """Helper to get the concrete Explanation class type."""
        raise NotImplementedError

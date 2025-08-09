import numpy as np
from typing import Generic, TypeVar, List

from .gaussian_diagonal import GaussianDiagonal
from .gaussian_full import GaussianFull
from .mixture import Mixture
from .kde_utils import (
    MatrixType,
    VectorType,
    f_c_ft,
    select_subset_vector,
    select_subset_matrix,
)
from .goldberger_k_means import goldberger_k_means

SamplePDF_T = TypeVar("SamplePDF_T", bound=GaussianDiagonal)


class ExplanationDiagonal(GaussianDiagonal, Generic[SamplePDF_T]):
    """
    Represents an "explanation" component with a diagonal covariance matrix.
    """

    def __init__(self, dims: int, mean: VectorType = None):
        super().__init__(dims)
        self.underlying_model = Mixture[SamplePDF_T](dims)
        self.base_covariance = np.zeros(dims)

        if mean is not None:
            self.mean = mean
            underlying_comp = GaussianDiagonal(
                dims, mean=mean, covariance=np.full(dims, 1e-9)
            )
            self.underlying_model.add(underlying_comp, 1.0)
            self.base_covariance = underlying_comp.covariance

    @classmethod
    def empty(cls, dims: int) -> "ExplanationDiagonal":
        """A factory to create an empty, default-initialized instance."""
        return cls(dims)

    @classmethod
    def from_moment_match(
        cls,
        mix: Mixture["ExplanationDiagonal[SamplePDF_T]"],
        use_base_covariance: bool = True,
    ) -> "ExplanationDiagonal[SamplePDF_T]":
        """Creates a diagonal Explanation by moment-matching a mixture."""
        if not mix:
            raise ValueError("Cannot create Explanation from an empty mixture.")

        dims = mix.dims
        instance = cls(dims)
        total_weight = mix.sum_of_weights()

        instance.mean = (
            np.sum([c.mean * w for c, w in zip(mix.components, mix.weights)], axis=0)
            / total_weight
        )

        new_base_cov_diag = np.zeros(dims)
        for c, w in zip(mix.components, mix.weights):
            cov_to_use = c.base_covariance if use_base_covariance else c.covariance
            second_moment_diag = cov_to_use + c.mean**2
            new_base_cov_diag += w * second_moment_diag

        new_base_cov_diag /= total_weight
        new_base_cov_diag -= instance.mean**2

        instance.base_covariance = new_base_cov_diag
        instance.covariance = new_base_cov_diag

        return instance

    @classmethod
    def from_merge(
        cls, mix: Mixture["ExplanationDiagonal[SamplePDF_T]"], bandwidth: VectorType
    ) -> "ExplanationDiagonal[SamplePDF_T]":
        """Creates a new Explanation by merging a mixture of other explanations."""
        dims = mix.dims

        all_details = Mixture[SamplePDF_T](dims)
        for expl_comp, expl_weight in zip(mix.components, mix.weights):
            for detailed_comp, detailed_weight in zip(
                expl_comp.underlying_model.components,
                expl_comp.underlying_model.weights,
            ):
                all_details.add(detailed_comp, detailed_weight * expl_weight)

        all_details.normalize_weights_preserve_relative_importance()

        all_details.convolve(bandwidth)
        k = min(2, len(all_details))

        if not all_details:
            raise ValueError("Cannot merge explanations with no underlying components.")
        DetailedComponentType = type(all_details.component(0))

        new_detailed_model = Mixture[DetailedComponentType](dims)
        for _ in range(k):
            new_detailed_model.add(DetailedComponentType(dims))

        goldberger_k_means(all_details, np.diag(bandwidth), new_detailed_model)
        new_detailed_model.deconvolve(bandwidth)

        temp_gaussian = GaussianDiagonal.from_mixture(new_detailed_model)
        final_explanation = cls(dims)
        final_explanation.mean = temp_gaussian.mean
        final_explanation.base_covariance = temp_gaussian.covariance
        final_explanation.underlying_model = new_detailed_model

        final_explanation.convolve(bandwidth)
        return final_explanation

    @classmethod
    def from_whitening(
        cls,
        to_whiten: "ExplanationDiagonal[SamplePDF_T]",
        smp_mean: VectorType,
        F_trns: MatrixType,
        eig_ok: List[int],
    ) -> "ExplanationDiagonal[SamplePDF_T]":
        """Creates a new, whitened Explanation from an existing one."""
        whitened_dims = len(eig_ok)
        instance = cls(whitened_dims)

        DetailedComponentType = type(to_whiten.underlying_model.component(0))

        for detailed_comp, w in zip(
            to_whiten.underlying_model.components, to_whiten.underlying_model.weights
        ):
            white_mean_full = F_trns @ (detailed_comp.mean - smp_mean)
            white_mean_valid = select_subset_vector(white_mean_full, eig_ok)

            white_cov_full = f_c_ft(np.diag(detailed_comp.covariance), F_trns)
            white_cov_valid_full = select_subset_matrix(white_cov_full, eig_ok)

            white_cov_valid_diag = np.diag(white_cov_valid_full)

            instance.underlying_model.add(
                DetailedComponentType(
                    whitened_dims, white_mean_valid, white_cov_valid_diag
                ),
                w,
            )

        moment_matched = GaussianDiagonal.from_mixture(instance.underlying_model)
        instance.mean = moment_matched.mean
        instance.base_covariance = moment_matched.covariance
        instance.covariance = moment_matched.covariance

        return instance

    def revitalize(self) -> Mixture["ExplanationDiagonal[SamplePDF_T]"]:
        """
        Expands this single explanation back into a mixture of two simpler explanations.
        """
        splitted_mix = Mixture(self.dims)
        DetailedComponentType = type(self.underlying_model.component(0))

        if len(self.underlying_model) < 2:
            new_expl = type(self)(self.dims)
            new_expl.mean = self.underlying_model.component(0).mean
            new_expl.base_covariance = self.underlying_model.component(0).covariance
            new_expl.underlying_model = self.underlying_model
            splitted_mix.add(new_expl, 1.0)
            return splitted_mix

        for detailed_comp, weight in zip(
            self.underlying_model.components, self.underlying_model.weights
        ):
            split1, split2 = detailed_comp.decompose_in_two()

            new_detailed_model = Mixture[DetailedComponentType](self.dims)
            new_detailed_model.add(split1, 0.5)
            new_detailed_model.add(split2, 0.5)

            new_expl = type(self)(self.dims)
            new_expl.mean = detailed_comp.mean
            new_expl.base_covariance = detailed_comp.covariance
            new_expl.underlying_model = new_detailed_model
            splitted_mix.add(new_expl, weight)

        return splitted_mix

    def convolve(self, bandwidth: VectorType) -> None:
        self.covariance = self.base_covariance + bandwidth

    def deconvolve(self, bandwidth: VectorType) -> None:
        self.covariance = self.base_covariance

import warnings
import numpy as np
from typing import TypeVar, List

from ._okde_base import OKDEBase
from .explanation_diagonal import ExplanationDiagonal
from .gaussian_diagonal import GaussianDiagonal
from .mixture import Mixture
from .kde_utils import (
    VectorType,
    MatrixType,
    ft_c_f_full_mat,
    f_c_ft,
    select_subset_matrix,
)
from .similarity_groups import generate_similarity_groups_indexes

SamplePDF_T = TypeVar("SamplePDF_T")


class OKDEDiagonal(OKDEBase[SamplePDF_T, VectorType, ExplanationDiagonal[SamplePDF_T]]):
    """Online Kernel Density Estimator with Diagonal Covariance matrices."""

    def __init__(self, dims: int, d_th: float = 0.1, forgetting_factor: float = 1.0):
        super().__init__(dims, d_th, forgetting_factor)

    def _get_explanation_type(self) -> type:
        return ExplanationDiagonal[SamplePDF_T]

    @classmethod
    def from_whitening(
        cls,
        to_whiten: "OKDEDiagonal",
        smp_mean: VectorType,
        F_trns: MatrixType,
        eig_ok: List[int],
    ) -> "OKDEDiagonal":
        """
        A factory to create a new, whitened OKDE from an existing one.
        """
        whitened_dims = len(eig_ok)
        instance = cls(whitened_dims)

        instance.d_th = to_whiten.d_th
        instance.nt = to_whiten.nt
        instance.n_alpha = to_whiten.n_alpha
        instance.m_thc = to_whiten.m_thc
        instance.forgetting_factor = to_whiten.forgetting_factor

        if to_whiten.optimal_bandwidth is not None:
            new_full_h = f_c_ft(np.diag(to_whiten.optimal_bandwidth), F_trns)
            instance.optimal_bandwidth = select_subset_matrix(
                new_full_h, eig_ok
            ).diagonal()

        for component, weight in zip(to_whiten.components, to_whiten.weights):
            whitened_expl = ExplanationDiagonal.from_whitening(
                component, smp_mean, F_trns, eig_ok
            )
            instance.add(whitened_expl, weight)

        if instance.optimal_bandwidth is not None:
            instance.convolve(instance.optimal_bandwidth)

        return instance

    def _compute_rpfg(
        self, mix: Mixture, F_smp_cov_diag: VectorType, N: float
    ) -> float:
        dims = F_smp_cov_diag.shape[0]
        F = np.diag(F_smp_cov_diag)
        f = (4.0 / ((dims + 2.0) * N)) ** (2.0 / (dims + 4.0))
        G_diag = F_smp_cov_diag * f
        F_sq = F @ F
        rpfg = 0.0
        for i in range(len(mix)):
            wi = mix.weight(i)
            e_base = self._get_explanation_type()(dims)
            e_base.mean = mix.component(i).mean
            e_base.base_covariance = mix.component(i).base_covariance
            e_base.convolve(G_diag)
            for j in range(i, len(mix)):
                wj = mix.weight(j)
                ei = self._get_explanation_type()(dims)
                ei.mean = e_base.mean
                ei.base_covariance = e_base.covariance
                ei.convolve(mix.component(j).base_covariance)
                Aij_diag = ei.covariance_inverse()
                Aij = np.diag(Aij_diag)
                dij = mix.mean(i) - mix.mean(j)
                mij = np.sum(dij * Aij_diag * dij)
                p1 = wi * wj * ei.likelihood(mix.mean(j))
                p2 = 2 * np.trace(F_sq @ Aij @ Aij) * (1 - (2 * mij))
                mulF_Aij_tr = np.trace(F @ Aij)
                p3 = (mulF_Aij_tr**2) * ((1 - mij) ** 2)
                mult = 1.0 if i == j else 2.0
                rpfg += p1 * (p2 + p3) * mult
        return rpfg

    def estimate_bandwidth(self) -> None:
        N = self.nt
        if N <= 1:
            self.optimal_bandwidth = np.ones(self.dims)
            return

        moment_matched = ExplanationDiagonal.from_moment_match(
            self, use_base_covariance=True
        )
        smp = GaussianDiagonal(
            self.dims, moment_matched.mean, moment_matched.base_covariance
        )
        eig_ok, _, F, iF = smp.get_whitening_parameters_eigen()

        if not eig_ok:
            warnings.warn("Bandwidth estimation failed: covariance matrix is singular.")
            self.optimal_bandwidth = np.ones(self.dims) * 1e-5
            return

        F_smp_cov_diag = ft_c_f_full_mat(np.diag(smp.covariance), F).diagonal()
        F_smp = GaussianDiagonal(len(eig_ok), covariance=F_smp_cov_diag)

        n_mix = Mixture.from_whitening(
            self,
            smp_mean=smp.mean,
            F_trns=F,
            eig_ok=eig_ok,
            ExplanationType=self._get_explanation_type(),
        )

        rpfg = self._compute_rpfg(n_mix, F_smp.covariance, N)
        if rpfg == 0:
            warnings.warn("Bandwidth estimation failed: R(p,F,G) is zero.")
            self.optimal_bandwidth = np.ones(self.dims) * 1e-5
            return

        dims_eff = F_smp.covariance.shape[0]
        log_det_F_smp_cov = F_smp.covariance_log_determinant()
        log_inner_scale = (
            -0.5 * log_det_F_smp_cov
            - np.log(N)
            - np.log(pow(np.sqrt(4.0 * np.pi), dims_eff) * dims_eff * rpfg)
        )
        inner_scale = np.exp(log_inner_scale)

        bandwidth_opt_scaled = inner_scale ** (1.0 / (dims_eff + 4.0))
        bandwidth_opt_scaled_sq = bandwidth_opt_scaled**2

        H_diag = np.ones(self.dims)
        if bandwidth_opt_scaled_sq != 0:
            for v_idx in eig_ok:
                H_diag[v_idx] = bandwidth_opt_scaled_sq

        self.optimal_bandwidth = f_c_ft(np.diag(H_diag), iF).diagonal()

    def hierarchical_clustering(self) -> None:
        smp = ExplanationDiagonal.from_moment_match(self, use_base_covariance=False)
        eig_ok, _, F_trns = smp.get_whitening_parameters_svd()

        whitened_kde = OKDEDiagonal.from_whitening(self, smp.mean, F_trns, eig_ok)

        revitalized_indices_in_whitened_kde = whitened_kde.revitalize()

        if revitalized_indices_in_whitened_kde:
            new_components = []
            new_weights = []
            revitalized_set = set(revitalized_indices_in_whitened_kde)

            for i in range(len(self)):
                component = self.component(i)
                weight = self.weight(i)

                if i in revitalized_set:
                    splitted_mixture = component.revitalize()
                    splitted_mixture.convolve(self.optimal_bandwidth)

                    for j in range(len(splitted_mixture)):
                        new_components.append(splitted_mixture.component(j))
                        new_weights.append(weight * splitted_mixture.weight(j))
                else:
                    new_components.append(component)
                    new_weights.append(weight)

            self._components = new_components
            self._weights = new_weights

        whitened_bw_full = np.diag(whitened_kde.optimal_bandwidth)
        group_indexes = generate_similarity_groups_indexes(
            whitened_kde, whitened_bw_full, self.d_th
        )

        if len(group_indexes) == len(self):
            return

        compressed_mix = Mixture(self.dims)
        opt_band = (self.optimal_bandwidth / 2.0) * (0.5 * 0.5)

        for group in group_indexes:
            sub_mix = Mixture.from_components(
                [self.components[i] for i in group],
                [self.weights[i] for i in group],
                self.dims,
            )
            merged_expl = ExplanationDiagonal.from_merge(sub_mix, opt_band)
            compressed_mix.add(merged_expl, sub_mix.sum_of_weights())

        self._components = compressed_mix.components
        self._weights = compressed_mix.weights

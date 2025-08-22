import numpy as np
from typing import TypeVar, List
import warnings

from ._okde_base import OKDEBase
from .explanation_full import ExplanationFull
from .gaussian_full import GaussianFull
from .mixture import Mixture
from .kde_utils import (
    MatrixType,
    VectorType,
    ft_c_f_full_mat,
    f_c_ft,
    select_subset_matrix,
)
from .similarity_groups import generate_similarity_groups_indexes

SamplePDF_T = TypeVar("SamplePDF_T")


class OKDEFull(OKDEBase[SamplePDF_T, MatrixType, ExplanationFull[SamplePDF_T]]):
    """Online Kernel Density Estimator with Full Covariance matrices."""

    def __init__(self, dims: int, d_th: float = 0.1, forgetting_factor: float = 1.0):
        super().__init__(dims, d_th, forgetting_factor)

    def _get_explanation_type(self) -> type:
        return ExplanationFull[SamplePDF_T]

    @classmethod
    def from_whitening(
        cls,
        to_whiten: "OKDEFull",
        smp_mean: VectorType,
        F_trns: MatrixType,
        eig_ok: List[int],
    ) -> "OKDEFull":
        """
        A factory to create a new, whitened OKDE from an existing one.
        This corresponds to the whitening constructor in the C++ code.
        """
        whitened_dims = len(eig_ok)
        instance = cls(whitened_dims)

        instance.d_th = to_whiten.d_th
        instance.nt = to_whiten.nt
        instance.n_alpha = to_whiten.n_alpha
        instance.m_thc = to_whiten.m_thc
        instance.forgetting_factor = to_whiten.forgetting_factor

        if to_whiten.optimal_bandwidth is not None:
            new_full_h = f_c_ft(to_whiten.optimal_bandwidth, F_trns)
            instance.optimal_bandwidth = select_subset_matrix(new_full_h, eig_ok)

        for component, weight in zip(to_whiten.components, to_whiten.weights):
            whitened_expl = ExplanationFull.from_whitening(
                component, smp_mean, F_trns, eig_ok
            )
            instance.add(whitened_expl, weight)

        if instance.optimal_bandwidth is not None:
            instance.convolve(instance.optimal_bandwidth)

        return instance

    def _compute_rpfg(self, mix: Mixture, F: MatrixType, N: float) -> float:
        dims = F.shape[0]
        f = (4.0 / ((dims + 2.0) * N)) ** (2.0 / (dims + 4.0))
        G = F * f
        F_sq = F @ F

        rpfg = 0.0
        for i in range(len(mix)):
            wi = mix.weight(i)
            e_base = self._get_explanation_type()(dims)
            e_base.mean = mix.component(i).mean
            e_base.base_covariance = mix.component(i).base_covariance
            e_base.convolve(G)

            for j in range(i, len(mix)):
                wj = mix.weight(j)
                ei = self._get_explanation_type()(dims)
                ei.mean = e_base.mean
                ei.base_covariance = e_base.covariance
                ei.convolve(mix.component(j).base_covariance)

                Aij = ei.covariance_inverse()
                dij = mix.mean(i) - mix.mean(j)
                mij = ft_c_f_full_mat(Aij, dij)

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
            self.optimal_bandwidth = np.identity(self.dims)
            return

        moment_matched = ExplanationFull.from_moment_match(
            self, use_base_covariance=True
        )
        smp = GaussianFull(
            self.dims, moment_matched.mean, moment_matched.base_covariance
        )

        eig_ok, _, F, iF = smp.get_whitening_parameters_eigen()

        if not eig_ok:
            warnings.warn("Bandwidth estimation failed: covariance matrix is singular.")
            self.optimal_bandwidth = np.identity(self.dims) * 1e-5
            return

        F_smp_cov = ft_c_f_full_mat(smp.covariance, F)
        F_smp = GaussianFull(len(eig_ok), covariance=F_smp_cov)

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
            self.optimal_bandwidth = np.identity(self.dims) * 1e-5
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

        H = np.zeros((self.dims, self.dims))
        diag_H = np.ones(self.dims)
        if bandwidth_opt_scaled_sq != 0:
            for v_idx in eig_ok:
                diag_H[v_idx] = bandwidth_opt_scaled_sq

        H = np.diag(diag_H)
        self.optimal_bandwidth = f_c_ft(H, iF)

    def hierarchical_clustering(self) -> None:
        smp = ExplanationFull.from_moment_match(self, use_base_covariance=False)
        eig_ok, _, F_trns = smp.get_whitening_parameters_svd()

        whitened_kde = OKDEFull.from_whitening(self, smp.mean, F_trns, eig_ok)

        revitalized_indices_in_whitened_kde = whitened_kde.revitalize()

        # If any components were revitalized in the whitened space, we must
        # apply the same revitalization to the original KDE to keep them in sync.
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

        group_indexes = generate_similarity_groups_indexes(
            whitened_kde, whitened_kde.optimal_bandwidth, self.d_th
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
            merged_expl = ExplanationFull.from_merge(sub_mix, opt_band)
            compressed_mix.add(merged_expl, sub_mix.sum_of_weights())

        self._components = compressed_mix.components
        self._weights = compressed_mix.weights

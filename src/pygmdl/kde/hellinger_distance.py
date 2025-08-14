import numpy as np
from typing import TypeVar
import copy

from .mixture import Mixture

PDF_T = TypeVar("PDF_T")


def _compute_sigma_point_integrand(p1: float, p2: float, p0: float) -> float:
    """
    Computes the core term ((sqrt(p1)-sqrt(p2))^2)/p0 for the Hellinger distance integration.
    Includes checks for numerical stability.
    """
    if np.isinf(p1) and np.isinf(p2):
        diff = 0.0
    else:
        diff = p1 - p2

    if np.isclose(p0, 0.0):
        return 0.0

    gx = np.nan_to_num((diff * diff) / p0, nan=0.0)

    return gx


def hellinger_distance(one: Mixture[PDF_T], other: Mixture[PDF_T]) -> float:
    """
    Calculates the Unscented Hellinger Distance between two mixture models.

    Args:
        one: The first mixture model.
        other: The second mixture model.

    Returns:
        The Hellinger distance, a value between 0 and 1.
    """
    if one.dims != other.dims:
        raise ValueError("Mixtures must have the same number of dimensions.")
    dims = one.dims

    importance_mix = copy.deepcopy(one)
    importance_mix.add_all(other)
    importance_mix.scale_weights(0.5)

    k = max(0, 3 - dims)
    nk = float(dims + k)
    wi0 = k / nk
    wi = 1.0 / (2 * nk)

    global_sum = 0.0

    for i in range(len(importance_mix)):
        component = importance_mix.component(i)
        mean = component.mean()

        cov = component.covariance()
        if cov.ndim == 1:
            cov = np.diag(cov)

        U, s_vals, _ = np.linalg.svd(cov)

        s_prime = nk * s_vals
        sqrt_s_prime = np.sqrt(s_prime)

        ucols = U * sqrt_s_prime

        inner_sum = 0.0

        if not np.isclose(wi0, 0.0):
            p0 = importance_mix.likelihood(mean)
            p1 = np.sqrt(one.likelihood(mean))
            p2 = np.sqrt(other.likelihood(mean))
            gx = _compute_sigma_point_integrand(p1, p2, p0)
            inner_sum += gx * wi0

        for j in range(dims):
            mean_plus_ucols = mean + ucols[:, j]
            mean_minus_ucols = mean - ucols[:, j]

            p0_plus = importance_mix.likelihood(mean_plus_ucols)
            p1_plus = np.sqrt(one.likelihood(mean_plus_ucols))
            p2_plus = np.sqrt(other.likelihood(mean_plus_ucols))
            gx_plus = _compute_sigma_point_integrand(p1_plus, p2_plus, p0_plus)
            inner_sum += gx_plus * wi

            p0_minus = importance_mix.likelihood(mean_minus_ucols)
            p1_minus = np.sqrt(one.likelihood(mean_minus_ucols))
            p2_minus = np.sqrt(other.likelihood(mean_minus_ucols))
            gx_minus = _compute_sigma_point_integrand(p1_minus, p2_minus, p0_minus)
            inner_sum += gx_minus * wi

            if np.isnan(inner_sum):
                raise RuntimeError(f"NaN detected in inner_sum at dimension j={j}")

        if np.isnan(inner_sum):
            raise RuntimeError(
                "NaN detected in inner_sum after processing a component."
            )

        global_sum += importance_mix.weight(i) * inner_sum

    term = 0.5 * global_sum
    distance = np.sqrt(max(0, term))

    if np.isnan(distance):
        raise RuntimeError("NaN detected in final distance calculation.")

    return min(distance, 1.0)

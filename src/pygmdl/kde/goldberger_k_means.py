import numpy as np
from typing import List, Tuple, TypeVar

from .mixture import Mixture
from .kde_utils import MatrixType

PDF_T = TypeVar("PDF_T")


def regroup(
    mix: Mixture[PDF_T], compressed_mix: Mixture[PDF_T], mapping: np.ndarray
) -> Tuple[bool, bool]:
    """
    The assignment step of the K-Means algorithm (with k=2).

    Assigns each component of the input mixture `mix` to one of the two
    clusters in `compressed_mix` based on the minimum KL-Divergence.

    Args:
        mix: The original, larger mixture model.
        compressed_mix: The mixture being fitted, which acts as the cluster centers. Must have size 2.
        mapping: A numpy array to be filled with the assignments (0 or 1).

    Returns:
        A tuple (error_occurred, force_stop).
        - error_occurred (bool): True if a critical error happened.
        - force_stop (bool): True if an empty cluster was detected and corrected, signaling to stop iteration.
    """
    assert len(mix) > 0, "Input mixture cannot be empty."
    assert (
        len(compressed_mix) == 2
    ), "Compressed mixture must have exactly 2 components for this algorithm."

    max_dist = -np.inf
    max_dist_idx = -1
    force_stop = False

    for i in range(len(mix)):
        dist_0 = abs(compressed_mix.component(0).KL_divergence(mix.component(i)))
        dist_1 = abs(compressed_mix.component(1).KL_divergence(mix.component(i)))

        internal_max_dist = max(dist_0, dist_1)
        if internal_max_dist > max_dist:
            max_dist = internal_max_dist
            max_dist_idx = i

        if dist_1 < dist_0:
            mapping[i] = 1
        else:
            mapping[i] = 0

    counts = np.bincount(mapping, minlength=2)
    c0, c1 = counts[0], counts[1]

    if c0 == 0:
        if max_dist_idx != -1:
            mapping[max_dist_idx] = 0
            force_stop = True
        else:
            print("Error: max_dist_idx was not found. Aborting regroup.")
            return True, True

    elif c1 == 0:
        if max_dist_idx != -1:
            mapping[max_dist_idx] = 1
            force_stop = True
        else:
            print("Error: max_dist_idx was not found. Aborting regroup.")
            return True, True

    return False, force_stop


def refit(
    mix: Mixture[PDF_T],
    optimal_bandwidth: MatrixType,
    compressed_mix: Mixture[PDF_T],
    mapping: np.ndarray,
) -> None:
    """
    The update step of the K-Means algorithm.

    Recalculates the cluster centers in `compressed_mix` by performing a
    moment-match on the components assigned to each cluster.

    It assumes the component type PDF_T can be instantiated from a mixture
    and a bandwidth (e.g., via a specific constructor or classmethod).

    Args:
        mix: The original, larger mixture model.
        optimal_bandwidth: The bandwidth matrix to be used during component refitting.
        compressed_mix: The mixture to be updated with the new cluster centers.
        mapping: The current assignment of components to clusters.
    """
    dims = mix.dims
    ComponentType = type(mix.component(0))

    for i in range(len(compressed_mix)):

        assigned_indices = np.where(mapping == i)[0]

        if len(assigned_indices) == 0:
            print(f"Warning: Cluster {i} has no components assigned during refit.")
            continue

        comp_weight = np.sum([mix.weight(j) for j in assigned_indices])

        if comp_weight > 0.0:
            close_mix = Mixture[PDF_T](dims)
            for j in assigned_indices:
                close_mix.add(mix.component(j), mix.weight(j) / comp_weight)

            new_component = ComponentType(close_mix, optimal_bandwidth)

            compressed_mix.replace(i, new_component, comp_weight)

            if np.isnan(compressed_mix.component(i).covariance_log_determinant()):
                print(
                    "Warning: NaN detected in covariance log-determinant after refit."
                )


def goldberger_k_means(
    mix: Mixture[PDF_T],
    optimal_bandwidth: MatrixType,
    compressed_mix: Mixture[PDF_T],
    max_iterations: int = 20,
) -> np.ndarray:
    """
    Compresses a mixture model into two components using a K-Means-like algorithm.

    This algorithm uses KL-Divergence as its distance metric and is specifically
    tailored for mixture models of probability distributions.

    Args:
        mix: The original mixture model to be compressed.
        optimal_bandwidth: The bandwidth matrix used for refitting.
        compressed_mix: An initialized mixture of size 2, which will be modified in-place
                        to hold the final compressed components.
        max_iterations: The maximum number of iterations to perform.

    Returns:
        A numpy array containing the final mapping of original components to the two new clusters.
    """
    if not len(mix) > 0:
        return np.array([], dtype=int)

    mapping = np.zeros(len(mix), dtype=int)

    if len(mix) <= 2:
        for i in range(len(mix)):
            compressed_mix.replace(i, mix.component(i), mix.weight(i))
            mapping[i] = i
        return mapping

    assert len(compressed_mix) == 2, "This implementation only supports k=2."

    ComponentType = type(mix.component(0))

    moment_matched = ComponentType.from_mixture(mix)
    split_1, split_2 = moment_matched.decompose_in_two()

    compressed_mix.replace(0, split_1, 0.5)
    compressed_mix.replace(1, split_2, 0.5)

    visited_mappings = []
    iteration = 0
    stop = False

    while not stop:
        visited_mappings.append(mapping.copy())

        error, force_stop = regroup(mix, compressed_mix, mapping)
        if error:
            break

        refit(mix, optimal_bandwidth, compressed_mix, mapping)

        if (
            force_stop
            or iteration > 0
            and np.array_equal(visited_mappings[-1], mapping)
        ):
            stop = True

        if not stop:
            for old_mapping in visited_mappings[:-1]:
                if np.array_equal(old_mapping, mapping):
                    stop = True
                    break

        iteration += 1
        if iteration >= max_iterations:
            stop = True

    return mapping

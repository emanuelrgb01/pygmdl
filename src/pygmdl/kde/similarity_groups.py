import numpy as np
from typing import List, TypeVar

from .mixture import Mixture
from .hellinger_distance import hellinger_distance
from .goldberger_k_means import goldberger_k_means
from .kde_utils import vector_max_val_and_idx, MatrixType

PDF_T = TypeVar("PDF_T")


def generate_similarity_groups_indexes(
    mix: Mixture[PDF_T], optimal_bandwidth: MatrixType, d_th: float
) -> List[List[int]]:
    """
    Generates similarity groups using a divisive clustering algorithm.

    This function starts with all components in one group and iteratively splits
    the group with the highest error (measured by Hellinger distance) until
    the error of all groups is below a given threshold `d_th`.

    Args:
        mix: The original mixture model to be grouped.
        optimal_bandwidth: The bandwidth matrix used for the k-means splitting.
        d_th: The Hellinger distance threshold to stop splitting.

    Returns:
        A list of lists, where each inner list contains the indices of the
        components from the original mixture that form a group.
    """
    if len(mix) == 0:
        return []

    dims = mix.dims
    ComponentType = type(mix.component(0))

    one_gauss_approx = ComponentType.from_mixture(mix, is_flat=True)

    compressed_mixture = Mixture[PDF_T](dims)
    compressed_mixture.add(one_gauss_approx, mix.sum_of_weights())

    dist = hellinger_distance(compressed_mixture, mix)

    hell_dists: List[float] = [dist]

    group_indexes: List[List[int]] = [list(range(len(mix)))]

    while True:
        max_elem, max_idx = vector_max_val_and_idx(hell_dists)

        if max_elem <= d_th:
            break

        sub_mix = Mixture[PDF_T](dims)
        for index in group_indexes[max_idx]:
            sub_mix.add(mix.component(index), mix.weight(index))

        sub_mix_original_weight_sum = sub_mix.sum_of_weights()
        sub_mix.normalize_weights_preserve_relative_importance()

        k_for_split = min(2, len(sub_mix))
        compressed_split_mix = Mixture[PDF_T](dims)
        for _ in range(k_for_split):
            compressed_split_mix.add(ComponentType.empty(dims))

        mapping = goldberger_k_means(sub_mix, optimal_bandwidth, compressed_split_mix)

        if len(compressed_split_mix) <= 1:
            new_weight = sub_mix_original_weight_sum * compressed_split_mix.weight(0)
            compressed_mixture.replace(
                max_idx, compressed_split_mix.component(0), new_weight
            )
            hell_dists[max_idx] = 0.0

        elif len(compressed_split_mix) == 2:
            new_mappings_0, new_mappings_1 = [], []
            for i, cluster_id in enumerate(mapping):
                original_index = group_indexes[max_idx][i]
                if cluster_id == 0:
                    new_mappings_0.append(original_index)
                else:
                    new_mappings_1.append(original_index)

            weight_0 = sub_mix_original_weight_sum * compressed_split_mix.weight(0)
            compressed_mixture.replace(
                max_idx, compressed_split_mix.component(0), weight_0
            )
            group_indexes[max_idx] = new_mappings_0

            weight_1 = sub_mix_original_weight_sum * compressed_split_mix.weight(1)
            compressed_mixture.add(compressed_split_mix.component(1), weight_1)
            group_indexes.append(new_mappings_1)

            sub_mix_0 = Mixture.from_components(
                [mix.component(i) for i in new_mappings_0],
                [mix.weight(i) for i in new_mappings_0],
                dims,
            )
            approx_mix_0 = Mixture.from_components(
                [compressed_mixture.component(max_idx)],
                [compressed_mixture.weight(max_idx)],
                dims,
            )
            hell_dists[max_idx] = hellinger_distance(approx_mix_0, sub_mix_0)

            last_idx = len(compressed_mixture) - 1
            sub_mix_1 = Mixture.from_components(
                [mix.component(i) for i in new_mappings_1],
                [mix.weight(i) for i in new_mappings_1],
                dims,
            )
            approx_mix_1 = Mixture.from_components(
                [compressed_mixture.component(last_idx)],
                [compressed_mixture.weight(last_idx)],
                dims,
            )
            hell_dists.append(hellinger_distance(approx_mix_1, sub_mix_1))

        else:
            raise RuntimeError("K-Means split should result in 1 or 2 components only.")

    return group_indexes

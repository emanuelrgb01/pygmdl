import numpy as np
from typing import List, Generic, TypeVar
import copy

from .kde_utils import VectorType, MatrixType, DataType

PDF_T = TypeVar("PDF_T")


class Mixture(Generic[PDF_T]):
    """
    Represents a mixture model, a weighted sum of probability distribution components.
    """

    def __init__(self, dims: int):
        """
        Initializes an empty mixture model.

        Args:
            dims: The number of dimensions for the data space.
        """
        self._dims: int = dims
        self._components: List[PDF_T] = []
        self._weights: List[DataType] = []

    @classmethod
    def from_components(
        cls, components: List[PDF_T], weights: List[DataType], dims: int
    ) -> "Mixture[PDF_T]":
        """
        A factory to create a Mixture instance directly from lists of components and weights.

        Args:
            components: A list of component objects.
            weights: A list of corresponding weights.
            dims: The dimensionality of the mixture.

        Returns:
            A new Mixture instance populated with the given components.
        """
        if len(components) != len(weights):
            raise ValueError(
                "The number of components must match the number of weights."
            )

        instance = cls(dims)
        instance._components = components
        instance._weights = weights

        return instance

    @classmethod
    def from_whitening(
        cls,
        to_whiten: "Mixture",
        smp_mean: VectorType,
        F_trns: MatrixType,
        eig_ok: List[int],
        ExplanationType: type,
    ) -> "Mixture":
        """
        A factory to create a new, whitened Mixture from an existing one.

        Args:
            to_whiten: The original mixture to be transformed.
            smp_mean: The sample mean used for centering during whitening.
            F_trns: The whitening transformation matrix.
            eig_ok: A list of indices for the valid (non-degenerate) dimensions.
            ExplanationType: The concrete class of the explanation to be created.

        Returns:
            A new Mixture instance containing the whitened components.
        """
        whitened_dims = len(eig_ok)
        whitened_mix = cls(whitened_dims)

        for component, weight in zip(to_whiten.components, to_whiten.weights):
            whitened_component = ExplanationType.from_whitening(
                to_whiten=component, smp_mean=smp_mean, F_trns=F_trns, eig_ok=eig_ok
            )
            whitened_mix.add(whitened_component, weight)

        return whitened_mix

    @property
    def dims(self) -> int:
        """Returns the number of dimensions of the mixture model."""
        return self._dims

    @property
    def components(self) -> List[PDF_T]:
        """Returns the list of components in the mixture."""
        return self._components

    @property
    def weights(self) -> np.ndarray:
        """Returns the weights of the components as a NumPy array."""
        return np.array(self._weights)

    def __len__(self) -> int:
        """Returns the number of components in the mixture."""
        return len(self._components)

    def __str__(self) -> str:
        """Provides a string representation of the mixture model."""
        header = f"Mixture Model (dims={self.dims}, components={len(self)})\n"
        lines = []
        for i in range(len(self)):
            lines.append(
                f"  Component[{i}]: weight={self.weight(i):.6f}\n    {str(self.component(i))}"
            )
        return header + "\n".join(lines)

    def reset(self) -> None:
        """Clears all components and weights from the mixture."""
        self._components.clear()
        self._weights.clear()

    def add(self, pdf: PDF_T, weight: DataType = 1.0) -> None:
        """
        Adds a new component and its corresponding weight to the mixture.
        """
        self._components.append(pdf)
        self._weights.append(weight)

    def add_all(self, other: "Mixture") -> None:
        """Adds all components from another mixture into this one."""
        for i in range(len(other)):
            self.add(copy.deepcopy(other.component(i)), other.weight(i))

    def replace(self, index: int, pdf: PDF_T, weight: DataType = 1.0) -> None:
        """Replaces a component and weight at a specific index."""
        if not (0 <= index < len(self)):
            raise IndexError("Index out of bounds")
        self._components[index] = pdf
        self._weights[index] = weight

    def component(self, index: int) -> PDF_T:
        """Returns the component at a specific index."""
        return self._components[index]

    def weight(self, index: int) -> DataType:
        """Returns the weight of the component at a specific index."""
        return self._weights[index]

    def set_weight(self, index: int, new_weight: DataType) -> None:
        """Sets the weight of the component at a specific index."""
        self._weights[index] = new_weight

    def sum_of_weights(self) -> DataType:
        """Calculates the sum of all component weights."""
        return np.sum(self._weights)

    def normalize_weights_preserve_relative_importance(self) -> None:
        """Normalizes weights so they sum to 1, while preserving their relative proportions."""
        total = self.sum_of_weights()
        if total > 0:
            self._weights = [w / total for w in self._weights]

    def normalize_weights(self) -> None:
        """Normalizes weights to be uniform (1/N)."""
        if not self._components:
            return
        self._weights = [1.0 / len(self)] * len(self)

    def scale_weights(self, factor: DataType) -> None:
        """Multiplies all weights by a given factor."""
        self._weights = [w * factor for w in self._weights]

    def mean(self, index: int) -> VectorType:
        """Returns the mean of the component at a specific index."""
        return self._components[index].mean

    def covariance(self, index: int) -> MatrixType:
        """Returns the covariance of the component at a specific index."""
        return self._components[index].covariance

    def likelihood(self, sample: VectorType) -> DataType:
        """Calculates the likelihood of a given sample."""
        return np.sum(
            [
                w * np.exp(c.log_likelihood(sample))
                for w, c in zip(self._weights, self._components)
            ]
        )

    def convolve(self, bandwidth: MatrixType) -> None:
        """Convolves each component in the mixture with a given bandwidth matrix."""
        for component in self._components:
            component.convolve(bandwidth)

    def deconvolve(self, bandwidth: MatrixType) -> None:
        """Deconvolves each component in the mixture with a given bandwidth matrix."""
        for component in self._components:
            component.deconvolve(bandwidth)

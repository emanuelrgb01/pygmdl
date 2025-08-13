import numpy as np
from typing import List, Dict, Any, Optional

from . import kde

type KDE_Class = kde.OKDEDiagonal[kde.GaussianDiagonal]

Sample = Dict[str, Any]
Prediction = Dict[str, Any]


class GMDL:
    """
    Gaussian Mixture Descriptors Learner.

    This class implements an online learning algorithm that uses Online Kernel
    Density Estimators (oKDE) to model the probability distribution of each
    class. It classifies new samples based on the Minimum Description Length
    (MDL) principle.
    """

    def __init__(self, n_classes: int, n_dims: int, seed: int = 42):
        if n_classes < 2 or n_dims < 1:
            raise ValueError(
                "Number of classes must be >= 2 and dimensions must be >= 1"
            )

        self.n_classes = n_classes
        self.n_dims = n_dims
        self.rng = np.random.default_rng(seed)

        # Hyperparameters with default values from GMDL C++ version
        self.omega = 2**-32
        self.sigma = 1.0
        self.forgetting_factor = 1.0
        self.eta = 0.01
        self.alpha = 0.9
        self.tau = 1.0

        self.max_theta = 0.999999999
        self.min_theta = 2**-32
        self.theta = np.full(n_dims, self.max_theta, dtype=float)
        self.gradients = np.zeros(n_dims, dtype=float)

        # KDE_Class = kde.OKDEDiagonal[kde.GaussianDiagonal]

        self.distributions: Dict[int, List[KDE_Class]] = {
            c: [KDE_Class(dims=1) for _ in range(n_dims)] for c in range(n_classes)
        }

        self.class_distributions: Dict[int, KDE_Class] = {
            c: KDE_Class(dims=n_dims) for c in range(n_classes)
        }

        self.samples_per_attr: Dict[int, np.ndarray] = {
            c: np.zeros(n_dims, dtype=np.int64) for c in range(n_classes)
        }
        self.means_per_attr: Dict[int, np.ndarray] = {
            c: np.zeros(n_dims, dtype=float) for c in range(n_classes)
        }
        self.var_acc_per_attr: Dict[int, np.ndarray] = {
            c: np.zeros(n_dims, dtype=float) for c in range(n_classes)
        }

    def set_omega(self, omega: float) -> None:
        """Set the floor probability value (from -log2)."""
        self.omega = 2**-omega

    def set_tau(self, tau: float) -> None:
        """Set the class prototype distance impact."""
        self.tau = tau

    def set_sigma(self, sigma: float) -> None:
        """Set the standard deviation for noise injection."""
        self.sigma = sigma

    def set_learning_rate(self, eta: float) -> None:
        """Set the learning rate for feature weight updates."""
        self.eta = eta

    def set_momentum(self, alpha: float) -> None:
        """Set the momentum for feature weight updates."""
        self.alpha = alpha

    def set_forgetting_factor(self, factor: float) -> None:
        """Set the forgetting factor for the online KDEs."""
        self.forgetting_factor = factor
        for c in range(self.n_classes):
            self.class_distributions[c].forgetting_factor = factor
            for attr_kde in self.distributions[c]:
                attr_kde.forgetting_factor = factor

    def train(
        self, features: np.ndarray, label: int, prediction: Optional[int] = None
    ) -> None:
        """
        Trains the model with a single sample.

        Args:
            features: A 1D NumPy array of feature values.
            label: The integer class label for the sample.
            prediction: If provided, the model will also update its feature
                        weights (Theta) based on this prediction.
        """
        if features.shape[0] != self.n_dims:
            raise ValueError(
                f"Feature vector has wrong dimension {features.shape[0]}, expected {self.n_dims}"
            )

        noisy_features = self._add_noise_if_needed(features, label)

        if self.tau != 0:
            self.class_distributions[label].add_sample(noisy_features)

        for attr_idx in range(self.n_dims):
            feature_1d = np.array([noisy_features[attr_idx]])
            self.distributions[label][attr_idx].add_sample(feature_1d)

        self._estimate_kernel_densities(label)

        if prediction is not None:
            self._update_theta(features, label, prediction)

    def predict(self, features: np.ndarray) -> Prediction:
        """
        Predicts the class label for a single sample.

        Args:
            features: A 1D NumPy array of feature values.

        Returns:
            A Prediction dictionary containing the predicted 'label' and the
            'description_lengths' for each class.
        """
        if features.shape[0] != self.n_dims:
            raise ValueError(
                f"Feature vector has wrong dimension {features.shape[0]}, expected {self.n_dims}"
            )

        description_lengths = np.array(
            [self._L(features, c) for c in range(self.n_classes)]
        )

        predicted_label = np.argmin(description_lengths)

        return {"label": predicted_label, "description_lengths": description_lengths}

    def _add_noise_if_needed(self, features: np.ndarray, label: int) -> np.ndarray:
        """
        Adds a small amount of Gaussian noise to features if their inclusion
        would lead to a degenerate (zero) covariance in an attribute's KDE.
        Also updates the running statistics for this check.
        """
        noisy_features = features.copy()

        for i in range(self.n_dims):
            samples = self.samples_per_attr[label][i]
            mean = self.means_per_attr[label][i]
            var_acc = self.var_acc_per_attr[label][i]
            value = noisy_features[i]

            while self._is_covariance_degenerate(samples + 1, mean, var_acc, value):
                noise = self.rng.normal(0, self.sigma)
                value += noise

            noisy_features[i] = value

            samples += 1
            delta = value - mean
            mean += delta / samples
            delta2 = value - mean
            var_acc += delta * delta2

            self.samples_per_attr[label][i] = samples
            self.means_per_attr[label][i] = mean
            self.var_acc_per_attr[label][i] = var_acc

        return noisy_features

    def _is_covariance_degenerate(
        self, n_samples: int, mean: float, var_acc: float, new_value: float
    ) -> bool:
        if n_samples < 2:
            return False

        delta = new_value - mean
        new_mean = mean + delta / n_samples
        delta2 = new_value - new_mean
        new_var_acc = var_acc + delta * delta2

        variance = new_var_acc / (n_samples - 1)

        return np.isclose(variance, 0.0)

    def _estimate_kernel_densities(self, class_index: int) -> None:
        """Triggers the bandwidth estimation for a class's KDEs if they are large enough."""
        if self.tau != 0:
            pdf = self.class_distributions[class_index]
            if len(pdf) >= 3:
                pdf.estimate_kernel_density()

        for attr_pdf in self.distributions[class_index]:
            if len(attr_pdf) >= 3:
                attr_pdf.estimate_kernel_density()

    def _L(self, features: np.ndarray, class_index: int) -> float:
        """Calculates the final description length for a class."""
        distances = self._get_distances(features)
        total_dl = self._L_hat(features, class_index, distances) / self._L_total(
            features, distances
        )
        return total_dl

    def _L_hat(
        self, features: np.ndarray, class_index: int, distances: np.ndarray
    ) -> float:
        """Calculates the unnormalized description length."""
        dl = 0.0
        for attr_idx in range(self.n_dims):
            dl += self._L_hat_attribute(features[attr_idx], class_index, attr_idx)

        lm = self._LM(class_index)
        lm_log = np.floor(np.log2(lm)) if lm > 0 else 0

        return max(0.0, dl * distances[class_index] + lm_log)

    def _L_hat_attribute(self, value: float, class_index: int, attr_idx: int) -> float:
        """Calculates the description length for a single attribute."""
        pdf = self.distributions[class_index][attr_idx]
        sample = np.array([value])

        density = pdf.likelihood(sample)

        if np.isclose(density, 0.0) or np.isinf(density) or np.isnan(density):
            density = self.omega

        density = min(density, 1.0)

        cost = -np.log2(density + 1e-32)

        return cost ** self.theta[attr_idx]

    def _L_total(self, features: np.ndarray, distances: np.ndarray) -> float:
        """Calculates the total description length over all classes for normalization."""
        total_dl = 0.0
        for c in range(self.n_classes):
            dl = self._L_hat(features, c, distances)
            if not np.isinf(dl):
                total_dl += dl
        return total_dl if total_dl > 0 else 1.0

    def _LM(self, class_index: int) -> int:
        """Calculates the model complexity (LM) term."""
        complexity = 1
        if self.tau != 0:
            complexity += len(self.class_distributions[class_index])

        for attr_pdf in self.distributions[class_index]:
            complexity += len(attr_pdf)

        return complexity

    def _update_theta(self, features: np.ndarray, label: int, prediction: int) -> None:
        """Updates the feature weights (Theta) using gradient descent with momentum."""
        norm = self._L(features, prediction)
        kronecker = 1 if prediction == label else 0

        for i in range(self.n_dims):
            partial = self._L_hat_attribute(features[i], prediction, i)
            grad = partial * np.log(self.theta[i]) * (1 - norm) * (kronecker - norm)

            self.gradients[i] = self.eta * -grad + self.alpha * self.gradients[i]
            self.theta[i] -= self.gradients[i]

        self.theta = np.clip(self.theta, self.min_theta, self.max_theta)

    def _get_distances(self, features: np.ndarray) -> np.ndarray:
        """Calculates the normalized Mahalanobis distance to each class prototype."""
        if self.tau == 0:
            return np.ones(self.n_classes)

        distances = np.full(self.n_classes, np.inf)

        for c in range(self.n_classes):
            pdf = self.class_distributions[c]
            if len(pdf) == 0:
                continue

            mu_bar = np.sum(
                [comp.mean * w for comp, w in zip(pdf.components, pdf.weights)], axis=0
            )

            variance = 0.0
            for comp, w in zip(pdf.components, pdf.weights):
                mean_diff = comp.mean - mu_bar
                variance += w * (mean_diff.T @ mean_diff)

            S_diag = np.sum(
                [comp.covariance * w for comp, w in zip(pdf.components, pdf.weights)],
                axis=0,
            )

            S_total_diag = S_diag + variance

            if np.any(np.isclose(S_total_diag, 0.0)):
                continue

            S_total_inv_diag = 1.0 / S_total_diag

            x_minus_mu = features - mu_bar
            dist_sq = np.sum(x_minus_mu**2 * S_total_inv_diag)
            distances[c] = np.sqrt(dist_sq)

        valid_distances = distances[np.isfinite(distances)]
        if len(valid_distances) == 0:
            return np.ones(self.n_classes)

        min_dist, max_dist = np.min(valid_distances), np.max(valid_distances)

        s = np.ones(self.n_classes)
        for c in range(self.n_classes):
            if np.isfinite(distances[c]):
                if np.isclose(max_dist, min_dist):
                    normalized = 0.0
                else:
                    normalized = (distances[c] - min_dist) / (max_dist - min_dist)

                cost = -np.log2(0.5 * (1 - normalized) + 1e-32)
                s[c] = cost**self.tau

        return s

    def diagnose(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Provides a dictionary with internal state information for a given sample,
        useful for debugging.

        Args:
            features: The feature vector to diagnose.

        Returns:
            A dictionary containing description lengths, theta weights, and distances.
        """
        distances = self._get_distances(features)
        description_lengths = np.array(
            [self._L_hat(features, c, distances) for c in range(self.n_classes)]
        )

        total_dl = self._L_total(features, distances)
        final_lengths = description_lengths / total_dl

        return {
            "description_lengths": final_lengths,
            "theta": self.theta,
            "distances_S": distances,
        }

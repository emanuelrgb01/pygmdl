# This script demonstrates how to use the pygmdl library in an online learning scenario
#
# Expected Input Format (line by line from stdin):
# <Token>
# feature1,feature2,...,featureN,class_name
#
# Valid Tokens:
# <Training>     - The following line is a training sample.
# <Test>         - The following line is a test sample to be classified.
# <Correction>   - The following line is a correction for the previous misclassification.
#                  The format is: feature1,...,featureN,true_class,predicted_class_id

import argparse
import sys
import numpy as np

from typing import Tuple, Optional

from pygmdl import GMDL, SampleType, dataset_utils


def print_debug_info(
    iteration: int,
    features: np.ndarray,
    true_label: int,
    prediction: dict,
    model: GMDL,
):
    """
    Prints detailed debug information about a misclassification by calling the
    model's diagnose method.
    """
    diagnostics = model.diagnose(features)

    print(f"--- Debug Info (Iteration: {iteration}) ---", file=sys.stderr)
    print(f"Data: {features}", file=sys.stderr)
    print(f"Description Lengths: {diagnostics['description_lengths']}", file=sys.stderr)
    print(f"Theta: {diagnostics['theta']}", file=sys.stderr)
    print(f"Distances (S): {diagnostics['distances_S']}", file=sys.stderr)

    predicted_label = prediction["label"]
    dl_predicted = diagnostics["description_lengths"][predicted_label]
    dl_true = diagnostics["description_lengths"][true_label]
    diff = abs(dl_predicted - dl_true)

    print(f"Predicted: {predicted_label}, Expected: {true_label}", file=sys.stderr)
    print(f"DL Difference: {diff}", file=sys.stderr)
    print("------------------------------------------", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="GMDL Online Classifier")

    # Arguments
    parser.add_argument(
        "--labels",
        required=True,
        type=str,
        help='Comma-separated class labels (e.g., "cat,dog,fish")',
    )
    parser.add_argument(
        "--dimension", required=True, type=int, help="Number of features in the dataset"
    )
    parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        help="Print confusion matrix at the end",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Omit logging when classification fails",
    )

    # Hyperparameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for theta updates",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for theta updates"
    )
    parser.add_argument(
        "--tau", type=float, default=1.0, help="Impact of class prototype distance"
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=32.0,
        help="-log2 of the default probability for unseen events",
    )
    parser.add_argument(
        "--forgetting_factor",
        type=float,
        default=1.0,
        help="Forgetting factor for sample weighting (1.0 = no forgetting)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Std deviation of noise added to prevent degenerate covariances",
    )

    args = parser.parse_args()

    class_names = args.labels.split(",")
    class_map = {name: i for i, name in enumerate(class_names)}
    n_classes = len(class_names)

    model = GMDL(n_classes=n_classes, n_dims=args.dimension)

    model.set_learning_rate(args.learning_rate)
    model.set_momentum(args.momentum)
    model.set_tau(args.tau)
    model.set_omega(args.omega)
    model.set_forgetting_factor(args.forgetting_factor)
    model.set_sigma(args.sigma)

    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    iteration = 0

    data_stream = dataset_utils.load_online_stream(sys.stdin, class_map)

    for sample_type, features, true_label, correction_label in data_stream:
        iteration += 1

        if sample_type == SampleType.TRAINING:
            model.train(features, true_label)

        elif sample_type == SampleType.CORRECTION:
            model.train(features, true_label, prediction=correction_label)

        elif sample_type == SampleType.TEST:
            prediction = model.predict(features)
            predicted_label = prediction["label"]

            print(class_names[predicted_label], flush=True)

            confusion_matrix[true_label, predicted_label] += 1

            if predicted_label != true_label and not args.quiet:
                print_debug_info(iteration, features, true_label, prediction, model)

    if args.confusion_matrix:
        print("\n--- Confusion Matrix ---", file=sys.stderr)
        header = " ".join(class_names)
        print(f"Actual\\Predicted | {header}", file=sys.stderr)
        for i, row in enumerate(confusion_matrix):
            print(f"{class_names[i]:<16} | {' '.join(map(str, row))}", file=sys.stderr)


if __name__ == "__main__":
    main()

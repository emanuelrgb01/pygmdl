# This script demonstrates how to use the pygmdl library in a batch learning scenario
#
# Example Usage:
# python -m examples.run_batch --training-file path/to/train.csv --testing-file path/to/test.csv --labels cat,dog --label-column class_name

import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

import pygmdl


def main():
    """Main function to run the batch GMDL classifier."""
    parser = argparse.ArgumentParser(description="GMDL Batch Classifier")

    parser.add_argument(
        "--training-file", required=True, type=str, help="Path to the training CSV file"
    )
    parser.add_argument(
        "--testing-file", required=True, type=str, help="Path to the testing CSV file"
    )
    parser.add_argument(
        "--label-column",
        required=True,
        type=str,
        help="The name of the column containing the class label",
    )

    parser.add_argument(
        "--labels",
        type=str,
        help='Comma-separated class labels in the desired order (e.g., "cat,dog,fish")',
    )
    parser.add_argument("--separator", type=str, default=",", help="CSV file separator")
    parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        help="Print confusion matrix and classification report at the end",
    )

    # Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=32.0)
    parser.add_argument("--forgetting_factor", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=1.0)

    args = parser.parse_args()

    try:
        print(f"Loading training data from '{args.training_file}'...")
        train_df = pd.read_csv(args.training_file, sep=args.separator)
        print(f"Loading testing data from '{args.testing_file}'...")
        test_df = pd.read_csv(args.testing_file, sep=args.separator)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if (args.label_column not in train_df.columns) or (
        args.label_column not in test_df.columns
    ):
        print(
            f"Error: Label column '{args.label_column}' not found in one or both files.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.labels:
        class_names = args.labels.split(",")
    else:
        class_names = (
            train_df[args.label_column].astype("category").cat.categories.tolist()
        )

    class_map = {name: i for i, name in enumerate(class_names)}
    n_classes = len(class_names)
    n_dims = len(train_df.columns) - 1

    print(f"Found {n_classes} classes: {class_names}")
    print(f"Data has {n_dims} dimensions.")

    model = pygmdl.GMDL(n_classes=n_classes, n_dims=n_dims)

    model.set_learning_rate(args.learning_rate)
    model.set_momentum(args.momentum)
    model.set_tau(args.tau)
    model.set_omega(args.omega)
    model.set_forgetting_factor(args.forgetting_factor)
    model.set_sigma(args.sigma)

    print(f"\n--- Training on {len(train_df)} samples ---")

    X_train = train_df.drop(args.label_column, axis=1).to_numpy()
    y_train_names = train_df[args.label_column].to_numpy()

    for i in range(len(X_train)):
        features = X_train[i]
        label = class_map[y_train_names[i]]
        model.train(features, label)

    print("Training complete.")

    print(f"\n--- Testing on {len(test_df)} samples ---")

    X_test = test_df.drop(args.label_column, axis=1).to_numpy()
    y_test_names = test_df[args.label_column].to_numpy()

    true_labels = []
    predicted_labels = []

    for i in range(len(X_test)):
        features = X_test[i]
        true_label = class_map[y_test_names[i]]

        prediction = model.predict(features)
        predicted_label = prediction["label"]

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    print("Testing complete.")

    predicted_class_names = [class_names[p] for p in predicted_labels]
    print("\n--- Predictions ---")
    print(", ".join(predicted_class_names))

    if args.confusion_matrix:
        print("\n--- Evaluation Report ---")

        cm = confusion_matrix(true_labels, predicted_labels)
        report = classification_report(
            true_labels, predicted_labels, target_names=class_names, zero_division=0
        )

        print("Confusion Matrix:")
        print("Actual \\ Predicted")
        print(pd.DataFrame(cm, index=class_names, columns=class_names))
        print("\nClassification Report:")
        print(report)


if __name__ == "__main__":
    main()

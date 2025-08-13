import pandas as pd
import numpy as np
import sys
from enum import Enum
from typing import Iterator, Tuple, List, TextIO, Dict, Optional

Sample = Tuple[np.ndarray, int]


class SampleType(Enum):
    """Enumeration for the type of sample being processed in online mode."""

    TRAINING = 1
    TEST = 2
    CORRECTION = 3


def load_online_stream(
    stream: TextIO, class_map: Dict[str, int]
) -> Iterator[Tuple[SampleType, np.ndarray, int, Optional[int]]]:
    """
    A generator that reads from a stream (stdin) line by line,
    parses the online protocol with tokens like <Training>, <Test>,
    and yields samples with their type.

    Yields:
        A tuple of (SampleType, features, true_label, optional_correction_label).
    """
    for line in stream:
        token = line.strip()

        try:
            if token == "<Training>":
                data_line = next(stream).strip()
                parts = data_line.split(",")
                features = np.array([float(p) for p in parts[:-1]], dtype=float)
                label_name = parts[-1]
                yield SampleType.TRAINING, features, class_map[label_name], None

            elif token == "<Test>":
                data_line = next(stream).strip()
                parts = data_line.split(",")
                features = np.array([float(p) for p in parts[:-1]], dtype=float)
                label_name = parts[-1]
                yield SampleType.TEST, features, class_map[label_name], None

            elif token == "<Correction>":
                data_line = next(stream).strip()
                parts = data_line.split(",")
                features = np.array([float(p) for p in parts[:-2]], dtype=float)
                true_label_name = parts[-2]
                predicted_label_id = int(parts[-1])
                yield SampleType.CORRECTION, features, class_map[
                    true_label_name
                ], predicted_label_id

        except (StopIteration, ValueError, KeyError, IndexError) as e:
            print(
                f"Warning: Skipping malformed data stream entry near token '{token}'. Error: {e}",
                file=sys.stderr,
            )
            continue


def _process_dataframe(df: pd.DataFrame, label_col: str) -> Iterator[Sample]:
    """A generator that processes a DataFrame and yields samples."""

    label_codes = df[label_col].astype("category").cat.codes
    feature_cols = df.drop(label_col, axis=1)

    for i in range(len(df)):
        features = feature_cols.iloc[i].to_numpy()
        label = label_codes.iloc[i]
        yield (features, label)


def load_from_file(
    filepath: str, label_col_name: str, separator: str = ","
) -> Iterator[Sample]:
    """Loads a dataset from a file and returns it as an iterator of samples."""
    df = pd.read_csv(filepath, sep=separator)
    yield from _process_dataframe(df, label_col_name)


def load_from_stream(
    stream: TextIO, class_names: List[str], label_col: int = -1, separator: str = ","
) -> Iterator[Sample]:
    """Reads a dataset from a stream (like sys.stdin) line by line."""
    class_map = {name: i for i, name in enumerate(class_names)}

    for line in stream:
        line = line.strip()
        if not line:
            continue

        parts = line.split(separator)
        label_idx = label_col if label_col != -1 else len(parts) - 1

        try:
            label_name = parts.pop(label_idx)
            features = np.array([float(p) for p in parts])
            label_id = class_map[label_name]
            yield (features, label_id)
        except (ValueError, KeyError, IndexError) as e:
            print(f"Warning: Skipping malformed line: '{line}' -> {e}", file=sys.stderr)
            continue

import pandas as pd
import numpy as np
import sys
from typing import Iterator, Tuple, List, TextIO

Sample = Tuple[np.ndarray, int]


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

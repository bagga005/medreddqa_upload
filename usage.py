import csv
import os
import json
from datasets import load_dataset, Dataset
from typing import Iterable, List, Tuple

DATA_DIR = os.path.join("data", "medredqa")

# Expected MedRedQA schema used during upload
FEATURES: List[Tuple[str, str]] = [
    ("id", "string"),
    ("title", "string"),
    ("body", "string"),
    ("response", "string"),
    ("response_score", "float32"),
    ("occupation", "string"),
    ("pmcids", "string"),
]

SPLIT_FILES: List[Tuple[str, str]] = [
    ("train", "medredqa_train.csv"),
    ("test", "medredqa_test.csv"),
    ("validation", "medredqa_val.csv"),
]


def count_examples(csv_path: str) -> int:
    """Count rows in a CSV file, skipping the header line."""
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # drop header row
        return sum(1 for _ in reader)


def emit_features(features: Iterable[Tuple[str, str]]) -> None:
    print("dataset_info:")
    print("  features:")
    for name, dtype in features:
        print(f"  - name: {name}")
        print(f"    dtype: {dtype}")


def emit_splits(split_counts: Iterable[Tuple[str, int]]) -> None:
    print("  splits:")
    for name, count in split_counts:
        print(f"  - name: {name}")
        print(f"    num_examples: {count}")


def print_info() -> None:
    split_metadata: List[Tuple[str, int]] = []
    total_size = 0

    for split_name, filename in SPLIT_FILES:
        csv_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(csv_path):
            print(f"# Warning: {csv_path} not found; skipping {split_name}")
            continue

        num_examples = count_examples(csv_path)
        file_size = os.path.getsize(csv_path)

        split_metadata.append((split_name, num_examples))
        total_size += file_size

    emit_features(FEATURES)
    emit_splits(split_metadata)
    print(f"  download_size: {total_size}")
    print(f"  dataset_size: {total_size}")

def simple_use() -> None:
    # load only train and validation splits
    medredqa_train, medredqa_eval = load_dataset("bagga005/medredqa", split=["train", "validation"])
    print("# of train:", len(medredqa_train), "# of validation:", len(medredqa_eval))
    
    print("\nmedredqa_train\n", json.dumps(medredqa_train[0], indent=2))
    print("\nmedredqa_eval\n", json.dumps(medredqa_eval[0], indent=2))

if __name__ == "__main__":
    print_info()
    simple_use()

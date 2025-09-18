#!/usr/bin/env python3
import argparse
import csv
import os
from typing import Dict, List

from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import create_repo


def str_to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}

parser = argparse.ArgumentParser(description="Format MedRedQA data and push to the Hugging Face Hub")
parser.add_argument("--hf_username", default=os.environ.get("HF_USERNAME", "bagga005"))
parser.add_argument("--hf_repo_name", default=os.environ.get("HF_REPO_NAME", "MedRedQA"))
parser.add_argument("--private", default=os.environ.get("PRIVATE", "false"),
                    help="Whether the HF dataset repo should be private (true/false)")
args = parser.parse_args()

HF_USERNAME = args.hf_username
HF_REPO_NAME = args.hf_repo_name
PRIVATE = str_to_bool(args.private)
HF_REPO_ID = f"{HF_USERNAME}/{HF_REPO_NAME}"

DATA_DIR = "data"
MEDREDQA_FEATURES = Features({
    "id": Value("string"),
    "title": Value("string"),
    "body": Value("string"),
    "response": Value("string"),
    "response_score": Value("float32"),
    "occupation": Value("string"),
    "pmcids": Value("string"),
})

def _to_str(x):
    return "" if x is None else str(x)

def load_medredqa_csv(file_path: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    seen_ids = set()

    with open(file_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            score_raw = row.get("Response Score", "")
            try:
                score = float(score_raw) if score_raw not in {"", None} else None
            except ValueError:
                score = None

            raw_id = row.get("") or row.get("ID")
            record_id = _to_str(raw_id)
            if not record_id:
                record_id = f"row_{idx:06d}"

            if record_id in seen_ids:
                deduped = f"{record_id}_{idx:06d}"
                print(f"Duplicate id '{record_id}' detected at row {idx}; using '{deduped}' instead")
                record_id = deduped

            seen_ids.add(record_id)

            record = {
                "id": record_id,
                "title": _to_str(row.get("Title")),
                "body": _to_str(row.get("Body")),
                "response": _to_str(row.get("Response")),
                "response_score": score,
                "occupation": _to_str(row.get("Occupation")),
                "pmcids": _to_str(row.get("PMCID(s)")),
            }

            records.append(record)

    return records

if __name__ == "__main__":
    splits_config = {
        "train": "medredqa_train.csv",
        "validation": "medredqa_val.csv",
        "test": "medredqa_test.csv",
    }

    medredqa_splits = {}

    for split_name, filename in splits_config.items():
        file_path = os.path.join(DATA_DIR, "medredqa", filename)
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found; skipping {split_name} split")
            continue

        records = load_medredqa_csv(file_path)
        print(f"Loaded {len(records)} examples from {file_path}")

        medredqa_splits[split_name] = Dataset.from_list(records, features=MEDREDQA_FEATURES)

    if not medredqa_splits:
        raise FileNotFoundError("No MedRedQA splits were found under data/medredqa")

    print(f"\nPushing dataset to Hugging Face Hub as {HF_REPO_ID} (private={PRIVATE})...")
    create_repo(HF_REPO_ID, repo_type="dataset", private=PRIVATE, exist_ok=True)

    dataset_dict = DatasetDict(medredqa_splits)
    dataset_dict.push_to_hub(HF_REPO_ID, private=PRIVATE)
    print(f"Dataset pushed to https://huggingface.co/datasets/{HF_REPO_ID}")

---
dataset_info:
  features:
  - name: id
    dtype: string
  - name: title
    dtype: string
  - name: body
    dtype: string
  - name: response
    dtype: string
  - name: response_score
    dtype: float32
  - name: occupation
    dtype: string
  - name: pmcids
    dtype: string
  splits:
  - name: train
    num_bytes: 59383722
    num_examples: 40792
  - name: validation
    num_bytes: 7307243
    num_examples: 5100
  - name: test
    num_bytes: 7343299
    num_examples: 5099
  download_size: 46290063
  dataset_size: 74034264
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

# MedRedQA

HuggingFace upload of a QA dataset of questions and responses from qualified medical professionals sourced from [MedRedQA](https://data.csiro.au/collection/csiro:62454). If used, please cite the original authors using the citation below.

## Dataset Details

### Dataset Description

The dataset contains three splits:
  - **train**: 40792 questions and responses from a qualified expert
  - **test**: 5099 questions and responses from a qualified expert
  - **validation**: 5100 questions and responses from a qualified expert

### Dataset Sources

- **Repository:** https://data.csiro.au/collection/csiro:62454
- **Paper:** https://aclanthology.org/2023.ijcnlp-main.42/

### Direct Use

```python
import json
from datasets import load_dataset, Dataset

if __name__ == "__main__":
    # load only train and validation splits
    medredqa_train, medredqa_eval = load_dataset("bagga005/medredqa", split=["train", "validation"])
    print("# of train:", len(medredqa_train), "# of validation:", len(medredqa_eval))
    
    print("\nmedredqa_train\n", json.dumps(medredqa_train[0], indent=2))
    print("\nmedredqa_eval\n", json.dumps(medredqa_eval[0], indent=2))
```



## Citation 

```
@inproceedings{nguyen-etal-2023-medredqa,
    title = "{M}ed{R}ed{QA} for Medical Consumer Question Answering: Dataset, Tasks, and Neural Baselines",
    author = "Nguyen, Vincent  and
      Karimi, Sarvnaz  and
      Rybinski, Maciej  and
      Xing, Zhenchang",
    editor = "Park, Jong C.  and
      Arase, Yuki  and
      Hu, Baotian  and
      Lu, Wei  and
      Wijaya, Derry  and
      Purwarianti, Ayu  and
      Krisnadhi, Adila Alfa",
    booktitle = "Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = nov,
    year = "2023",
    address = "Nusa Dua, Bali",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.ijcnlp-main.42/",
    doi = "10.18653/v1/2023.ijcnlp-main.42",
    pages = "629--648"
}
```
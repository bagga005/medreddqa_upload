**How to run:**

First, create the venv:
```
uv venv --python 3.12 --seed
source .venv/bin/activate
uv sync
```

Login to the HF CLI:
```sh
huggingface-cli login 
``` 
or set HF_TOKEN environment variable

And enter the configs you like:
```sh
chmod +x run.sh
./run.sh --username <username> \
         --repo MedRedQA \
         --private true
```

Voila! The dataset now lives [on HuggingFace](https://huggingface.co/datasets/<username>/medredqa.

All credit belongs to the [original authors](https://data.csiro.au/collection/csiro:62454)
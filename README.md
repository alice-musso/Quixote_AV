# Quixote Authorship Verification

This repository contains experiments for authorship verification focused on the Quixote corpus. The current codebase works on early modern Spanish texts, builds stylometric feature representations, and trains binary classifiers to distinguish a target author from all others.

## What Is In Scope

- Quixote authorship verification
- Cervantes vs. non-Cervantes classification
- Quijote topic-ablation experiments
- Corpus loading, spaCy-based preprocessing, segmentation, feature extraction, model training, and inference

## Project Structure

- `src/data_preparation`: corpus loading, caching, and segmentation
- `src/feature_extraction`: stylometric feature extractors
- `src/oversampling`: DRO oversampling utilities
- `src/authorship_verification.py`: feature preparation and model selection
- `src/inference.py`: main Quixote inference workflow
- `src/quijote_classifier/quijote_experiment.py`: Quijote topic-ablation utilities
- `corpus/training`: training texts
- `corpus/test`: test texts
- `hyperparams`: saved hyperparameters
- `results`: generated outputs

## Environment

The code currently expects:

- Python 3.11
- spaCy with `es_dep_news_trf`
- NLTK Spanish stopwords
- scikit-learn, scipy, numpy, pandas, tqdm

The checked-in `requirements.txt` is a conda export of one working environment, not a minimal dependency list.

## Setup Notes

Install the spaCy Spanish pipeline and the required NLTK data in your environment.

```python
import nltk
nltk.download("stopwords")
```

## Main Entry Point

```bash
cd src
python -m inference \
  --train-dir ../corpus/training \
  --test-dir ../corpus/test \
  --positive-author Cervantes \
  --classifier-type lr
```

Use `--no-load-hyperparams` to rerun model selection instead of loading a saved hyperparameter file.

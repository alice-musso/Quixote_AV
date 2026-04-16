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

Use `--skip-ablation` to bypass topic-feature removal, and `--no-skip-decision-changes` to enable the slower decision-flip tracing pass.

## Output Tables

Inference writes JSON and CSV tables under the configured results directory.

- `score`: one row per `phase` (`pre_ablation`, `post_ablation`), per `author`, and per `scope` (`books`, `segments`), with `accuracy`, `f1`, fold counts, and `model_selection_score`.
- `predictions`: one row per test book, with `title`, `author`, and one-vs-rest columns like `pre_ablation_pred_<author>` and `pre_ablation_score_<author>` plus the matching `post_ablation_*` columns.
- `ablation`: one row per deleted feature, including deletion order, original rank, feature index, and feature name.
- `decision_changes`: one row per test-book and classifier-author pair, recording whether the one-vs-rest prediction changed during sequential feature deletion and, if so, when it first flipped.

Older sample outputs with the previous multiclass-style schema are kept in `src/results/legacy/` for reference.

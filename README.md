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
  --classifier-type lr \
  --n-jobs 1
```

Use `--no-load-hyperparams` to rerun model selection instead of loading a saved hyperparameter file.

Use `--skip-ablation` to bypass topic-feature removal, and `--no-skip-decision-changes` to enable the slower decision-flip tracing pass.

Topic ablation considers every selected feature family for removal, including
frequent words. Features remain subject to the chosen ranking mode's
positive-information-gain criteria.

## Output Tables

Inference writes JSON and CSV tables under the configured results directory.

- `score`: one row per `phase` (`pre_ablation`, `post_ablation`), per `author`, and per `scope` (`books`, `segments`), with `accuracy`, `f1`, fold counts, and `model_selection_score`.
- `predictions`: one row per test book, with `title`, `author`, and one-vs-rest columns like `pre_ablation_pred_<author>` and `pre_ablation_score_<author>` plus the matching `post_ablation_*` columns.
- `ablation`: one row per deleted feature, including deletion order, original rank, feature index, and feature name.
- `decision_changes`: one row per test-book and classifier-author pair, recording whether the one-vs-rest prediction changed during sequential feature deletion and, if so, when it first flipped.

Older sample outputs with the previous multiclass-style schema are kept in `src/results/legacy/` for reference.

## Full topic information-gain distribution

To plot all features in the saved selected feature families, including positive,
negative, and zero scores, run from the repository root:

```bash
python src/analysis_deleted_features/plot_information_gain_distribution.py
```

This compares Quijote-titled works with other works by Cervantes using the same
signed information-gain calculation as topic ablation, before positive-candidate
filtering or deletion. Positive scores indicate association with Quijote;
negative scores indicate association with other Cervantes works. Ordinary
information gain is nonnegative; the project adds a sign based on feature
presence rates. Observations include complete books and their segments, matching
the ablation workflow. Scores measure feature presence, not TF-IDF magnitude.

Outputs in `results/information_gain/` include a histogram (logarithmic count axis),
PNG/PDF figures, a CSV containing every selected
feature and its score, and JSON metadata. This does not rerun classifier training
or model selection. Use `--hyperparams`, `--positive-author`, `--target-title`,
`--bins`, or `--output-dir` to customize the run. Existing preprocessing caches
are reused, as in the inference pipeline.

To compare Gaussian and Student's t fits using the saved score CSV (without
recomputing features):

```bash
python src/analysis_deleted_features/fit_information_gain_distribution.py
```

This writes `_fits.png`/`.pdf` histogram overlays, `_fits_qq.png`/`.pdf` Q–Q
plots, fitted parameters and AIC in `_fits.json`, and observed/expected bin
counts in `_fits.csv`. Both models are fitted using interval probabilities
for the same 80 histogram bins, including empty overflow bins. This avoids
assigning infinite measurement precision to the repeated zero scores. Use
`--bins` to change resolution or `--input-csv` for another score table.
Lower AIC indicates a better relative fit; it does not establish that either
model describes the data well. Feature dependence, the zero spike, asymmetry,
and binning affect interpretation. Student's t `scale` is not its standard
deviation. Q–Q plots use symmetric logarithmic axes to show center and tails.


## Book and segment performance reports

Each inference run also saves `results_<author>_<classifier>_book_report.csv` and
`results_<author>_<classifier>_segment_report.csv`, with equivalent JSON files,
next to the prediction table. Both standard and word-list ablation runs produce
these reports.

These reports cover the **training corpus under leave-one-book-out validation**,
with pre-ablation and post-ablation results side by side. Both reports have one
row per book. They reuse the
configured target-author verifier's held-out predictions, including DRO if selected;
they correspond to the target-verifier evaluation printed to the console. They do
not use the later per-author base-classifier fits that populate `score.csv`.
Existing caveats about fitting feature selection before cross-validation still apply.

The book report has exactly these columns, in order:

1. `title`
2. `actual_author`
3. `predicted_author_pre_ablation`
4. `predicted_author_post_ablation`

Each prediction is the target author's name or `Not<target>`. Predictions come
from the full-book feature vector, not a vote over its segments.

The segment report has exactly these columns, in order:

1. `title`
2. `actual_author`
3. `total_segments`
4. `segment_predicted_target_pre_ablation`
5. `segment_predicted_not_target_pre_ablation`
6. `segment_predicted_target_post_ablation`
7. `segment_predicted_not_target_post_ablation`

The pre- and post-ablation counts appear side by side on the same book row.
For each phase, the two counts sum to `total_segments`; the full-book prediction
is excluded from these counts. Books without segments have zero counts. Each
book's segments are held out together with its full-book row during validation.

The existing prediction table continues to cover the separate test corpus. The new
reports are generated on the next run; previously saved outputs are not retroactively
recomputed. When ablation is skipped, both report phases reuse the same evaluation.

## Annotated frequent-word and dependency ablation

Run this separate experiment from the repository root:

```bash
python src/annotated_word_ablation.py --n-jobs 1
```

It reads `corpus/Ablation K Frequency Words Annotated (1).csv` (semicolon
separated), selecting only rows whose `feature_deleted` column is `yes`
(case-insensitive, ignoring surrounding whitespace). It zeros matching
`feat_k_freq_words` and `feat_dep` columns in training and test matrices,
then retrains and evaluates. For example, `sancho` removes its frequent-word
feature and dependency features such as `Sancho:nsubj` and `Sancho:obj`.
Matching uses the existing word-list normalization (case/accent insensitive,
removing non-alphanumeric characters), with exact normalized words rather than
substrings. Other feature families are retained. Only features present in the
selected model representation can be removed; per-word counts are saved.

Outputs go to `results/annotated_word_ablation/`, including pre/post scores,
test predictions, book/segment reports, deleted features, per-word counts,
and `annotation_selection.json` recording the source and selected words.
Use `--list-words` to preview selection without training, `--words-file` to
choose another annotated CSV, or `--output-dir` to choose a results directory.
Saved hyperparameters are loaded by default, as in the main experiment.

## Secondary combined annotated-word and IG experiment

```bash
cd src
python combined_annotated_ig_ablation.py --n-jobs 1
```

This removes the union of:
- POS and Mendenhall feature names in the saved topic-ablation deletion table
  `result_legacy/results_2026-09-23_11-47-26_withFreqWordabl/inference/ablation.csv` (override with
  `--ig-ablation-file`). This is the deletion list, not every positive-IG feature.
- Words annotated `yes`, matched in dependency, function-word, and frequent-word
  families using the annotated experiment's normalization.

Features are matched by name against the current selected representation, not
by saved column indices. Mendenhall is not selected by the current saved
hyperparameters; it contributes no deletions. Missing IG features are recorded
in `ig_features_not_in_selected_model.json`. Outputs, including inference and
book/segment reports, go to `results/combined_annotated_ig_ablation/`.

## Archived results

Existing results directories were moved into `result_legacy/`. Archive names use
`results_YYYY-MM-DD_HH-MM-SS_<original-description>`, with timestamps in
Europe/Rome based on the latest file modification time in each directory.
These timestamps are estimates of the last update, not verified generation times;
some directories contain multiple runs. Internal directory names and file contents
are preserved. `result_legacy/archive_manifest.json` records original paths,
archive paths, timestamp sources, and file counts. New runs continue to write
to the configured `results/` location.

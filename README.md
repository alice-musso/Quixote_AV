# Authorship Verification for Medieval Latin 
This repository contains a system for authorship verification of medieval Latin texts, implementing feature extraction, distributional random oversampling, and classification algorithms to determine text authorship.

## Project Structure

The project is organized into the following modules:

- **data_preparation**: Contains tools for corpus loading and text segmentation
- **feature_extraction**: Implements various feature extraction methods for text analysis
- **oversampling**: Contains the Distributional Random Oversampling (DRO) algorithm and supporting functions
- **main.py**: Main execution script for the authorship verification system

## Requirements

This project requires Python 3.6+ and the following packages:

```
numpy
scikit-learn
scipy
spacy
nltk
cltk
tqdm
```

## Usage

The system can be run in three modes based on the `--test-document` argument:

1. Full Leave-One-Out:
This runs verification on all documents in the corpus.

2. Author-specific Leave-One-Out:
This only processes documents by the specified author.

3. Single Document Verification:
This verifies only the specified document.

The target author can be set with `--target`:
By default, target author is "Dante" and test-document is empty (full LOO).

# Authorship Verification for Medieval Latin 

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

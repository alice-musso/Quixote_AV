from typing import List, Tuple, Dict, Optional, Any, Union
import numpy as np
import spacy
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (
    f1_score, 
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    make_scorer
)
import csv
import time
from pathlib import Path
from tqdm import tqdm
import nltk

from src.data_preparation.data_loader import binarize_corpus

nltk.download('punkt_tab')
from nltk import sent_tokenize
from data_preparation.data_loader import load_corpus, get_spanish_function_words, Book
from data_preparation.segmentation import Segmentation
from feature_extraction.features import (
    FeaturesFunctionWords,
    FeaturesDistortedView,
    FeaturesMendenhall,
    FeaturesSentenceLength,
    FeaturesPOST,
    FeatureSetReductor,
    FeaturesDEP,
    FeaturesPunctuation,
    HstackFeatureSet,
)

import warnings
warnings.filterwarnings("ignore")

import time

QUIXOTE_DOCUMENTS = [
    'Avellaneda - Quijote apocrifo',
    'Avellaneda - Quijote apocrifo nucleo',
    'Avellaneda - Quijote apocrifo prologo',
    'Avellaneda - Quijote apocrifo prima novelle',
    'Avellaneda - Quijote apocrifo seconda novelle',
    'Cervantes - Don Quijote I',
    'Cervantes - Don Quijote II'
]

# ----------------------------------------------
# Data loader
# ----------------------------------------------
"""def load_dataset(
    test_documents: str, path: str = "src/data/corpus", keep_quixote=False
) -> Tuple[List[str], List[str], List[str]]:
    print("Loading data...")
    print(f"Looking for files in: {path}")

    corpus_path = Path(path)
    assert corpus_path.exists(), f"ERROR: Path {path} does not exist!"
    all_files = list(corpus_path.glob("*.txt"))
    print(f"All .txt files found: {[f.name for f in all_files]}")

    # print(f'Calling load_corpus with remove_test={False if test_document == "Avellaneda - Quijote apocrifo" else True}')

    documents, authors, filenames = load_corpus(
        path=path,
        # remove_test= False if test_document == ["Avellaneda - Quijote apocrifo"] else True,
        remove_unique_authors=False,
        remove_quixote=not keep_quixote,
        remove_avellaneda=True,
        test_documents=test_documents,
    )
    print(f"After load_corpus, filenames: {filenames}")
    print("Data loaded.")
    return documents, authors, filenames
    """


# ----------------------------------------------
# Model
# ----------------------------------------------
class AuthorshipVerification:
    """Main class for authorship verification system"""

    def __init__(self, config: "ModelConfig", nlp: spacy.Language):
        self.config = config
        self.nlp = nlp

    """def loo_multiple_test_split( self, test_index: int, test_indexes, X: List[str], y: List[int],
        doc: str, ylabel: int, filenames: List[str],) -> Tuple[List[str], List[str], List[int], List[int], List[str], List[str]]:

        doc_name = filenames[test_index]
        print(f"Test document: {doc_name[:-2]}")

        X_test = [doc]
        y_test = [int(ylabel)]
        X_dev = list(np.delete(X, test_indexes))
        y_dev = list(np.delete(y, test_indexes))
        groups_dev = list(np.delete(filenames, test_indexes))

        return X_dev, X_test, y_dev, y_test, groups_dev, [doc_name]

    def loo_split(self, i: int, X: List[str], y: List[int], doc: str,
        ylabel: int, filenames: List[str]) -> Tuple[List[str], List[str], List[int], List[int], List[str], List[str]]:

        doc_name = filenames[i]
        print(f"Test document: {doc_name[:-2]}")

        X_test = [doc]
        y_test = [int(ylabel)]
        X_dev = list(np.delete(X, i))
        y_dev = list(np.delete(y, i))
        groups_dev = list(np.delete(filenames, i))

        return X_dev, X_test, y_dev, y_test, groups_dev, [doc_name]"""

    """def find_segment(self, segment: str, processed_document: spacy.tokens.Doc
    ) -> spacy.tokens.Span:
        Find a segment within a processed document
        segment = sent_tokenize(segment)[0]
        start_segment = (
            segment  # segment.replace('\n',' ').replace('  ', ' ').replace('\t', ' ')
        )
        start_idx = processed_document.text.find(start_segment)
        end_idx = start_idx + len(segment)

        if start_idx == -1:
            print("mismatch found:::")
            print("SEGMENT:", start_segment)
            print("PROCESSED:", processed_document.text)

        processed_seg = processed_document.char_span(
            start_idx, end_idx, alignment_mode="expand"
        )
        if not processed_seg:
            processed_seg = processed_document.char_span(
                start_idx, end_idx - 1, alignment_mode="expand"
            )

        return processed_seg

    def get_processed_segments( self, processed_docs: Dict[str, spacy.tokens.Doc], X: List[str], groups: List[str],
        dataset: str = "") -> List[Union[spacy.tokens.Doc, spacy.tokens.Span]]:
        Extract processed segments from documents
        # print(f'Extracting processed {dataset} segments...')

        none_count = 0
        processed_X = []

        for segment, group in tqdm(zip(X, groups), total=len(X), desc="Progress"):
            if group.endswith("_0_0"):  # entire doc
                processed_doc = processed_docs[group[:-4]]
                processed_X.append(processed_doc)
            else:  # segment
                group_idx = group.find("_0")
                group_key = group[:group_idx]
                ent_doc_processed = processed_docs[group_key]
                processed_segment = self.find_segment(segment, ent_doc_processed)

                if not processed_segment:
                    none_count += 1
                processed_X.append(processed_segment)

        print(f"None count: {none_count}\n")
        return processed_X"""

    def extract_feature_vectors(
        self, processed_docs: List[spacy.tokens.Doc], y: List[str], filenames: List[str]) -> Tuple[np.ndarray, ...]:

        spanish_function_words = get_spanish_function_words()

        vectorizers = [
            FeaturesFunctionWords(
                function_words=spanish_function_words, ngram_range=(1, 1)
            ),
            FeaturesPOST(n=(1, 3)),
            FeaturesMendenhall(upto=27),
            FeaturesSentenceLength(),
            FeaturesDistortedView(method="DVEX", function_words=spanish_function_words),
            FeaturesPunctuation(),
            FeaturesDEP(n=(2, 3), use_words=True),
        ]

        hstacker = HstackFeatureSet(vectorizers)
        feature_sets = []
        feature_sets_orig = []
        orig_filenames = filenames.copy()

        for vectorizer in vectorizers:
            # print(f'\nExtracting {vectorizer}')
            reductor = FeatureSetReductor(vectorizer, k_ratio=self.config.k_ratio)

            # print('\nProcessing set')
            features_set= reductor.fit_transform(processed_docs, y)

            if self.config.oversample:
                feature_sets_orig.append(features_set)
                orig_y = y.copy()

                (
                    features_set,
                    y_oversampled,
                    features_set,
                    y_oversampled,
                    groups,
                ) = reductor.oversample_DRO(
                    Xtr=features_set,
                    ytr=y,
                    groups=orig_filenames,
                    rebalance_ratio=self.config.rebalance_ratio,
                )
                feature_sets.append(features_set)
            else:
                feature_sets.append(features_set)

        orig_feature_sets_idxs = self._compute_feature_set_idx(
            vectorizers, feature_sets_orig
        )
        feature_sets_idxs = self._compute_feature_set_idx(vectorizers, feature_sets)

        # print(f'Feature sets computed: {len(feature_sets_dev)}')
        # print('\nStacking feature vectors')

        if feature_sets_orig:
            X_stacked_orig = hstacker._hstack(feature_sets_orig)
            # print(f'X_dev_stacked_orig shape: {X_dev_stacked_orig.shape}')
            # print(f'X_test_stacked_orig shape: {X_test_stacked_orig.shape}')

        X_stacked = hstacker._hstack(feature_sets)

        # print(f'X_dev_stacked shape: {X_dev_stacked.shape}')
        # print(f'X_test_stacked shape: {X_test_stacked.shape}')

        y_final = y_oversampled if self.config.oversample else y

        #print("Feature vectors extracted.")
        #print(f"Vector document final shape: {X_dev_stacked.shape}")
        #print(f"X_dev_stacked: {X_dev_stacked.shape[0]}")
        #print(f"y_dev: {len(y_dev_final)}")
        #print(f"groups_dev: {len(groups_dev)}")
        #print(f"groups_dev_orig: {len(orig_groups_dev)}")

        if self.config.oversample:
            return (X_stacked, y_final, filenames, feature_sets_idxs,
                orig_feature_sets_idxs, X_stacked_orig, orig_y, orig_filenames)
        else:
            return (X_stacked, y, filenames, feature_sets_idxs, None, None, None, None)

    def _compute_feature_set_idx(self, vectorizers, feature_sets_dev):
        """Helper method to compute feature set indices"""
        start_idx = 0
        end_idx = 0
        feature_sets_idxs = {}

        for vect, fset in zip(vectorizers, feature_sets_dev):
            if isinstance(fset, list):
                fset = np.array(fset)

            if len(fset.shape) == 1:
                fset = fset.reshape(-1, 1)

            feature_shape = fset.shape[1]
            end_idx += feature_shape
            feature_sets_idxs[vect] = (start_idx, end_idx)
            start_idx = end_idx

        return feature_sets_idxs

    def train_model(self, X_dev: np.ndarray, y_dev: List[str], groups_dev: List[str],
        model: BaseEstimator, model_name: str) -> BaseEstimator:

        param_grid = {"C": np.logspace(-4, 4, 9), "class_weight": ["balanced", None]}

        cv = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=self.config.random_state
        )

        if self.config.positive_author is None:
            scoring_method = make_scorer(f1_score, average="macro", zero_division=1)
        else:
            scoring_method = make_scorer(f1_score, zero_division=1)

        grid_search = GridSearchCV(
            model,
            param_grid=param_grid,
            refit=False,
            cv=cv,
            n_jobs=self.config.n_jobs,
            scoring=scoring_method,
            verbose=True,
        )

        grid_search.fit(X_dev, y_dev, groups=groups_dev)
        print(f"Model fitted. Best params: {grid_search.best_params_}")
        print(f"Best scores: {grid_search.best_score_}\n")

        h = LogisticRegression(**grid_search.best_params_)

        h1 = CalibratedClassifierCV(h, n_jobs=self.config.n_jobs).fit(X_dev, y_dev)

        return h1

    def evaluate_model(self, clf: BaseEstimator,X_test: np.ndarray, y_test: List[str],return_proba: bool = True,
    ) -> Tuple[float, float, np.ndarray, float, Optional[str]]:

        # print('Evaluating performance...(on fragmented text)' if len(y_test) > 110 else '\n')

        y_test = np.array(y_test * X_test.shape[0])
        y_pred = clf.predict(X_test)
        y_pred_list = y_pred.tolist()

        predicted_author = None
        if self.config.multiclass and len(y_pred) > 0:
            single_pred = y_pred[0]
            predicted_author = self.id_to_author.get(
                single_pred, f"Unknown_{single_pred}"
            )
            print(f"Predicted author: {predicted_author}")

        if return_proba:
            probabilities = clf.predict_proba(X_test)
            self.posterior_proba = np.median(
                [prob[class_idx] for prob, class_idx in zip(probabilities, y_pred)]
            )
            proba_values = [
                prob[class_idx] for prob, class_idx in zip(probabilities, y_pred)
            ]
            print(f"Posterior probability: {self.posterior_proba}")
            print(f"Posterior probability vector: {probabilities}")

        self.accuracy = accuracy_score(y_test, y_pred)

        if self.config.multiclass:
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=1.0)
            precision, recall, _, _ = precision_recall_fscore_support(
                y_test, y_pred, average="macro", zero_division=1.0
            )
            # print(f'Precision (macro): {precision}')
            # print(f'Recall (macro): {recall}')
            # print(f'F1 (macro): {f1}')

            if self.config.multiclass and self.id_to_author:
                unique_test_classes = sorted(set(y_test))
                target_names = [
                    self.id_to_author[i]
                    for i in unique_test_classes
                    if i in self.id_to_author
                ]

                print(f"\nPer-class metrics:")
                print(f"Classes in test data: {unique_test_classes}")
                print(f"Corresponding target names: {target_names}")

                # print(classification_report(
                # y_test, y_pred,
                # labels=unique_test_classes,
                # target_names=target_names,
                # zero_division=1.0
            # ))
            # else:
            # print(classification_report(y_test, y_pred_list, zero_division=1.0))
        else:

            f1 = f1_score(y_test, y_pred, average="binary", zero_division=1.0)
            precision, recall, _, _ = precision_recall_fscore_support(
                y_test, y_pred, average="binary", zero_division=1.0
            )
            # print(f'Precision: {precision}')
            # print(f'Recall: {recall}')
        # print(f'F1: {f1}')
        # print(classification_report(y_test, y_pred, zero_division=1.0))

        # print(f'Accuracy: {self.accuracy}')

        cf = confusion_matrix(y_test, y_pred)
        if not self.config.multiclass:
            cf = cf.ravel()
            print(f"\nConfusion matrix: (tn, fp, fn, tp)\n{cf}\n")
        else:
            print(f"\nConfusion matrix:\n{cf}\n")

        # print(f"Random seed: {self.config.random_state}")

        return self.accuracy, f1, cf, self.posterior_proba, predicted_author

    def save_results(
        self,
        target_author: str,
        accuracy: float,
        f1: float,
        posterior_proba: float,
        model_name: str,
        doc_name: str,
        features: List[str],
        file_name: str,
        path_name: str,
        y_test: List[int],
        predicted_author: Optional[str] = None,
    ):

        path = Path(path_name)
        print(f"Saving results in {file_name}\n")

        if self.config.multiclass:
            unique_test_classes = sorted(set(y_test))
            target_names = [
                self.id_to_author[i]
                for i in unique_test_classes
                if i in self.id_to_author
            ]

            target_info = (
                str(target_names).replace("[", "").replace("]", "").replace("'", "")
            )

        else:
            target_info = target_author

        if self.config.multiclass:
            classification_type = "multiclass"
        else:
            classification_type = "binary"

        data = {
            "Classification Type": classification_type,
            "Target author": target_info,
            "Predicted author": predicted_author if predicted_author else "N/A",
            "Document test": doc_name[:-2],
            "Accuracy": accuracy,
            "Proba": posterior_proba,
            "Features": features,
        }

        output_path = path / file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(data)

        print(f"{model_name} results for author {target_author} saved in {file_name}\n")

    def fit(self, test_documents: List[Book]):

        y = [book.author for book in test_documents]
        filenames =[book.path.name for book in test_documents]
        processed_documents = [book.processed for book in test_documents]

        print(f"Label distribution: {np.unique(y, return_counts=True)}")

        """if test_documents:
            test_indices = []
            for test_document in test_documents:
                test_document_normalized = test_document.strip()
                for test_idx, filename in enumerate(filenames):
                    filename_normalized = filename.strip()
                    if test_document_normalized in filename_normalized:
                        test_indices.append(test_idx)
                # print(f'Testing on: {test_documents}')
                # print(f'Found test indices: {test_indices}')
                if not test_indices:
                    print(
                        f'ERROR: Test document "{documents}" not found in available filenames'
                    )
                    print(f"Available filenames: {filenames}")
                    return
        else:
            test_indices = list(range(len(documents)))

        print(f"Total documents to test: {len(test_indices)}")

        #TODO: estrarre i feature vectors, passare a train_model le features, le label, groups_dev,
        # il modello e il model name

        for test_idx in test_indices:
            print(f"\n=== Processing document {test_idx + 1}/{len(test_indices)} ===")"""

        (X_stacked, y, filenames, feature_sets_idxs, *_) = self.extract_feature_vectors(
            processed_documents, y, filenames)

        #todo: valori oversample

        model = LogisticRegression(
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )
        print(f"\nBuilding classifier...\n")
        clf = self.train_model(X_stacked, y, filenames, model, "LogisticRegression")

        return clf


    def run(self, target: str, test_documents: Union[str, List[str]]):
        """Run the complete authorship verification process"""
        start_time = time.time()
        print(f'Start time: {time.strftime("%H:%M")}')

        if isinstance(test_documents, str):
            test_documents = [test_documents]

        if self.config.multiclass:
            print(f"Building multiclass model for all authors.\n")
        else:
            print(f"Building binary LOO model for author {target}.\n")

        corpus = load_corpus(path=self.config.corpus_path, nlp=self.nlp)

        documents = [ book.clean_text for book in corpus]
        processed_documents = [ book.processed for book in corpus]
        segmented = [ book.segmented for book in corpus]
        authors = [book.author for book in corpus]
        filenames = [book.path.name for book in corpus]
        genres = [ "Trattato" if "epistola" not in filename.lower() else "Epistola"
                   for filename in filenames]

        print(f"Available filenames: {filenames}")

        print(f"Genres: {np.unique(genres, return_counts=True)}")

        y = binarize_corpus(corpus, target) if not self.config.multiclass else authors

        print(f"Label distribution: {np.unique(y, return_counts=True)}")

        if test_documents:
            test_indices = []
            for test_document in test_documents:
                test_document_normalized = test_document.strip()
                for test_idx, filename in enumerate(filenames):
                    filename_normalized = filename.strip()
                    if test_document_normalized in filename_normalized:
                        test_indices.append(test_idx)

                # print(f'Testing on: {test_documents}')
                # print(f'Found test indices: {test_indices}')
                if not test_indices:
                    print(
                        f'ERROR: Test document "{test_documents}" not found in available filenames'
                    )
                    print(f"Available filenames: {filenames}")
                    return
        else:
            test_indices = list(range(len(documents)))
            if self.config.multiclass:
                print(
                    f"Full LOO evaluation: testing on all {len(test_indices)} documents (multiclass)"
                )
            else:
                print(
                    f"Full LOO evaluation: testing on all {len(test_indices)} documents (binary classification for {target})"
                )

        print(f"Total documents to test: {len(test_indices)}")

        for test_idx in test_indices:
            print(f"\n=== Processing document {test_idx + 1}/{len(test_indices)} ===")

            self.train_and_test_single_document(
                test_idx,
                test_indices,
                documents,
                y,
                processed_documents,
                filenames,
                target,
                self.config.save_results,
                self.config.results_filename,
                self.config.results_path,
                experiment_type=self.config.experiment,
            )

        total_time = round((time.time() - start_time) / 60, 2)
        if self.config.multiclass:
            print(
                f"Total time spent for multiclass model building: {total_time} minutes."
            )
        else:
            print(
                f"Total time spent for model building for author {target}: {total_time} minutes."
            )

    def train_and_test_single_document(self, test_idx: int, test_indexes: List[int], documents: List[str],
        y: List[int], processed_documents: Dict[str, spacy.tokens.Doc], filenames: List[str], target: str,
        save_results: bool, file_name: str, path_name: str, experiment_type: str):
        """Process a single document for authorship verification"""
        start_time_single_iteration = time.time()
        np.random.seed(self.config.random_state)

        doc, ylabel = documents[test_idx], y[test_idx]

       #SPLIT PER LA LOO
        if experiment_type == "single-test":
            X_dev, X_test, y_dev, y_test, groups_dev, groups_test = self.loo_split(
                test_idx, documents, y, doc, ylabel, filenames
            )
        elif experiment_type == "multiple-test":
            X_dev, X_test, y_dev, y_test, groups_dev, groups_test = (
                self.loo_multiple_test_split(
                    test_idx, test_indexes, documents, y, doc, ylabel, filenames
                )
            )
        else:
            raise NotImplementedError("not yet implemented")


        (
            X_dev,
            X_test,
            y_dev,
            y_test,
            X_test_frag,
            y_test_frag,
            groups_dev,
            groups_test,
            groups_test_frag,
        ) = self.segment_data(X_dev, X_test, y_dev, y_test, groups_dev, groups_test)
        print(np.unique(y, return_counts=True))

        X_dev_processed = self.get_processed_segments(
            processed_documents, X_dev, groups_dev, dataset="training"
        )
        X_test_processed = self.get_processed_segments(
            processed_documents, X_test, groups_test, dataset="test"
        )

        X_len = len(X_dev_processed)
        print(f"X_len: {X_len}")

        (X_dev,
            X_test,
            y_dev,
            y_test,
            groups_dev,
            feature_idxs,
            original_feature_idxs,
            original_X_dev,
            original_X_test,
            orig_y_dev,
            orig_groups_dev,
        ) = self.extract_feature_vectors(
            X_dev_processed, X_test_processed, y_dev, y_test, groups_dev
        )

        # da qui in poi dovrebbe andare

        model = LogisticRegression(
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )
        print(f"\nBuilding classifier...\n")
        clf = self.train_model(X_dev, y_dev, groups_dev, model, "LogisticRegression")

        acc, f1, cf, posterior_proba, predicted_author = self.evaluate_model(
            clf, X_test, y_test
        )

        doc_name = groups_test[0][:-2] if groups_test else f"doc_{test_idx}"

        if save_results:
            self.save_results(
                target,
                acc,
                f1,
                posterior_proba,
                model,
                groups_test[0][:-2],
                feature_idxs.keys(),
                file_name,
                path_name,
                y_test,
                predicted_author,
            )

        iteration_time = round((time.time() - start_time_single_iteration) / 60, 2)
        print(
            f"Time spent for model building for document {groups_test[0][:-2]}: {iteration_time} minutes."
        )

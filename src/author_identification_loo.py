import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Union
import numpy as np
import spacy
from sklearn.base import BaseEstimator
from sklearn.model_selection import LeaveOneOut
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    make_scorer
)
import csv
import time
from pathlib import Path
from tqdm import tqdm
import nltk
nltk.download('punkt_tab')
from nltk import sent_tokenize
from data_preparation.data_loader import load_corpus, get_spanish_function_words
from data_preparation.segmentation import Segmentation
from feature_extraction.features import (
    DocumentProcessor,
    FeaturesFunctionWords,
    FeaturesDistortedView,
    FeaturesMendenhall,
    FeaturesSentenceLength,
    FeaturesPOST,
    FeatureSetReductor,
    FeaturesDEP,
    FeaturesPunctuation,
    HstackFeatureSet,
    FeaturesCharNGram,
)
from model_serialization.serialization import Serialization
from model_serialization.serialization import Labels
import warnings
warnings.filterwarnings("ignore")


@dataclass
class LOOConfig:
    """Configuration for LOO evaluation"""
    n_jobs: int = 30
    segment_min_token_size: int = 500
    random_state: int = 0
    k_ratio: float = 1.0
    oversample: bool = False
    rebalance_ratio: float = 0.2
    save_res: bool = True
    test_genre: bool = False
    multiclass: bool = False
    results_filename: str = 'results_LOO.csv'
    results_path: str = './results'
    authors_dir: str = './authorslabel'

    @classmethod
    def from_args(cls):
        """Create config from command line arguments"""
        parser = argparse.ArgumentParser(description='LOO Authorship Verification')
        parser.add_argument('--target', default='Cervantes',
                            help='Target author for binary classification')
        parser.add_argument('--multiclass', action='store_true',
                            help='Use multiclass classification')
        parser.add_argument('--save-res', action='store_true',
                            help='Save results to file')
        parser.add_argument('--results-filename', default='results_LOO.csv',
                            help='Output filename for results')
        parser.add_argument('--results-path', default='./results',
                            help='Output directory for results')

        args = parser.parse_args()

        config = cls()
        config.multiclass = args.multiclass
        config.results_filename = args.results_filename
        config.results_path = args.results_path
        config.save_res = args.save_res

        return config, args.target


class LOOAuthorshipSystem:
    """Leave-One-Out AV System"""

    def __init__(self, config: LOOConfig, nlp: spacy.Language):
        self.config = config
        self.nlp = nlp
        self.accuracy = 0
        self.posterior_proba = 0

    def load_dataset(self, path: str = 'src/data/corpus') -> Tuple[List[str], List[str], List[str]]:
        print(f'Loading dataset from: {path}')

        corpus_path = Path(path)
        if not corpus_path.exists():
            raise FileNotFoundError(f'Corpus path {path} does not exist')

        documents, authors, filenames = load_corpus(
            path=path,
            remove_unique_authors=True,
            remove_quixote=True,
            remove_avellaneda=True
        )

        print(f'After load_corpus, filenames: {filenames}')
        print('Data loaded.')

        return documents, authors, filenames

    def create_labels(self, authors: List[str], target: str) -> Tuple[List[int], Dict[str, int], Dict[int, str]]:
        """Create labels for classification"""
        if self.config.multiclass:
            unique_authors = sorted(list(set(author.strip() for author in authors)))
            author_to_id = {author: idx for idx, author in enumerate(unique_authors)}
            id_to_author = {idx: author for author, idx in author_to_id.items()}
            labels = [author_to_id[author.strip()] for author in authors]

            print(f'Multiclass labels created for {len(unique_authors)} authors')

            return labels, author_to_id, id_to_author
        else:
            labels = [1 if author.strip() == target else 0 for author in authors]
            author_to_id = {target: 1, 'Others': 0}
            id_to_author = {1: target, 0: 'Others'}

            print(f'Binary labels created for target author: {target}')

            return labels, author_to_id, id_to_author

    def loo_multiple_test_split(self, test_index: int, test_indexes, X: List[str], y: List[int], doc: str, ylabel: int,
                                filenames: List[str]) -> Tuple[
        List[str], List[str], List[int], List[int], List[str], List[str]]:

        doc_name = filenames[test_index]
        print(f'Test document: {doc_name[:-2]}')

        X_test = [doc]
        y_test = [int(ylabel)]
        X_dev = list(np.delete(X, test_indexes))
        y_dev = list(np.delete(y, test_indexes))
        groups_dev = list(np.delete(filenames, test_indexes))

        return X_dev, X_test, y_dev, y_test, groups_dev, [doc_name]

    def loo_split(self, i: int, X: List[str], y: List[int], doc: str, ylabel: int,
                  filenames: List[str]) -> Tuple[List[str], List[str], List[int], List[int], List[str], List[str]]:

        doc_name = filenames[i]
        print(f'Test document: {doc_name[:-2]}')

        X_test = [doc]
        y_test = [int(ylabel)]
        X_dev = list(np.delete(X, i))
        y_dev = list(np.delete(y, i))
        groups_dev = list(np.delete(filenames, i))

        return X_dev, X_test, y_dev, y_test, groups_dev, [doc_name]

    def segment_data(self, X_dev: List[str], X_test: List[str], y_dev: List[int],
                    y_test: List[int], groups_dev: List[str], groups_test: List[str]
                    ) -> Tuple:
        """Segment documents into smaller chunks"""

        print('Segmenting data...')
        whole_docs_len = len(y_test)

        segmentator_dev = Segmentation(
            split_policy='by_sentence',
            tokens_per_fragment=self.config.segment_min_token_size
        )
        splitted_docs_dev = segmentator_dev.fit_transform(
            documents=X_dev,
            authors=y_dev,
            filenames=groups_dev
        )

        segmentator_test = Segmentation(
            split_policy='by_sentence',
            tokens_per_fragment=self.config.segment_min_token_size
        )
        splitted_docs_test = segmentator_test.transform(
            documents=X_test,
            authors=y_test,
            filenames=groups_test
        )
        groups_test = segmentator_test.groups

        X_dev = splitted_docs_dev[0]
        y_dev = splitted_docs_dev[1]
        groups_dev = segmentator_dev.groups

        X_test = splitted_docs_test[0][:whole_docs_len]
        y_test = splitted_docs_test[1][:whole_docs_len]
        groups_test_entire_docs = groups_test[:whole_docs_len]

        X_test_frag = splitted_docs_test[0][whole_docs_len:]
        y_test_frag = splitted_docs_test[1][whole_docs_len:]
        groups_test_frag = groups_test[whole_docs_len:]

        print('Segmentation complete.')

        return (X_dev, X_test, y_dev, y_test, X_test_frag, y_test_frag,
                groups_dev, groups_test_entire_docs, groups_test_frag)


    def get_processed_documents(self, documents: List[str], filenames: List[str],
                                processed: bool = False,
                                cache_file: str = '.cache/processed_docs.pkl') -> Dict[str, spacy.tokens.Doc]:
        """Process documents using spaCy"""
        print('Processing documents...')

        if not processed:
            self.nlp.max_length = max(len(doc) for doc in documents)
            processor = DocumentProcessor(language_model=self.nlp, savecache=cache_file)
            processed_docs = processor.process_documents(documents, filenames)
        else:
            processor = DocumentProcessor(savecache=cache_file)
            processed_docs = processor.process_documents(documents, filenames)

        return processed_docs

    def find_segment(self, segment: str, processed_document: spacy.tokens.Doc) -> spacy.tokens.Span:
        """Find a segment within a processed document"""
        segment = sent_tokenize(segment)[0]
        start_segment = segment  # segment.replace('\n',' ').replace('  ', ' ').replace('\t', ' ')
        start_idx = processed_document.text.find(start_segment)
        end_idx = start_idx + len(segment)

        if start_idx == -1:
            print('mismatch found:::')
            print('SEGMENT:', start_segment)
            print('PROCESSED:', processed_document.text)

        processed_seg = processed_document.char_span(start_idx, end_idx, alignment_mode='expand')
        if not processed_seg:
            processed_seg = processed_document.char_span(start_idx, end_idx - 1, alignment_mode='expand')

        return processed_seg

    def get_processed_segments(self, processed_docs: Dict[str, spacy.tokens.Doc],
                               X: List[str], groups: List[str], dataset: str = ''
                               ) -> List[Union[spacy.tokens.Doc, spacy.tokens.Span]]:
        """Extract processed segments from documents"""
        print(f'Extracting processed {dataset} segments...')

        none_count = 0
        processed_X = []

        for segment, group in tqdm(zip(X, groups), total=len(X), desc='Progress'):
            if group.endswith('_0_0'):  # entire doc
                processed_doc = processed_docs[group[:-4]]
                processed_X.append(processed_doc)
            else:  # segment
                group_idx = group.find('_0')
                group_key = group[:group_idx]
                ent_doc_processed = processed_docs[group_key]
                processed_segment = self.find_segment(segment, ent_doc_processed)

                if not processed_segment:
                    none_count += 1
                processed_X.append(processed_segment)

        print(f'None count: {none_count}\n')
        return processed_X

    def extract_feature_vectors(self, processed_docs_dev: List[spacy.tokens.Doc],
                                processed_docs_test: List[spacy.tokens.Doc],
                                y_dev: List[int], y_test: List[int],
                                groups_dev: List[str]) -> Tuple[np.ndarray, ...]:

        print('Extracting feature vectors...')

        spanish_function_words = get_spanish_function_words()
        vectorizers = [
            FeaturesFunctionWords(
                function_words=spanish_function_words,
                ngram_range=(1, 1)
            ),
            FeaturesPOST(n=(1, 3)),
            FeaturesMendenhall(upto=27),
            FeaturesSentenceLength(),
            FeaturesDistortedView(method='DVEX', function_words=spanish_function_words),
            FeaturesPunctuation(),
            FeaturesDEP(n=(2, 3), use_words=True)
        ]

        hstacker = HstackFeatureSet(vectorizers)
        feature_sets_dev = []
        feature_sets_test = []
        feature_sets_dev_orig = []
        feature_sets_test_orig = []
        orig_groups_dev = groups_dev.copy()

        for vectorizer in vectorizers:
            print(f'\nExtracting {vectorizer}')
            reductor = FeatureSetReductor(
                vectorizer,
                k_ratio=self.config.k_ratio
            )

            print('\nProcessing development set')
            features_set_dev = reductor.fit_transform(processed_docs_dev, y_dev)

            print('\nProcessing test set')
            features_set_test = reductor.transform(processed_docs_test)

            if self.config.oversample:
                feature_sets_dev_orig.append(features_set_dev)
                feature_sets_test_orig.append(features_set_test)
                orig_y_dev = y_dev.copy()

                (features_set_dev, y_dev_oversampled, features_set_test,
                 y_test_oversampled, groups_dev) = reductor.oversample_DRO(
                    Xtr=features_set_dev,
                    ytr=y_dev,
                    Xte=features_set_test,
                    yte=y_test,
                    groups=orig_groups_dev,
                    rebalance_ratio=self.config.rebalance_ratio
                )
                feature_sets_dev.append(features_set_dev)
                feature_sets_test.append(features_set_test)
            else:
                feature_sets_dev.append(features_set_dev)
                feature_sets_test.append(features_set_test)

        orig_feature_sets_idxs = self._compute_feature_set_idx(
            vectorizers,
            feature_sets_dev_orig
        )
        feature_sets_idxs = self._compute_feature_set_idx(
            vectorizers,
            feature_sets_dev
        )

        print(f'Feature sets computed: {len(feature_sets_dev)}')
        print('\nStacking feature vectors')

        if feature_sets_dev_orig:
            X_dev_stacked_orig = hstacker._hstack(feature_sets_dev_orig)
            X_test_stacked_orig = hstacker._hstack(feature_sets_test_orig)
            print(f'X_dev_stacked_orig shape: {X_dev_stacked_orig.shape}')
            print(f'X_test_stacked_orig shape: {X_test_stacked_orig.shape}')

        X_dev_stacked = hstacker._hstack(feature_sets_dev)
        X_test_stacked = hstacker._hstack(feature_sets_test)

        print(f'X_dev_stacked shape: {X_dev_stacked.shape}')
        print(f'X_test_stacked shape: {X_test_stacked.shape}')

        y_dev_final = y_dev_oversampled if self.config.oversample else y_dev
        y_test_final = y_test_oversampled if self.config.oversample else y_test

        print('\nFeature vectors extracted.\n')
        print(f'Vector document final shape: {X_dev_stacked.shape}')
        print(f"\nX_dev_stacked: {X_dev_stacked.shape[0]}")
        print(f"y_dev: {len(y_dev_final)}")
        print(f"groups_dev: {len(groups_dev)}")
        print(f"groups_dev_orig: {len(orig_groups_dev)}")

        if self.config.oversample:
            return (X_dev_stacked, X_test_stacked, y_dev_final, y_test_final,
                    groups_dev, feature_sets_idxs, orig_feature_sets_idxs,
                    X_dev_stacked_orig, X_test_stacked_orig, orig_y_dev,
                    orig_groups_dev)
        else:
            return (X_dev_stacked, X_test_stacked, y_dev_final, y_test_final,
                    groups_dev, feature_sets_idxs, None, None, None, None, None)

    def train_model(self, X_dev: np.ndarray, y_dev: List[int],
                    groups_dev: List[str], model: BaseEstimator,
                    model_name: str) -> BaseEstimator:

        param_grid = {
            'C': np.logspace(-4, 4, 9),
            'class_weight': ['balanced', None]
        }

        cv = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=self.config.random_state
        )

        if self.config.multiclass:
            scoring_method = make_scorer(f1_score, average='macro', zero_division=1)
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
        print(f'Model fitted. Best params: {grid_search.best_params_}')
        print(f'Best scores: {grid_search.best_score_}\n')

        h = LogisticRegression(**grid_search.best_params_)

        h1 = CalibratedClassifierCV(h, n_jobs=self.config.n_jobs).fit(X_dev, y_dev)

        return h1

    def evaluate_model(self, clf: BaseEstimator, X_test: np.ndarray,
                       y_test: List[int], return_proba: bool = True
                       ) -> Tuple[float, float, np.ndarray, float, Optional[str]]:

        print('Evaluating performance...',
              '(on fragmented text)' if len(y_test) > 110 else '\n')

        y_test = np.array(y_test * X_test.shape[0])
        y_pred = clf.predict(X_test)
        y_pred_list = y_pred.tolist()

        predicted_author = None
        if self.config.multiclass and len(y_pred) > 0:
            single_pred = y_pred[0]
            predicted_author = self.id_to_author.get(single_pred, f"Unknown_{single_pred}")
            print(f'Predicted author: {predicted_author}')

        if return_proba:
            probabilities = clf.predict_proba(X_test)
            self.posterior_proba = np.median(
                [prob[class_idx] for prob, class_idx in zip(probabilities, y_pred)]
            )
            proba_values = [prob[class_idx] for prob, class_idx in zip(probabilities, y_pred)]
            print(f'Posterior probability: {self.posterior_proba}')
            print(f'Posterior probability vector: {probabilities}')

        self.accuracy = accuracy_score(y_test, y_pred)

        if self.config.multiclass:
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=1.0)
            precision, recall, _, _ = precision_recall_fscore_support(
                y_test, y_pred, average='macro', zero_division=1.0
            )
            # print(f'Precision (macro): {precision}')
            # print(f'Recall (macro): {recall}')
            # print(f'F1 (macro): {f1}')

            if self.config.multiclass and self.id_to_author:
                unique_test_classes = sorted(set(y_test))
                target_names = [self.id_to_author[i] for i in unique_test_classes
                                if i in self.id_to_author]

                print(f'\nPer-class metrics:')
                print(f'Classes in test data: {unique_test_classes}')
                print(f'Corresponding target names: {target_names}')

                # print(classification_report(
                # y_test, y_pred,
                # labels=unique_test_classes,
                # target_names=target_names,
                # zero_division=1.0
            # ))
            # else:
            # print(classification_report(y_test, y_pred_list, zero_division=1.0))
        else:

            f1 = f1_score(y_test, y_pred, average='binary', zero_division=1.0)
            precision, recall, _, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary', zero_division=1.0
            )
            # print(f'Precision: {precision}')
            # print(f'Recall: {recall}')
        # print(f'F1: {f1}')
        # print(classification_report(y_test, y_pred, zero_division=1.0))

        # print(f'Accuracy: {self.accuracy}')

        cf = confusion_matrix(y_test, y_pred)
        if not self.config.multiclass:
            cf = cf.ravel()
            print(f'\nConfusion matrix: (tn, fp, fn, tp)\n{cf}\n')
        else:
           print(f'\nConfusion matrix:\n{cf}\n')

        print(f"Random seed: {self.config.random_state}")

        return self.accuracy, f1, cf, self.posterior_proba, predicted_author

    def save_results(self, target_author: str, accuracy: float, f1: float,
                     posterior_proba: float, model_name: str,
                     doc_name: str, features: List[str],
                     file_name: str, path_name: str, y_test: List[int], predicted_author: Optional[str] = None):

        path = Path(path_name)
        print(f'Saving results in {file_name}\n')

        if self.config.multiclass:
            unique_test_classes = sorted(set(y_test))
            target_names = [self.id_to_author[i] for i in unique_test_classes
                                if i in self.id_to_author]

            target_info = str(target_names).replace('[', '').replace(']', '').replace("'", '')

        else:
            target_info = target_author

        if self.config.multiclass:
            classification_type = 'multiclass'
        else:
            classification_type = 'binary'

        data = {
            'Classification Type': classification_type,
            'Target author': target_info,
            'Predicted author': predicted_author if predicted_author else 'N/A',
            'Document test': doc_name[:-2],
            'Proba': posterior_proba,
            'Features': features
        }

        output_path = path / file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(data)

        print(f"{model_name} results for author {target_author} saved in {file_name}\n")

    def run_loo(self, target_author: str, save_results: bool = True, corpus_path: str = 'src/data/corpus'):
        """Run the complete Leave-One-Out evaluation"""
        print(f'Starting LOO evaluation for {"multiclass" if self.config.multiclass else "binary"} classification')
        start_time = time.time()

        documents, authors, filenames = self.load_dataset(corpus_path)
        filenames = [f'{filename}_0' for filename in filenames]

        labels, author_to_id, id_to_author = self.create_labels(authors, target_author)

        processed_docs = self.process_documents(documents, filenames)

        print(f'Label distribution: {np.unique(labels, return_counts=True)}')

        all_results = []
        test_indices = list(range(len(documents)))

        for test_idx in test_indices:
            print(f'\n=== Processing document {test_idx + 1}/{len(test_indices)} ===')

            train_docs, test_docs, train_labels, test_labels, train_filenames, test_filenames = self.loo_split(
                test_indices, documents, labels, filenames
            )

            train_X, test_X, train_y, test_y, train_groups, test_groups = self.segment_documents(
                train_docs, test_docs, train_labels, test_labels, train_filenames, test_filenames
            )

            train_processed = self.get_processed_segments(processed_docs, train_X, train_groups)
            test_processed = self.get_processed_segments(processed_docs, test_X, test_groups)

            train_features, test_features = self.extract_features(train_processed, test_processed, train_y)

            model = self.train_classifier(train_features, train_y, train_groups)

            metrics = self.evaluate_model(model, test_features, test_y, id_to_author)

            if save_results and all_results:
                self.save_loo_results(all_results, target_author)

            # Print summary
            total_time = round((time.time() - start_time) / 60, 2)
            print(f'\n=== LOO Evaluation Complete ===')
            print(f'Total time: {total_time} minutes')

        return all_result

    def main(self, ):
        config, target_author = LOOConfig.from_args()
        nlp = spacy.load('es_dep_news_trf')

        loo_system = LOOAuthorshipSystem(config, nlp)
        results = loo_system.run_loo(
            target_author=target_author,
            save_results=config.save_res,
            corpus_path='src/data/corpus'

    if __name__ == '__main__':
        main()


import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
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
    FeaturesVerbalEndings,
    # FeaturesSyllabicQuantities
)
from model_serialization.serialization import Serialization
from model_serialization.serialization import Labels
import warnings
warnings.filterwarnings("ignore")

AVELLANEDA_DOCUMENTS = [
    'Avellaneda - Quijote apocrifo',
    'Avellaneda - Quijote apocrifo nucleo',
    'Avellaneda - Quijote apocrifo prologo',
    'Avellaneda - Quijote apocrifo prima novelle',
    'Avellaneda - Quijote apocrifo seconda novelle'
]

@dataclass
class ModelConfig:
    """Configuration for the model training and evaluation"""
    n_jobs: int = 30
    segment_min_token_size: int = 500
    random_state: int = 0
    k_ratio: float = 1.0
    oversample: bool = False
    rebalance_ratio: float = 0.2
    save_res: bool = True
    test_genre: bool = False
    multiclass: bool = False
    results_filename: str = 'results.csv'
    results_path: str = './results'
    authors_dir: str = './authorslabel'



    @classmethod
    def from_args(cls):
        """Create config from command line args"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--test-document', default=AVELLANEDA_DOCUMENTS,
                            help='Test document (empty=full LOO, author=author LOO, doc=specific doc, '
                             'list of docs to test more than one)')
        parser.add_argument('--target', default='Cervantes',
                        help='Target author')
        parser.add_argument('--multiclass', action='store_true',
                            help='Use multiclass classification instead of binary')
        parser.add_argument('--save-res', action='store_true',
                        help='Save results to file')
        parser.add_argument('--results-filename', default='results.csv',
                    help='Filename for saving results')
        parser.add_argument('--results-path', 
                    default='./results',
                    help='Directory path for saving results')
        parser.add_argument('--keep_quixote', action='store_true',
                            help='Keeps both parts of Quixote')
        parser.add_argument('--save-model', action='store_true',
                            help='Save trained models to disk')
        parser.add_argument('--load-model', type=str, default=None,
                            help='Path to load a pre-trained model from')
        parser.add_argument('--models-dir', type=str, default='./saved_models',
                            help='Directory to save/load models')
        parser.add_argument('--model-name', type=str, default=None,
                            help='Custom name for saved model (auto-generated if not provided)')
        parser.add_argument('--authors-dir', type=str, default='./authorslabel',
                            help='Directory for author label mappings')

        args = parser.parse_args()

        if '--target' in sys.argv and '--test-document' not in sys.argv:
            args.test_document = ''
            
        config = cls()
        config.results_filename = args.results_filename
        config.results_path = args.results_path
        config.save_res = args.save_res
        config.multiclass = args.multiclass
        config.authors_dir = args.authors_dir
        config.save_model = args.save_model
        config.load_model = args.load_model
        config.models_dir = args.models_dir
        config.model_name = args.model_name
        config.args = args

        if args.test_document == "":
            config.experiment = 'loo'
        elif isinstance(args.test_document, str):
            config.experiment = 'single-test'
        elif isinstance(args.test_document, list):
            config.experiment = 'multiple-test'
        else:
            raise ValueError(f'args.test_document not undestood')

        
        return config, args.target, args.test_document
            

class AuthorshipVerification:
    """Main class for authorship verification system"""
    
    def __init__(self, config: ModelConfig, nlp: spacy.Language):
        self.config = config
        self.nlp = nlp
        self.accuracy = 0
        self.posterior_proba = 0

        self.labels_manager = Labels(authors_dir=config.authors_dir)
        self.author_to_id = {}
        self.id_to_author = {}

        self.model_serializer = Serialization(models_dir=config.models_dir)
        
    def load_dataset(self, test_documents: str, path: str = 'src/data/corpus') -> Tuple[List[str], List[str], List[str]]:
        
        print('Loading data...')
        print(f'Looking for files in: {path}')

        corpus_path = Path(path)
        assert corpus_path.exists(), f'ERROR: Path {path} does not exist!'
        all_files = list(corpus_path.glob('*.txt'))
        print(f'All .txt files found: {[f.name for f in all_files]}')

        # print(f'Calling load_corpus with remove_test={False if test_document == "Avellaneda - Quijote apocrifo" else True}')

        documents, authors, filenames = load_corpus(
            path=path, 
            # remove_test= False if test_document == ["Avellaneda - Quijote apocrifo"] else True,
            remove_unique_authors=False,
            remove_quixote = not self.config.args.keep_quixote,
            remove_avellaneda = True,
            test_documents=test_documents
        )
        print(f'After load_corpus, filenames: {filenames}')
        print('Data loaded.')
        return documents, authors, filenames

    def create_labels(self, authors: List[str], target: str, test_genre: bool = False,
                      genres: List[str] = None) -> List[int]:

        if test_genre:
            return [1 if genre.rstrip() == 'Trattato' else 0 for genre in genres]

        if self.config.multiclass:

            author_to_id_path = Path(self.config.authors_dir) / "author_to_id.json"
            #id_to_author_path = Path(self.config.authors_dir) / "id_to_author.json"

            mappings_loaded = False

            if author_to_id_path.exists():
                try:
                    print("Loading existing author mappings...")
                    self.author_to_id = self.labels_manager.load_author_to_id(str(author_to_id_path))
                    #self.id_to_author = self.labels_manager.load_id_to_author(str(id_to_author_path))

                    mappings_loaded = True
                    print(f"Loaded existing mappings for {len(self.author_to_id)} authors")
                    print(f"Author mapping: {self.author_to_id}")

                except Exception as e:
                    print(f"Error loading existing mappings: {e}")
                    print("Creating new mappings...")
                    mappings_loaded = False

            if not mappings_loaded:
                print("Creating new author mappings...")
                unique_authors = sorted(list(set(author.rstrip() for author in authors)))

                self.labels_manager.save_author_to_id(unique_authors)
                self.author_to_id = self.labels_manager.author_to_id

                #self.labels_manager.save_id_to_author(self.author_to_id)
                #self.id_to_author = self.labels_manager.id_to_author
                #self.id_to_author = {int(k): v for k, v in self.labels_manager.id_to_author.items()}

                print(f"Created new mappings for {len(unique_authors)} authors")
                print(f"Author mapping: {self.author_to_id}")

            self.id_to_author = {i: author for author, i in self.author_to_id.items()}
            return [self.author_to_id[author.rstrip()] for author in authors]

        else:
            return [1 if author.rstrip() == target else 0 for author in authors]

    def loo_multiple_test_split(self, test_index: int, test_indexes, X: List[str], y: List[int], doc: str, ylabel: int,
                                filenames: List[str]) -> Tuple[List[str], List[str], List[int], List[int], List[str], List[str]]:

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
                    ) -> Tuple[List[str], List[str], List[int], List[int], List[str], List[str], List[str]]:
        """Segment the documents into smaller chunks"""
        
        print('Segmenting data...')
        whole_docs_len = len(y_test)

        #print(f"DEBUG: whole_docs_len = {whole_docs_len}")
        #print(f"DEBUG: Original X_test length = {len(X_test)}")
        #print(f"DEBUG: Original X_test[0] length = {len(X_test[0])} characters")
        X_test_original = X_test[0]

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

       # print(f"DEBUG: After segmentation, splitted_docs_test[0] has {len(splitted_docs_test[0])} fragments")
        #print(f"DEBUG: Fragment lengths: {[len(frag) for frag in splitted_docs_test[0]]}")

        X_dev = splitted_docs_dev[0]
        y_dev = splitted_docs_dev[1]
        groups_dev = segmentator_dev.groups

        X_test = splitted_docs_test[0][:whole_docs_len]
        y_test = splitted_docs_test[1][:whole_docs_len]
        groups_test_entire_docs = groups_test[:whole_docs_len]

       # print(f"DEBUG: X_test after slicing has {len(X_test)} items")
       # print(f"DEBUG: X_test[0] length = {len(X_test[0])} characters")

        X_test_frag = splitted_docs_test[0][whole_docs_len:]
        y_test_frag = splitted_docs_test[1][whole_docs_len:]
        groups_test_frag = groups_test[whole_docs_len:]

        #print(f"DEBUG: X_test_frag has {len(X_test_frag)} fragments")
       # print(f"DEBUG: Are X_test[0] and original X_test[0] the same? {X_test[0] == X_test_original}")

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
        start_segment = segment # segment.replace('\n',' ').replace('  ', ' ').replace('\t', ' ')
        start_idx = processed_document.text.find(start_segment)
        end_idx = start_idx + len(segment)

        if start_idx == -1:
            print('mismatch found:::')
            print('SEGMENT:', start_segment)
            print('PROCESSED:', processed_document.text)
        
        processed_seg = processed_document.char_span(start_idx, end_idx, alignment_mode='expand')
        if not processed_seg:
            processed_seg = processed_document.char_span(start_idx, end_idx-1, alignment_mode='expand')
        
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
                ngram_range=(1,1)
            ),
            FeaturesPOST(n=(1,3)),
            FeaturesMendenhall(upto=27),
            FeaturesSentenceLength(),
            FeaturesDistortedView(method = 'DVEX', function_words= spanish_function_words),
            FeaturesPunctuation(),
            FeaturesDEP(n=(2,3), use_words= True)
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
                refit= False,
                cv=cv,
                n_jobs=self.config.n_jobs,
                scoring= scoring_method,
                verbose= True,
            )

            
            grid_search.fit(X_dev, y_dev, groups=groups_dev)
            print(f'Model fitted. Best params: {grid_search.best_params_}')
            print(f'Best scores: {grid_search.best_score_}\n')

            h = LogisticRegression(**grid_search.best_params_)

            h1= CalibratedClassifierCV(h, n_jobs= self.config.n_jobs).fit(X_dev, y_dev)

            
            return h1

    def save_trained_model(self, model: BaseEstimator, target_author: str,
                          test_document: str) -> str:
        """Save the trained model and complete system"""

        if not self.config.save_model:
            return

        if self.config.model_name:
            model_name = self.config.model_name
        else:
            classification_type = "multiclass" if self.config.multiclass else "binary"
            model_name = f"{classification_type}"

        try:
            model_path = self.model_serializer.save_model(model, f"model_{model_name}")
            print(f"Model saved to: {model_path}")

        except Exception as e:
            print(f"Error saving model: {e}")

        filepath = self.model_serializer.save_model( model=model )
        return filepath

    def load_pretrained_model(self, filepath: str) -> Optional[BaseEstimator]:
        """Load a pre-trained model"""
        try:
            model = self.model_serializer.load_model(filepath)
            print(f"Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            return None

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


            print (f'lenght of posterior probability: {len(proba_values)}')
        
        self.accuracy = accuracy_score(y_test, y_pred)

        if self.config.multiclass:
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=1.0)
            precision, recall, _, _ = precision_recall_fscore_support(
                y_test, y_pred, average='macro', zero_division=1.0
            )
            print(f'Precision (macro): {precision}')
            print(f'Recall (macro): {recall}')
            print(f'F1 (macro): {f1}')

            if self.config.multiclass and self.id_to_author:
                unique_test_classes = sorted(set(y_test))
                target_names = [self.id_to_author[i] for i in unique_test_classes
                                if i in self.id_to_author]

                print(f'\nPer-class metrics:')
                print(f'Classes in test data: {unique_test_classes}')
                print(f'Corresponding target names: {target_names}')

                print(classification_report(
                    y_test, y_pred,
                    labels=unique_test_classes,
                    target_names=target_names,
                    zero_division=1.0
                ))
            else:
                print(classification_report(y_test, y_pred_list, zero_division=1.0))
        else:

            f1 = f1_score(y_test, y_pred, average='binary', zero_division=1.0)
            precision, recall, _, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary', zero_division=1.0
            )
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            print(f'F1: {f1}')
            print(classification_report(y_test, y_pred, zero_division=1.0))

        print(f'Accuracy: {self.accuracy}')

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
            if self.config.load_model and len(y_test):
                unique_test_classes = sorted(set(y_test))
                target_names = [self.id_to_author.get[i] for i in unique_test_classes
                                if i in self.id_to_author.get]
            else:
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
                'Accuracy': accuracy,
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

    def run(self, target: str, test_documents: str, multiclass: bool = True, save_results: bool = True,
            test_genre: bool = False, corpus_path='../corpus'):
        """Run the complete authorship verification process"""
        start_time = time.time()
        print(f'Start time: {time.strftime("%H:%M")}')

        if isinstance(test_documents, str):
            test_documents = [test_documents]

        if self.config.multiclass:
            print(f'Building multiclass model for all authors.\n')
        else:
            print(f'Building binary LOO model for author {target}.\n')

        authors: list[str]

        documents, authors, filenames = self.load_dataset(test_documents, path=corpus_path)
        filenames = [f'{filename}_0' for filename in filenames]
        test_documents = [f'{filename}_0' for filename in test_documents]

        print(f'Available filenames: {filenames}')

        genres = ['Trattato' if 'epistola' not in filename.lower()
                else 'Epistola' for filename in filenames]

        print(f'Genres: {np.unique(genres, return_counts=True)}')

        processed_documents = self.get_processed_documents(documents, filenames)

        # replace the plain text documents with clean processed strings after spaCy
        documents = [processed_documents[filename[:-2]].text for filename in filenames]


        y = self.create_labels(authors, target, test_genre, genres)

        print(f'Label distribution: {np.unique(y, return_counts=True)}')


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
                    print(f'ERROR: Test document "{test_documents}" not found in available filenames')
                    print(f'Available filenames: {filenames}')
                    return
        else:
            test_indices = list(range(len(documents)))
            if self.config.multiclass:
                print(f'Full LOO evaluation: testing on all {len(test_indices)} documents (multiclass)')
            else:
                print(
                    f'Full LOO evaluation: testing on all {len(test_indices)} documents (binary classification for {target})')

        print(f'Total documents to test: {len(test_indices)}')

        for test_idx in test_indices:
            print(f'\n=== Processing document {test_idx + 1}/{len(test_indices)} ===')
            self.train_and_test_single_document(
                test_idx, test_indices, documents, y, processed_documents, filenames, target,
                save_results, self.config.results_filename,
                self.config.results_path,
                experiment_type=self.config.experiment
            )

        total_time = round((time.time() - start_time) / 60, 2)
        if self.config.multiclass:
            print(f'Total time spent for multiclass model building: {total_time} minutes.')
        else:
            print(f'Total time spent for model building for author {target}: {total_time} minutes.')


    def train_and_test_single_document(self, test_idx: int, test_indexes: List[int], documents: List[str], y: List[int],
                                       processed_documents: Dict[str, spacy.tokens.Doc],
                                       filenames: List[str], target: str, save_results: bool,
                                       file_name: str, path_name: str, experiment_type: str):
                                  
        """Process a single document for authorship verification"""
        start_time_single_iteration = time.time()
        np.random.seed(self.config.random_state)
        
        doc, ylabel = documents[test_idx], y[test_idx]

        if experiment_type == 'single-test':
            X_dev, X_test, y_dev, y_test, groups_dev, groups_test = self.loo_split(
                test_idx, documents, y, doc, ylabel, filenames
            )
        elif experiment_type == 'multiple-test':
            X_dev, X_test, y_dev, y_test, groups_dev, groups_test = self.loo_multiple_test_split(
                test_idx, test_indexes, documents, y, doc, ylabel, filenames
            )
        else:
            raise NotImplementedError('not yet implemented')

        (X_dev, X_test, y_dev, y_test, X_test_frag, y_test_frag, groups_dev, 
         groups_test, groups_test_frag) = self.segment_data(
            X_dev, X_test, y_dev, y_test, groups_dev, groups_test
        )
        print(np.unique(y, return_counts=True))

        X_dev_processed = self.get_processed_segments(
            processed_documents, X_dev, groups_dev, dataset='training'
        )
        X_test_processed = self.get_processed_segments(
            processed_documents, X_test, groups_test, dataset='test'
        )
        # X_test_frag_processed = self.get_processed_segments(
        #     processed_documents, X_test_frag, groups_test_frag,
        #     dataset='test fragments'
        # )

        X_len = len(X_dev_processed)
        print(f'X_len: {X_len}')
        
        (X_dev, X_test, y_dev, y_test, groups_dev, feature_idxs, 
         original_feature_idxs, original_X_dev, original_X_test, 
         orig_y_dev, orig_groups_dev) = self.extract_feature_vectors(
            X_dev_processed, X_test_processed, y_dev, y_test, 
            groups_dev
        )

        if self.config.load_model:
            print(f"Loading pre-trained model from: {self.config.load_model}")
            model = self.load_pretrained_model(self.config.load_model)
            if model is None:
                print("Failed to load model, training new one...")
                model = LogisticRegression(
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                )
                print(f'\nBuilding classifier...\n')
                clf = self.train_model(X_dev, y_dev, groups_dev, model, 'LogisticRegression')
            else:
                clf = model

        else:
            model = LogisticRegression(
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )
            print(f'\nBuilding classifier...\n')
            clf = self.train_model(X_dev, y_dev, groups_dev, model, 'LogisticRegression')

        acc, f1, cf, posterior_proba, predicted_author = self.evaluate_model(
            clf, X_test, y_test
        )

        doc_name = groups_test[0][:-2] if groups_test else f"doc_{test_idx}"
        self.save_trained_model(clf, target, doc_name)

        if save_results:
            self.save_results(
                target, acc, f1, posterior_proba, model,
                groups_test[0][:-2], feature_idxs.keys(),
                file_name, path_name, y_test, predicted_author)

        iteration_time = round((time.time() - start_time_single_iteration) / 60, 2)
        print(f'Time spent for model building for document {groups_test[0][:-2]}: {iteration_time} minutes.')

def main():
    
    config, target, test_document = ModelConfig.from_args()
    nlp = spacy.load('es_dep_news_trf')

    av_system = AuthorshipVerification(config, nlp)
    av_system.run(
        target=target,
        test_documents=test_document,
        multiclass=config.multiclass,
        save_results=config.save_res,
        test_genre=config.test_genre
    )

if __name__ == '__main__':
    main()
import argparse
import os.path
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
import spacy


from commons import AuthorshipVerification, QUIXOTE_DOCUMENTS
from data_preparation.data_loader import load_corpus, binarize_corpus

warnings.filterwarnings("ignore")



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
    results_filename: str = 'results.csv'

    @classmethod
    def from_args(cls):
        """Create config from command line args"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--train-dir', default='../corpus/training')
        parser.add_argument('--test-dir', default='../corpus/test')
        parser.add_argument('--positive-author', default='Cervantes',
                        help='If indicated (default: Cervantes), binarizes the corpus, '
                             'otherwise assumes multiclass classification')
        parser.add_argument('--results-filename', default='../results/inference_results.csv',
                    help='Filename for saving results')

        args = parser.parse_args()

        if '--target' in sys.argv and '--test-document' not in sys.argv:
            args.test_document = ''
            
        config = cls()
        config.train_dir = args.train_dir
        config.test_dir = args.test_dir
        config.positive_author = args.positive_author
        config.results_filename = args.results_filename

        parent_dir = Path(args.results_filename).parent
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        return config
            


def main():
    
    config = ModelConfig.from_args()

    spacy_language_model = spacy.load('es_dep_news_trf')

    train_corpus = load_corpus(config.train_dir, spacy_language_model)
    test_corpus = load_corpus(config.test_dir, spacy_language_model)

    if config.positive_author:
        train_corpus = binarize_corpus(train_corpus, positive_author=config.positive_author)
        test_corpus = binarize_corpus(test_corpus, positive_author=config.positive_author)

    av_system = AuthorshipVerification(config, nlp = spacy_language_model)
    av_system.fit(train_corpus)
    av_system.predict(test_corpus)

if __name__ == '__main__':
    main()
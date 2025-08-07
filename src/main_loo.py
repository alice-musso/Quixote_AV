import argparse
import sys
import warnings
from dataclasses import dataclass

import spacy

from commons import AuthorshipVerification, QUIXOTE_DOCUMENTS, load_dataset
from data_preparation.data_loader import load_corpus

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
        parser.add_argument('--results-filename', default='results/results.csv',
                    help='Filename for saving results')

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
            


def main():
    
    config, target, test_document = ModelConfig.from_args()
    nlp = spacy.load('es_dep_news_trf')

    train_corpus = load_corpus('../corpus/training')
    test_corpus = load_corpus('../corpus/test')

    av_system = AuthorshipVerification(config, nlp)
    av_system.run(
        target=target,
        test_documents=test_document
    )

if __name__ == '__main__':
    main()
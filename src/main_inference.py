import os.path
import sys
import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
import spacy


from commons import AuthorshipVerification
from data_preparation.data_loader import load_corpus, binarize_corpus

warnings.filterwarnings("ignore")



@dataclass
class ModelConfig:
    """Configuration for the model training and evaluation"""
    n_jobs: int = 30
    segment_min_token_size: int = 500
    random_state: int = 0
    max_features: int = 5000
    oversample: bool = False
    rebalance_ratio: float = 0.2
    save_res: bool = True
    results_inference: str = 'inference_results.csv'
    results_loo:str = 'loo_results.csv'
    classifier_type:str = "lr"

    @classmethod
    def from_args(cls):
        """Create config from command line args"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--train-dir', default='../corpus/training')
        parser.add_argument('--test-dir', default='../corpus/test')
        parser.add_argument('--positive-author', default='Cervantes',
                        help='If indicated (default: Cervantes), binarizes the corpus, '
                             'otherwise assumes multiclass classification')
        parser.add_argument('--results-inference', default='../results/inference/results.csv',
                    help='Filename for saving results')
        parser.add_argument('--results-loo', default='../results/loo/results.csv',
                            help='Filename for saving results for the leave one out whole books + segments')
        parser.add_argument('--classifier-type', choices = ["lr", "svm"], default='lr')

        args = parser.parse_args()

        if '--target' in sys.argv and '--test-document' not in sys.argv:
            args.test_document = ''
            
        config = cls()
        config.train_dir = args.train_dir
        config.test_dir = args.test_dir
        config.positive_author = args.positive_author
        config.classifier_type = args.classifier_type

        inference_path = Path(args.results_inference)
        config.results_inference = str(inference_path.parent / f"results_{config.positive_author}_{config.classifier_type}.csv")

        loo_path = Path(args.results_loo)
        config.results_loo = str(loo_path.parent / f"loo_results_{config.positive_author}_{config.classifier_type}.csv")

        for paths in [args.results_inference, args.results_loo]:
            Path(paths).parent.mkdir(parents=True, exist_ok=True)

        return config
            

if __name__ == '__main__':
    config = ModelConfig.from_args()


    train_corpus = load_corpus(config.train_dir)
    test_corpus = load_corpus(config.test_dir)

    if config.positive_author:
        train_corpus = binarize_corpus(train_corpus, positive_author=config.positive_author)
        test_corpus = binarize_corpus(test_corpus, positive_author=config.positive_author)

    spacy_language_model = spacy.load('es_dep_news_trf')
    av_system = AuthorshipVerification(config, nlp=spacy_language_model)
    av_system.fit(train_corpus)

    av_system.leave_one_out(train_corpus)

    predicted_authors, posteriors = av_system.predict(test_corpus, return_posteriors=True)

    if config.positive_author:
        index_of_positive_author= av_system.index_of_author(config.positive_author)

    # output
    for i, book in enumerate(test_corpus):
        print(f'"{book.title}" author={book.author}, '
              f'predicted={predicted_authors[i]}, '
              f'posterior={posteriors[i,index_of_positive_author]:.4f}')
import os.path
import pickle
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
    rebalance_ratio: float = 0.5
    save_res: bool = True
    results_inference: str = 'inference_results.csv'
    results_loo:str = 'loo_results.csv'
    hyperparams_save:str = 'hyperparameters_posauth_Cervantes.pkl'
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
        parser.add_argument('--hyperparams-save', default='../hyperparams/hyperparameters_posauth_Cervantes.pkl')
        parser.add_argument('--classifier-type', choices = ["lr", "svm"], default='lr')

        args = parser.parse_args()

        if '--target' in sys.argv and '--test-document' not in sys.argv:
            args.test_document = ''
            
        config = cls()
        config.train_dir = args.train_dir
        config.test_dir = args.test_dir
        config.positive_author = args.positive_author
        config.classifier_type = args.classifier_type
        config.results_inference = str(Path(args.results_inference).parent /
                                       f"results_{config.positive_author}_{config.classifier_type}.csv")
        config.results_loo = str(Path(args.results_loo).parent /
                                 f"loo_results_{config.positive_author}_{config.classifier_type}.csv")
        config.hyperparams_save = str(Path(args.hyperparams_save).parent /f"hyperparameters_posauth_Cervantes.pkl")
        for path in [config.results_inference, config.results_loo, config.hyperparams_save]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
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

    _, _, slices, _ = av_system.prepare_X_y(train_corpus)
    #hyperparams={
    #    'C': 0.1,
    #    'feat_funct_words': None,
    #    'feat_post': slice(313, 4088),
    #    'feat_mendenhall': slice(4088, 4113),
    #    'feat_sentlength': slice(4113, 5111),
    #    'feat_dvex': slice(5111, 5527),
    #    'feat_dep': slice(5561, 10561),
    #    'feat_char': slice(10561, 15561),
    #    'feat_k_freq_words': slice(15561, 18561),
    #    'rebalance_ratio': 0.5
    #}
    #av_system.fit_with_hyperparams(train_corpus, hyperparams=hyperparams)

    if config.positive_author == "Cervantes":
         av_system.fit(train_corpus, save_hyper_path=config.hyperparams_save)
    else:
        hyper_path = Path(config.hyperparams_save)
        if not hyper_path.exists():
            raise FileNotFoundError(f"{hyper_path} does not exist")
        with hyper_path.oper("rb") as f:
            hyperparams = pickle.load(f)
        av_system.fit_with_hyperparams(train_corpus, hyperparams=hyperparams)

    #av_system.leave_one_out(train_corpus)

    predicted_authors, posteriors = av_system.predict(test_corpus, return_posteriors=True)

    if config.positive_author:
        index_of_positive_author= av_system.index_of_author(config.positive_author)

    # output
    for i, book in enumerate(test_corpus):
        print(f'"{book.title}" author={book.author}, '
              f'predicted={predicted_authors[i]}, '
              f'posterior={posteriors[i,index_of_positive_author]:.4f}')
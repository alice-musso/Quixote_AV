import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
import spacy
import pandas as pd
import json
from src.data_preparation.data_loader import load_corpus, binarize_title
from Quijote_classifier import TextClassificationTrainer
from src.feature_extraction.features import FeaturesFrequentWords

warnings.filterwarnings("ignore")


@dataclass
class ModelConfig:
    """Configuration for the model training and evaluation"""
    n_jobs: int = 30
    segment_min_token_size: int = 500
    random_state: int = 0
    max_features: int = 3000
    save_res: bool = True
    results_inference: str = 'inference_results.csv'
    classifier_type: str = "lr"
    # Logistic Regression hyperparameters
    C: float = 1.0
    penalty: str = 'l2'
    solver: str = 'lbfgs'
    class_weight: str = 'balanced'
    # Add these attributes
    train_dir: str = '../../corpus/training'
    test_dir: str = '../../corpus/test'
    target_title: str = 'Quijote'
    feature_importance_file: str = 'feature_importance.json'

    @classmethod
    def from_args(cls):
        """Create config from command line args"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--train-dir', default='../Quijote_classifier/corpus/training')
        parser.add_argument('--test-dir', default='../Quijote_classifier/corpus/test')
        parser.add_argument('--target-title', default='Quijote')
        parser.add_argument('--max-features', type=int, default=3000,
                            help='Number of most frequent words to use')
        parser.add_argument('--results-inference', default='../../results/Quijote_class/results_inference.csv',
                            help='Filename for saving results')
        parser.add_argument('--feature-importance', default='../../results/Quijote_class/feature_importance.json',
                            help='Filename for saving feature importance')

        args = parser.parse_args()

        # Create config instance first
        config = cls()
        config.train_dir = args.train_dir
        config.test_dir = args.test_dir
        config.max_features = args.max_features
        config.results_inference = str(Path(args.results_inference).parent /
                                       f"results_Quijote.csv")
        config.feature_importance_file = str(Path(args.feature_importance).parent /
                                             f"feature_importance_Quijote.json")
        config.target_title = args.target_title

        Path(config.results_inference).parent.mkdir(parents=True, exist_ok=True)
        return config


if __name__ == '__main__':
    config = ModelConfig.from_args()

    train_corpus = load_corpus(config.train_dir)
    train_corpus = binarize_title(train_corpus, config.target_title)
    test_corpus = load_corpus(config.test_dir)
    test_corpus = binarize_title(test_corpus, config.target_title)

    trainer = TextClassificationTrainer(
        max_features=config.max_features,
        target_title=config.target_title,
        C=config.C,
        penalty=config.penalty,
        solver=config.solver,
        class_weight=config.class_weight,
        random_state=config.random_state,
        n_jobs=config.n_jobs
    )

    trainer.fit(train_corpus)

    predicted_labels = trainer.predict(test_corpus)
    posteriors = trainer.predict_proba(test_corpus)
    posterior_pos = posteriors[:, 1] if posteriors.shape[1] > 1 else posteriors[:, 0]

    labels = [1 if book.author == config.target_title else 0
                   for book in test_corpus]
    labels_str = [config.target_title if book.author == config.target_title
                  else f"Not{config.target_title}" for book in test_corpus]
    predicted_labels_str = [config.target_title if lbl == 1
                            else f"Not{config.target_title}" for lbl in predicted_labels]

    f1 = trainer.score(test_corpus, labels)


    feature_importance = trainer.get_feature_importance()
    with open(config.feature_importance_file, 'w', encoding='utf-8') as f:
        json.dump(feature_importance, f, ensure_ascii=False, indent=2)
    print(f"Feature importance saved to {config.feature_importance_file}")

    if config.save_res:
        results_df = pd.DataFrame({
            'book': [book.title for book in test_corpus],
            'label': labels_str,
            'prediction': predicted_labels_str,
            'posterior': posterior_pos,
            'f1': f1
        })
        results_df.to_csv(config.results_inference, index=False)
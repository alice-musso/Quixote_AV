from typing import List, Dict, Optional
import numpy as np
import spacy
from sklearn.linear_model import LogisticRegression

from src.feature_extraction.features import FeaturesFrequentWords
from src.data_preparation.data_loader import Book, get_spanish_function_words
from sklearn.metrics import f1_score
from supervised_term_weighting.supervised_vectorizer import TSRweighting
from supervised_term_weighting.tsr_functions import information_gain


class TextClassificationTrainer:
    """
    Binary text classifier based on frequent-word TF-IDF features
    and Logistic Regression.
    """

    def __init__(
            self,
            max_features: int = 3000,
            target_title: str = "Quijote",
            C: float = 1.0,
            penalty: str = "l2",
            solver: str = "lbfgs",
            class_weight: Optional[str] = None,
            random_state: int = 0,
            n_jobs: int = -1,
    ):
        self.max_features = max_features
        self.target_title = target_title
        self.n_jobs = n_jobs

        # Frequent words extractor
        self.vectorizer = FeaturesFrequentWords(
            max_features=max_features,
            remove_stopwords=list(get_spanish_function_words())
        )

        # Logistic Regression classifier
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.is_fitted = False
        self._last_train_documents = None
        self._last_train_labels = None

    def _prepare_training_data(self, books: List[Book]):
        """
        Extract texts and binary labels from Book objects.
        """
        documents = []
        labels = []

        for book in books:
            # Binary label: target title vs not
            if book.author == self.target_title:
                label = 1
            else:
                label = 0

            # Full book
            documents.append(book.processed)
            labels.append(label)

            # Segments (same label)
            if book.segmented is not None:
                for fragment in book.segmented:
                    documents.append(fragment)
                    labels.append(label)

        return documents, labels

    def fit(self, train_documents: List[Book]) -> "TextClassificationTrainer":
        """
        Train the classifier.
        """
        documents, y = self._prepare_training_data(train_documents)

        # Store for later use in feature importance
        self._last_train_documents = documents
        self._last_train_labels = y

        X = self.vectorizer.fit_transform(documents)

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, documents: List[Book]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        processed_docs = [doc.processed for doc in documents]
        X = self.vectorizer.transform(processed_docs)
        return self.model.predict(X)

    def predict_proba(self, documents: List[Book]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        processed_docs = [doc.processed for doc in documents]
        X = self.vectorizer.transform(processed_docs)
        return self.model.predict_proba(X)

    def score(self, documents: List[Book], true_labels: List[str]) -> float:
        predictions = self.predict(documents)
        return f1_score(true_labels, predictions, pos_label=1)

    def ___get_feature_importance(self) -> Dict[str, float]:
        """
        Get information gain scores for frequent words.

        Returns:
            Dictionary mapping words to their correlation with the positive class
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")

        from supervised_term_weighting.tsr_functions import (
            get_supervised_matrix, get_tsr_matrix, posneg_information_gain
        )

        vocabulary = self.vectorizer.vectorizer.get_feature_names_out()

        X = self.vectorizer.fit_transform(self._last_train_documents)

        n_samples = len(self._last_train_labels)
        label_matrix = np.zeros((n_samples, 2), dtype=int)
        labels = np.array(self._last_train_labels)
        label_matrix[np.arange(n_samples), labels] = 1

        supervised_matrix = get_supervised_matrix(X, label_matrix, n_jobs=self.n_jobs)
        tsr_matrix = get_tsr_matrix(supervised_matrix, posneg_information_gain, n_jobs=self.n_jobs)

        scores = tsr_matrix[1, :]
        feature_importance = {word: float(score) for word, score in zip(vocabulary, scores)}
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        return feature_importance
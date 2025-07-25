import json
import pickle
from pathlib import Path
from sklearn.base import BaseEstimator
from typing import Any, Dict, Optional, List
import json


class Labels:
    """class for creating labels for authors"""

    def __init__(self, authors_dir: str = "./authorslabel"):
        """
        Args:
            authors_dir: Directory to save/load models from
        """
        self.authors_dir = Path(authors_dir)
        self.authors_dir.mkdir(parents=True, exist_ok=True)
        self.author_to_id = {}
        self.id_to_author = {}

    def save_author_to_id(self, authors: List[str]) -> str:
        unique_authors = list(set(author.rstrip() for author in authors))
        self.author_to_id = {author: i for i, author in enumerate(unique_authors)}
        filename_author_to_id = "author_to_id.json"
        filepath_author_to_id = self.authors_dir / filename_author_to_id
        with open(filepath_author_to_id, "w") as f:
            json.dump(self.author_to_id, f)

        return str(filepath_author_to_id)

    def load_author_to_id(self, filepath: str) -> Dict[str, int]:
        filepath = Path(filepath)

        with open(filepath, 'r') as file:
            author_to_id = json.load(file)

        self.author_to_id = author_to_id
        return author_to_id

    def save_id_to_author(self, ids: dict[str, int]) -> str:
        self.id_to_author = {i: author for author, i in self.author_to_id.items()}
        #self.id_to_author = {k: v for k, v in id_to_author.items()}

        filename_id_to_author = "id_to_author.json"
        filepath_id_to_author = self.authors_dir / filename_id_to_author
        with open(filepath_id_to_author, "w") as f:
            json.dump(self.id_to_author, f)

        return str(filepath_id_to_author)

    def load_id_to_author(self, filepath: str) -> Dict[int, str]:
        filepath = Path(filepath)

        with open(filepath, 'r') as file:
            id_to_author = json.load(file)

        self.id_to_author = id_to_author
        return id_to_author


class Serialization:
    """class for serializing models"""

    def __init__(self, models_dir: str = "./models"):
        """
        Args:
            models_dir: Directory to save/load models from
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model: BaseEstimator, model_name: str,
                   metadata: Optional[Dict] = None) -> str:
        """
        Save a trained model to disk

        Args:
            model: The trained model to serialize
            model_name: Name for the saved model file

        Returns:
            str: Path to the saved model file
        """

        filename = f"{model_name}.pkl"
        filepath = self.models_dir / filename
        with open(filepath, 'wb') as file:
            pickle.dump(model, file)


        return str(filepath)

    def load_model(self, filepath: str) -> BaseEstimator:
        """
        Load a model from disk

        Args:
            filepath: Path to the saved model file

        Returns:
            The loaded model
        """
        filepath = Path(filepath)

        with open(filepath, 'rb') as file:
            model = pickle.load(file)

        print(f"Model loaded from: {filepath}")
        return model


    def list_saved_models(self) -> list:
        """
        List all saved models in the models directory

        Returns:
            List of model file paths
        """
        model_files = []
        for ext in ['*.pkl']:
            model_files.extend(list(self.models_dir.glob(ext)))

        return sorted(model_files)



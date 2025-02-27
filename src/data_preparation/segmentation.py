from dataclasses import dataclass
from typing import List, Tuple
from copy import copy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm


@dataclass
class Segmentation:
    """A class to split documents into segments based on various policies."""
    
    VALID_POLICIES = {'by_endline', 'by_sentence'}
    
    split_policy: str = 'by_sentence'
    tokens_per_fragment: int = 500
    min_tokens: int = 8
    keep_full: bool = True
    groups: List[str] = None
    
    def __post_init__(self):
        """Validate initialization parameters."""
        if self.split_policy not in self.VALID_POLICIES:
            raise ValueError(
                f'Unknown policy, valid ones are {self.VALID_POLICIES}'
            )
    
    def fit(self, X: List[str], y: List[str]) -> 'Segmentation':
        """Fit method to maintain scikit-learn API compatibility."""
        return self
    
    def transform(
        self, 
        documents: List[str], 
        authors: List[str], 
        filenames: List[str], 
        label_encode: bool = False
    ) -> Tuple[List[str], List[str]]:
        """Transform documents into segments according to the specified policy."""
        
        # Initialize fragments and authors lists
        fragments = copy(documents) if self.keep_full else []
        authors_fragments = copy(authors) if self.keep_full else []
        groups = filenames if self.keep_full else []
        
        # Process each document
        for i, (text, group) in tqdm(
            enumerate(zip(documents, groups)), 
            total=len(documents),
            desc='Generating fragments'
        ):
            # Split text according to policy
            text_fragments = (
                self._split_by_endline(text) 
                if self.split_policy == 'by_endline'
                else self._split_by_sentences(text)
            )
            
            # Create windows of fragments
            text_fragments = self._create_windows(
                text_fragments, 
                self.tokens_per_fragment
            )
            
            # Extend results
            fragments.extend(text_fragments)
            groups.extend([group] * len(text_fragments))
            
            if authors is not None:
                authors_fragments.extend([authors[i]] * len(text_fragments))
        
        self.groups = self._add_indices(groups)
        return fragments, authors_fragments
    
    def fit_transform(
        self, 
        documents: List[str], 
        authors: List[str], 
        filenames: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Combined fit and transform operations."""
        return self.fit(documents, authors).transform(
            documents, authors, filenames=filenames
        )
    
    def _split_by_endline(self, text: str) -> List[str]:
        """Split text by newlines and remove empty lines."""
        return [line.strip() for line in text.split('\n') if line.strip()]
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences and merge short ones."""
        sentences = sent_tokenize(text)
        
        i = 0
        while i < len(sentences):
            if len(tokenize(sentences[i])) < self.min_tokens:
                if i < len(sentences) - 1:
                    # Merge with next sentence
                    sentences[i + 1] = f"{sentences[i]} {sentences[i + 1]}"
                else:
                    # Merge with previous sentence
                    sentences[i - 1] = f"{sentences[i - 1]} {sentences[i]}"
                sentences.pop(i)
                continue
            i += 1
            
        return sentences
    
    def _create_windows(
        self, 
        text_fragments: List[str], 
        tokens_per_fragment: int
    ) -> List[str]:
        """Create windows of text fragments based on token count."""
        new_fragments = []
        current_batch = ""
        
        for fragment in text_fragments:
            token_count = len(word_tokenize(fragment))
            
            if token_count >= tokens_per_fragment:
                new_fragments.append(fragment)
                continue
                
            current_batch = (
                f"{current_batch} {fragment}" 
                if current_batch 
                else fragment
            )
            
            if len(word_tokenize(current_batch)) >= tokens_per_fragment:
                new_fragments.append(current_batch.strip())
                current_batch = ""
        
        # Add any remaining batch
        if current_batch:
            new_fragments.append(current_batch.strip())
            
        return new_fragments
    
    @staticmethod    
    def _add_indices(filenames: List[str]) -> List[str]:
        """Add indices to filenames to make them unique."""
        count = {}
        indexed_groups = []
        
        for name in filenames:
            count[name] = count.get(name, -1) + 1
            indexed_groups.append(f"{name}_{count[name]}")
            
        return indexed_groups
    
# helper function
def tokenize(text):
    return [token.lower() for token in nltk.word_tokenize(text) if any(char.isalpha() for char in token)]

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
    
    def __init__(self, split_policy='by_sentence', tokens_per_fragment=500, min_tokens=0):
        assert split_policy in Segmentation.VALID_POLICIES, \
            f'unknown {split_policy=}, valid ones are {self.VALID_POLICIES}'
        self.split_policy = split_policy
        self.tokens_per_fragment = tokens_per_fragment
        self.min_tokens = min_tokens

    def transform(self, text):
        """Transform documents into segments according to the specified policy."""

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

        return text_fragments
    
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
    
# helper function
def tokenize(text):
    return [token.lower() for token in nltk.word_tokenize(text) if any(char.isalpha() for char in token)]

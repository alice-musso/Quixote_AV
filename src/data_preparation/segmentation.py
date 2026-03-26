from typing import List, Tuple
import spacy
from spacy.tokens import Doc


class Segmentator:

    def __init__(self, min_tokens: int = 500):
        """
        Parameters
        ----------
        min_tokens : int
            Min number of tokens in a segment
        """
        self.min_tokens = min_tokens

    def transform(self, doc: Doc):
        """
        Segments a document into chunks made of complete sentences, with an overall length >= min_tokens

        Parameters
        ----------
        doc : spacy.tokens.Doc
            A document already processed by spacy

        Returns
        -------
        List[spacy.tokens.Doc]
            list of spacy docs
        """
        segments = []
        segments_starts = []
        current_sents = []
        current_len = 0

        for sent in doc.sents:
            current_sents.append(sent)
            current_len += len(sent)

            # segmentation criterion
            if current_len >= self.min_tokens:
                start = current_sents[0].start
                end = current_sents[-1].end
                segments.append(doc[start:end].as_doc())
                segments_starts.append(start)
                current_sents = []
                current_len = 0

        # last block
        if current_sents:
            if segments:
                # extends the last segment to include the remainder
                last_start = segments_starts[-1]
                last_end = current_sents[-1].end
                segments[-1] = doc[last_start:last_end].as_doc()
            else:
                # if no segment was created, then this is the only one
                start = current_sents[0].start
                end = current_sents[-1].end
                segments.append(doc[start:end].as_doc())

        return segments


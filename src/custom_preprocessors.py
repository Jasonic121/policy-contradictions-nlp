"""
Custom HayStack PreProcessors.
"""

from typing import (Callable, Dict, Generator, List, Literal, Optional, Set,
                    Tuple, Union)

from haystack.nodes import PreProcessor
from haystack.schema import Document
from tqdm.auto import tqdm
import spacy

class SplitCleanerPreProcessor(PreProcessor):
    def __init__(
        self,
        *args,
        split_cleaner: Optional[Callable[[List[str]], List[str]]] = None,
        **kwargs,
    ):
        """
        Extension of PreProcessor which supports providing a cleaning function
        which will apply to the individual split units prior to assembling into
        documents of split_length.
        """
        super().__init__(*args, **kwargs)
        self.split_cleaner = split_cleaner
        # Load spaCy model for sentence splitting
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'parser'])
        self.nlp.enable_pipe('senter')

    def _split_into_units(
        self, 
        text: str, 
        split_by: str,
        tokenizer: Optional[any] = None  # Add tokenizer parameter
    ) -> Tuple[List[str], str]:
        """Split text into units using the specified method."""
        if split_by == "passage":
            elements = text.split("\n\n")
            split_at = "\n\n"
        elif split_by == "sentence":
            doc = self.nlp(text)
            elements = [sent.text.strip() for sent in doc.sents]
            split_at = " "
        elif split_by == "word":
            elements = text.split(" ")
            split_at = " "
        else:
            raise NotImplementedError("PreProcessor only supports 'passage', 'sentence' or 'word' split_by options.")

        if self.split_cleaner is not None:
            elements = self.split_cleaner(elements)
        return elements, split_at


class NestedPreProcessor(PreProcessor):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Override to PreProcessor which returns processed docs as a list of lists
        of documents rather than flattening.
        """
        super().__init__(*args, **kwargs)

    def _process_batch(
        self, documents: List[Union[dict, Document]], id_hash_keys: Optional[List[str]] = None, **kwargs
    ) -> List[List[Document]]:
        nested_docs = [
            self._process_single(d, id_hash_keys=id_hash_keys, **kwargs)
            for d in tqdm(documents, disable=not self.progress_bar, desc="Preprocessing", unit="docs")
        ]
        return nested_docs

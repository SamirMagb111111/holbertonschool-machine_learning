#!/usr/bin/env python3
"""Dataset class for the Portuguese-to-English TED translation corpus."""
import transformers
from setup import load_pt2en


class Dataset:
    """Loads and tokenizes the pt-to-en TED translation dataset."""

    def __init__(self):
        """Load the train/validation splits and train the tokenizers."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """Train sub-word tokenizers on ``data``.

        Args:
            data: tf.data.Dataset of ``(pt, en)`` tf.string tensor pairs.

        Returns:
            tokenizer_pt, tokenizer_en: the trained BertTokenizerFast objects
            with a vocabulary of ``2 ** 13`` tokens.
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased')

        def iterate(index):
            """Yield the decoded sentences of one language from ``data``."""
            for pair in data:
                yield pair[index].numpy().decode('utf-8')

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            iterate(0), vocab_size=2 ** 13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            iterate(1), vocab_size=2 ** 13)
        return tokenizer_pt, tokenizer_en

#!/usr/bin/env python3
"""Dataset class for the pt-to-en TED corpus with a full data pipeline."""
import transformers
import tensorflow as tf
from setup import load_pt2en


class Dataset:
    """Loads, tokenizes and pipelines the pt-to-en TED dataset."""

    def __init__(self, batch_size, max_len):
        """Build the train and validation pipelines.

        Args:
            batch_size: batch size for the padded batches.
            max_len: maximum number of tokens allowed in either sentence.
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        def keep(pt, en):
            """Return True when both sentences fit within ``max_len``."""
            return tf.logical_and(tf.size(pt) <= max_len,
                                  tf.size(en) <= max_len)

        self.data_train = self.data_train.filter(keep)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(
            batch_size, padded_shapes=([None], [None]))
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

        self.data_valid = self.data_valid.filter(keep)
        self.data_valid = self.data_valid.padded_batch(
            batch_size, padded_shapes=([None], [None]))

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

    def encode(self, pt, en):
        """Encode a pt/en sentence pair into lists of token ids.

        A sentence-start token (``vocab_size``) and a sentence-end token
        (``vocab_size + 1``) are added around each encoded sentence.

        Args:
            pt: tf.Tensor containing the Portuguese sentence.
            en: tf.Tensor containing the English sentence.

        Returns:
            pt_tokens, en_tokens: Python lists of ints.
        """
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        pt_tokens = self.tokenizer_pt.encode(
            pt_text, add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(
            en_text, add_special_tokens=False)

        pt_tokens = ([self.tokenizer_pt.vocab_size] + pt_tokens +
                     [self.tokenizer_pt.vocab_size + 1])
        en_tokens = ([self.tokenizer_en.vocab_size] + en_tokens +
                     [self.tokenizer_en.vocab_size + 1])
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """TensorFlow wrapper around ``encode``.

        Args:
            pt: tf.Tensor containing the Portuguese sentence.
            en: tf.Tensor containing the English sentence.

        Returns:
            pt_tokens, en_tokens: ``tf.int64`` tensors of rank 1.
        """
        pt_tokens, en_tokens = tf.py_function(
            self.encode, [pt, en], [tf.int64, tf.int64])
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        return pt_tokens, en_tokens

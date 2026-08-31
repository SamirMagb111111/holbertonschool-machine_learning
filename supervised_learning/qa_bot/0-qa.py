#!/usr/bin/env python3
"""Question answering with a BERT model served from TensorFlow Hub."""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")
model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")


def question_answer(question, reference):
    """Find the snippet of ``reference`` that answers ``question``.

    Args:
        question: string containing the question to answer.
        reference: string containing the reference document to search.

    Returns:
        String with the answer, or ``None`` if no answer can be extracted.
    """
    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

    # keep the whole sequence within BERT's 512 token limit
    max_reference = 512 - len(question_tokens) - 3
    if max_reference < 0:
        max_reference = 0
    reference_tokens = reference_tokens[:max_reference]

    tokens = (["[CLS]"] + question_tokens + ["[SEP]"] +
              reference_tokens + ["[SEP]"])

    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = ([0] * (len(question_tokens) + 2) +
                      [1] * (len(reference_tokens) + 1))

    input_word_ids, input_mask, input_type_ids = map(
        lambda t: tf.expand_dims(tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_word_ids, input_mask, input_type_ids))

    outputs = model([input_word_ids, input_mask, input_type_ids])
    short_start = int(tf.argmax(outputs[0][0][1:]) + 1)
    short_end = int(tf.argmax(outputs[1][0][1:]) + 1)

    if short_end < short_start:
        return None

    answer_tokens = tokens[short_start:short_end + 1]
    if not answer_tokens or set(answer_tokens) <= {"[CLS]", "[SEP]", "[PAD]"}:
        return None

    answer = tokenizer.convert_tokens_to_string(answer_tokens).strip()
    if not answer:
        return None
    return answer

#!/usr/bin/env python3
"""Multi-reference question answering: semantic search then BERT QA."""

qa = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search

EXIT_WORDS = ("exit", "quit", "goodbye", "bye")


def question_answer(corpus_path):
    """Answer user questions using the most relevant document of a corpus.

    For every question the most semantically similar document is selected
    with ``semantic_search`` and passed to the BERT QA model.

    Args:
        corpus_path: path to the directory holding the reference documents.
    """
    while True:
        question = input("Q: ")
        if question.strip().lower() in EXIT_WORDS:
            print("A: Goodbye")
            break
        reference = semantic_search(corpus_path, question)
        answer = qa(question, reference)
        if not answer:
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))

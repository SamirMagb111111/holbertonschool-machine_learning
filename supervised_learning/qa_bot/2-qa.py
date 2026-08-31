#!/usr/bin/env python3
"""Interactive question answering loop over a single reference document."""

question_answer = __import__('0-qa').question_answer

EXIT_WORDS = ("exit", "quit", "goodbye", "bye")


def answer_loop(reference):
    """Answer user questions about ``reference`` until an exit word is given.

    Args:
        reference: string containing the reference document.
    """
    while True:
        question = input("Q: ")
        if question.strip().lower() in EXIT_WORDS:
            print("A: Goodbye")
            break
        answer = question_answer(question, reference)
        if not answer:
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))

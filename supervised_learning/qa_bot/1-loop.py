#!/usr/bin/env python3
"""Minimal interactive question/answer prompt loop."""

EXIT_WORDS = ("exit", "quit", "goodbye", "bye")


def main():
    """Prompt the user with ``Q:`` until an exit word is entered."""
    while True:
        question = input("Q: ")
        if question.strip().lower() in EXIT_WORDS:
            print("A: Goodbye")
            break
        print("A:")


if __name__ == "__main__":
    main()

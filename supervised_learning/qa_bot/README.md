# QA Bot

A question-answering bot that extracts answers from reference documents using
BERT, and picks the right document with semantic search.

## Concepts

- **Question Answering (extractive QA)** — given a question and a reference
  text, return the exact span of the text that answers the question (or
  nothing if the text has no answer).
- **BERT** — a transformer language model. The fine-tuned checkpoint
  `bert-large-uncased-whole-word-masking-finetuned-squad` predicts, for each
  token, how likely it is to be the start or the end of the answer span.
- **Semantic search** — ranking documents by meaning rather than keywords, by
  embedding the query and each document into the same vector space and taking
  the closest one (here, inner-product similarity of Universal Sentence
  Encoder embeddings).
- **TensorFlow Hub** — hosts the pre-trained models used here:
  `see--/bert-uncased-tf2-qa/1` for QA and
  `google/universal-sentence-encoder-large/5` for embeddings.
- **Transformers** — the Hugging Face library; used for `BertTokenizer`
  (WordPiece tokenization and decoding).
- **Multi-reference QA** — combining semantic search and extractive QA: first
  find the most relevant document in a corpus, then extract the answer from
  it.

## Files

| File | Description |
| --- | --- |
| `0-qa.py` | `question_answer(question, reference)` — BERT span extraction from one document. |
| `1-loop.py` | Interactive `Q:` / `A:` loop with case-insensitive exit words (`exit`, `quit`, `goodbye`, `bye`). |
| `2-qa.py` | `answer_loop(reference)` — Task 1 loop answering with Task 0 over a single document. |
| `3-semantic_search.py` | `semantic_search(corpus_path, sentence)` — return the full text of the most similar corpus document. |
| `4-qa.py` | `question_answer(corpus_path)` — loop combining semantic search and BERT QA over a corpus. |

## Usage

```python
answer_loop = __import__('2-qa').answer_loop
with open('ZendeskArticles/PeerLearningDays.md') as f:
    answer_loop(f.read())
```

```python
question_answer = __import__('4-qa').question_answer
question_answer('ZendeskArticles')
```

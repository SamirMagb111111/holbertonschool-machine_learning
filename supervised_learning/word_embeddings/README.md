# Word Embeddings

Turning text into numeric vectors, from simple counts to contextual models.

## Concepts

- **Natural Language Processing (NLP)** — getting computers to work with human
  language: classification, translation, question answering, and more.
- **Word embeddings** — dense vector representations of words where distance
  and direction carry meaning, learned so similar words sit close together.
- **Bag of Words** — represent a sentence by how many times each vocabulary
  word appears; order is ignored.
- **TF-IDF** — like Bag of Words, but each count is weighted down for words
  that are common across many documents, up for rare informative ones.
- **CBOW** — Word2Vec training mode that predicts a target word from the
  average of its surrounding context words.
- **Skip-gram** — Word2Vec training mode that predicts the context words from
  a single target word; better for rare words.
- **Negative sampling** — training shortcut that, instead of a full softmax,
  updates the true context pair and a few randomly sampled "negative" pairs.
- **Word2Vec** — shallow neural model (CBOW or Skip-gram) that learns one
  vector per vocabulary word.
- **FastText** — Word2Vec extension that represents a word as the sum of its
  character n-gram vectors, so it can embed out-of-vocabulary and misspelled
  words.
- **ELMo** — deep contextual embeddings from a bidirectional LSTM language
  model over character-based token representations; a word's vector depends
  on the whole sentence.

## Tasks

| File | Description |
| --- | --- |
| `0-bag_of_words.py` | `bag_of_words(sentences, vocab=None)` — Bag of Words matrix (sklearn `CountVectorizer`). |
| `1-tf_idf.py` | `tf_idf(sentences, vocab=None)` — TF-IDF matrix (sklearn `TfidfVectorizer`). |
| `2-word2vec.py` | `word2vec_model(...)` — train a gensim Word2Vec model. |
| `3-gensim_to_keras.py` | `gensim_to_keras(model)` — trainable Keras `Embedding` layer from Word2Vec weights. |
| `4-fasttext.py` | `fasttext_model(...)` — train a gensim FastText model. |
| `5-elmo` | Multiple-choice answer: what is trained during ELMo pre-training. |

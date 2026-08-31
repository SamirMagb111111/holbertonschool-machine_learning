# Transformer Applications

Train a Transformer from *Attention Is All You Need* to translate
**Portuguese → English** on the TED HRLR talks dataset.

## Overview

- **Dataset** — `ted_hrlr_translate/pt_to_en`, loaded through the project's
  `setup.py` helper (`load_pt2en('train' | 'validation' | 'test')`) which
  returns a `tf.data.Dataset` of `(pt, en)` `tf.string` pairs.
- **Tokenizers** — Hugging Face fast BERT tokenizers
  (`neuralmind/bert-base-portuguese-cased`, `bert-base-uncased`) retrained on
  the TED training text with `train_new_from_iterator`, vocabulary
  `2 ** 13 = 8192`. **Sub-word tokenization** keeps the vocabulary small
  while covering rare words.
- **Start / end tokens** — added manually as ids `vocab_size` (8192) and
  `vocab_size + 1` (8193); the Transformer embeddings therefore use
  `vocab_size + 2`.
- **`tf.py_function`** — wraps the Python tokenizer call so `encode` can run
  inside a `tf.data` `.map`; static shapes are restored with `set_shape`.
- **Pipeline** (`3-dataset.py`) — `filter` (drop sentences longer than
  `max_len`) → `cache` → `shuffle(20000)` → `padded_batch` (pad with 0) →
  `prefetch`; validation is only filtered and padded-batched.
- **Masks** (`4-create_masks.py`) — encoder padding mask, decoder look-ahead
  + target-padding combined mask, and the encoder-decoder padding mask
  (built from the *inputs*).
- **Model** (`5-transformer.py`) — self-contained Transformer: TensorFlow-only
  positional encoding, scaled dot-product attention, multi-head attention,
  encoder / decoder blocks with residual connections, layer norm and
  dropout, final `Dense(target_vocab)` returning logits.
- **Training** (`5-train.py`) — custom loop with teacher forcing
  (`target[:, :-1]` in, `target[:, 1:]` compared), masked loss
  (`SparseCategoricalCrossentropy(from_logits=True)` ignoring padding),
  masked accuracy, Adam (`beta_1=0.9, beta_2=0.98, epsilon=1e-9`) and the
  **warmup learning-rate schedule** (`warmup_steps=4000`). Progress is
  printed every 50 batches and at the end of each epoch.

## Files

| File | Content |
| --- | --- |
| `setup.py` | Official Holberton dataset loader (`load_pt2en`). |
| `0-dataset.py` | `Dataset` — load splits, train tokenizers. |
| `1-dataset.py` | adds `Dataset.encode`. |
| `2-dataset.py` | adds `Dataset.tf_encode`, maps the splits. |
| `3-dataset.py` | `Dataset(batch_size, max_len)` — full data pipeline. |
| `4-create_masks.py` | `create_masks(inputs, target)`. |
| `5-transformer.py` | self-contained `Transformer`. |
| `5-train.py` | `train_transformer(N, dm, h, hidden, max_len, batch_size, epochs)`. |

# NLP Metrics

From-scratch implementations of the BLEU score for evaluating machine
translation / text generation.

## Concepts

- **NLP evaluation metrics** — automatic scores that compare a model's
  generated text against one or more human reference texts.
- **BLEU score** — precision-oriented metric: how many of the candidate's
  n-grams appear in the references, combined with a length penalty. Ranges
  from 0 to 1 (higher is better).
- **Modified n-gram precision** — matching n-grams are *clipped* to the
  maximum number of times the n-gram occurs in any single reference, so
  repeating a good word many times cannot inflate the score.
- **Brevity penalty** — `BP = 1` if the candidate is longer than the closest
  reference, else `exp(1 - r / c)` where `c` is the candidate length and `r`
  the length of the reference closest to it (ties → shorter). It punishes
  translations that are too short.
- **Unigram BLEU** — `BP * p1`, using only single-word precision.
- **N-gram BLEU** — `BP * pn`, using only n-gram precision of one order.
- **Cumulative BLEU** — `BP * (p1 * p2 * ... * pn) ** (1 / n)`, the geometric
  mean of every order from 1 to n with equal weights; returns 0 if any
  precision is 0.
- **ROUGE** — a recall-oriented counterpart used mainly for summarization
  (overlap of n-grams / longest common subsequence with the reference).
- **Perplexity** — how well a language model predicts a text; the
  exponentiated average negative log-likelihood, lower is better.

## Files

| File | Function | Description |
| --- | --- | --- |
| `0-uni_bleu.py` | `uni_bleu(references, sentence)` | Unigram BLEU score. |
| `1-ngram_bleu.py` | `ngram_bleu(references, sentence, n)` | BLEU using only n-gram precision. |
| `2-cumulative_bleu.py` | `cumulative_bleu(references, sentence, n)` | Cumulative BLEU up to n-grams, equal weights. |

# Attention

Attention mechanisms, from Bahdanau attention on an RNN translator to the full
Transformer of *Attention Is All You Need*.

## Concepts

- **Attention mechanism** — instead of compressing a whole sequence into one
  vector, the decoder looks back at every encoder state and takes a weighted
  average, where the weights say how relevant each position is right now.
- **Encoder-decoder architecture** — an encoder turns the source sequence
  into a set of representations; a decoder generates the target sequence one
  token at a time, conditioned on those representations.
- **Bahdanau / self attention (RNN decoder)** — additive attention:
  `score = V·tanh(W·s_prev + U·hidden_states)`, softmax over the source
  positions, context = weighted sum of encoder states.
- **Positional encoding** — since attention has no notion of order, fixed
  sine/cosine signals (even dims → sin, odd dims → cos) are added to the
  embeddings so the model can use position.
- **Scaled dot-product attention** — `softmax(QKᵀ / √dk + mask)·V`.
- **Multi-head attention** — project Q, K, V `h` times into `dm/h`-dim
  subspaces, run attention in parallel, concatenate, project back.
- **Transformer encoder** — embedding × √dm + positional encoding, then N
  blocks of {multi-head self-attention, feed-forward}.
- **Transformer decoder** — N blocks of {masked self-attention,
  encoder-decoder attention, feed-forward}.
- **Residual connections** — every sub-layer output is added to its input
  (`x + sublayer(x)`) so gradients flow and layers refine rather than replace.
- **Layer normalization** — applied after each residual add (`epsilon=1e-6`)
  to stabilize activations.
- **Dropout** — applied to sub-layer outputs and to the embedding sum during
  training for regularization.
- **BERT / GPT** — Transformer-based pretrained models: BERT stacks encoders
  and is trained bidirectionally (masked LM); GPT stacks decoders and is
  trained left-to-right (causal LM).

## Files

| File | Content |
| --- | --- |
| `0-rnn_encoder.py` | `RNNEncoder` — GRU encoder. |
| `1-self_attention.py` | `SelfAttention` — Bahdanau attention. |
| `2-rnn_decoder.py` | `RNNDecoder` — GRU decoder with attention. |
| `4-positional_encoding.py` | `positional_encoding(max_seq_len, dm)`. |
| `5-sdp_attention.py` | `sdp_attention(Q, K, V, mask=None)`. |
| `6-multihead_attention.py` | `MultiHeadAttention`. |
| `7-transformer_encoder_block.py` | `EncoderBlock`. |
| `8-transformer_decoder_block.py` | `DecoderBlock`. |
| `9-transformer_encoder.py` | `Encoder`. |
| `10-transformer_decoder.py` | `Decoder`. |
| `11-transformer.py` | `Transformer`. |

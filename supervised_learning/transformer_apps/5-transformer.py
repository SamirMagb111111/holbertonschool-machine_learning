#!/usr/bin/env python3
"""Self-contained Transformer network for machine translation."""
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """Compute the positional encoding matrix with TensorFlow ops.

    Args:
        max_seq_len: maximum sequence length.
        dm: model depth.

    Returns:
        tf.Tensor of shape ``(max_seq_len, dm)``.
    """
    positions = tf.range(max_seq_len, dtype=tf.float32)[:, tf.newaxis]
    indices = tf.range(dm, dtype=tf.float32)[tf.newaxis, :]
    angle_rates = 1 / tf.pow(
        10000.0, (2 * tf.math.floor(indices / 2)) / tf.cast(dm, tf.float32))
    angles = positions * angle_rates
    sines = tf.math.sin(angles[:, 0::2])
    cosines = tf.math.cos(angles[:, 1::2])
    pos_encoding = tf.reshape(
        tf.stack([sines, cosines], axis=-1), (max_seq_len, dm))
    return pos_encoding


def sdp_attention(Q, K, V, mask=None):
    """Compute scaled dot-product attention.

    Args:
        Q: query tensor, ``(..., seq_len_q, dk)``.
        K: key tensor, ``(..., seq_len_v, dk)``.
        V: value tensor, ``(..., seq_len_v, dv)``.
        mask: optional mask broadcastable to
            ``(..., seq_len_q, seq_len_v)``.

    Returns:
        output: ``(..., seq_len_q, dv)``.
        weights: ``(..., seq_len_q, seq_len_v)``.
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += mask * -1e9
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Perform multi-head attention."""

    def __init__(self, dm, h):
        """Initialize the layer with ``h`` heads and model depth ``dm``."""
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch):
        """Reshape ``x`` to ``(batch, h, seq_len, depth)``."""
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """Apply multi-head attention and return ``(output, weights)``."""
        batch = tf.shape(Q)[0]
        Q = self.split_heads(self.Wq(Q), batch)
        K = self.split_heads(self.Wk(K), batch)
        V = self.split_heads(self.Wv(V), batch)

        scaled_attention, weights = sdp_attention(Q, K, V, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch, -1, self.dm))
        output = self.linear(concat_attention)
        return output, weights


class EncoderBlock(tf.keras.layers.Layer):
    """A single Transformer encoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the block."""
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Run the encoder block and return ``(batch, seq_len, dm)``."""
        attention, _ = self.mha(x, x, x, mask)
        attention = self.dropout1(attention, training=training)
        out1 = self.layernorm1(x + attention)

        ffn = self.dense_hidden(out1)
        ffn = self.dense_output(ffn)
        ffn = self.dropout2(ffn, training=training)
        return self.layernorm2(out1 + ffn)


class DecoderBlock(tf.keras.layers.Layer):
    """A single Transformer decoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the block."""
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """Run the decoder block and return ``(batch, seq_len, dm)``."""
        attention1, _ = self.mha1(x, x, x, look_ahead_mask)
        attention1 = self.dropout1(attention1, training=training)
        out1 = self.layernorm1(x + attention1)

        attention2, _ = self.mha2(out1, encoder_output, encoder_output,
                                  padding_mask)
        attention2 = self.dropout2(attention2, training=training)
        out2 = self.layernorm2(out1 + attention2)

        ffn = self.dense_hidden(out2)
        ffn = self.dense_output(ffn)
        ffn = self.dropout3(ffn, training=training)
        return self.layernorm3(out2 + ffn)


class Encoder(tf.keras.layers.Layer):
    """The full Transformer encoder."""

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initialize the encoder."""
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Encode ``x`` (token indices) into ``(batch, seq_len, dm)``."""
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += tf.cast(self.positional_encoding[:seq_len], x.dtype)
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, training, mask)
        return x


class Decoder(tf.keras.layers.Layer):
    """The full Transformer decoder."""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initialize the decoder."""
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """Decode ``x`` (token indices) into ``(batch, seq_len, dm)``."""
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += tf.cast(self.positional_encoding[:seq_len], x.dtype)
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, encoder_output, training, look_ahead_mask,
                      padding_mask)
        return x


class Transformer(tf.keras.Model):
    """A Transformer model for sequence-to-sequence tasks."""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """Initialize the Transformer."""
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_input,
                               drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """Run the Transformer and return ``(batch, seq_len, target_vocab)``.

        Returns raw logits (no softmax).
        """
        encoder_output = self.encoder(inputs, training, encoder_mask)
        decoder_output = self.decoder(target, encoder_output, training,
                                      look_ahead_mask, decoder_mask)
        return self.linear(decoder_output)

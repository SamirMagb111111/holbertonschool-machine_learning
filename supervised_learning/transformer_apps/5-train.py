#!/usr/bin/env python3
"""Custom training loop for the pt-to-en translation Transformer."""
import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """The "Attention Is All You Need" learning-rate schedule."""

    def __init__(self, dm, warmup_steps=4000):
        """Store the model depth and the number of warmup steps."""
        super().__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """Return the learning rate for training ``step``."""
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Train a Transformer on the pt-to-en dataset.

    Args:
        N: number of encoder and decoder blocks.
        dm: model dimensionality.
        h: number of attention heads.
        hidden: feed-forward hidden units.
        max_len: maximum number of tokens per sentence.
        batch_size: training batch size.
        epochs: number of training epochs.

    Returns:
        The trained Transformer model.
    """
    data = Dataset(batch_size, max_len)
    input_vocab = data.tokenizer_pt.vocab_size + 2
    target_vocab = data.tokenizer_en.vocab_size + 2

    transformer = Transformer(N, dm, h, hidden, input_vocab, target_vocab,
                              max_len, max_len)

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                         beta_2=0.98, epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        """Masked sparse categorical crossentropy over real tokens."""
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    def accuracy_function(real, pred):
        """Masked token-level accuracy over real tokens."""
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)
        accuracies = tf.cast(accuracies, tf.float32)
        mask = tf.cast(mask, tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    @tf.function
    def train_step(inputs, target):
        """Run one teacher-forced gradient-descent step."""
        target_input = target[:, :-1]
        target_real = target[:, 1:]
        enc_mask, comb_mask, dec_mask = create_masks(inputs, target_input)
        with tf.GradientTape() as tape:
            predictions = transformer(inputs, target_input, True, enc_mask,
                                      comb_mask, dec_mask)
            loss = loss_function(target_real, predictions)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables))
        train_loss(loss)
        train_accuracy(accuracy_function(target_real, predictions))

    for epoch in range(epochs):
        train_loss.reset_state()
        train_accuracy.reset_state()
        for batch, (inputs, target) in enumerate(data.data_train):
            train_step(inputs, target)
            if batch % 50 == 0:
                print('Epoch {}, Batch {}: Loss {} Accuracy {}'.format(
                    epoch + 1, batch, train_loss.result(),
                    train_accuracy.result()))
        print('Epoch {}: Loss {} Accuracy {}'.format(
            epoch + 1, train_loss.result(), train_accuracy.result()))

    return transformer

#!/usr/bin/env python3
"""Simple GAN implementation based on keras.Model."""
import tensorflow as tf
from tensorflow import keras


class Simple_GAN(keras.Model):
    """A simple Generative Adversarial Network.

    The generator is trained so that the discriminator outputs 1 on fake
    samples, and the discriminator is trained so that it outputs 1 on real
    samples and -1 on fake samples, both through a mean squared error.
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """Initialize the GAN and compile both sub models.

        Args:
            generator: the generator keras model.
            discriminator: the discriminator keras model.
            latent_generator: callable returning latent vectors given a size.
            real_examples: tensor holding the real samples.
            batch_size: number of samples per training batch.
            disc_iter: discriminator updates per generator update.
            learning_rate: learning rate for the Adam optimizers.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        # define the generator loss and optimizer
        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(
            x, tf.ones(tf.shape(x)))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(optimizer=self.generator.optimizer,
                               loss=self.generator.loss)

        # define the discriminator loss and optimizer
        self.discriminator.loss = lambda x, y: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(tf.shape(x)))
            + tf.keras.losses.MeanSquaredError()(y, -1 * tf.ones(tf.shape(y))))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(optimizer=self.discriminator.optimizer,
                                   loss=self.discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """Generate a batch of fake samples with the generator."""
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """Return a random batch of real samples."""
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """Run one training step: disc_iter discriminator updates then one
        generator update. Returns the last discriminator and generator loss."""
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample(training=True)
                real_output = self.discriminator(real_sample, training=True)
                fake_output = self.discriminator(fake_sample, training=True)
                discr_loss = self.discriminator.loss(real_output, fake_output)
            gradients = tape.gradient(
                discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(gradients, self.discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(training=True)
            gen_output = self.discriminator(fake_sample, training=False)
            gen_loss = self.generator.loss(gen_output)
        gradients = tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}

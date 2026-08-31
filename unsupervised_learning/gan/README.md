# GAN

Implementations of several Generative Adversarial Networks with
TensorFlow / Keras.

## Concepts

- **GAN** — two networks trained against each other: a *generator* that turns
  noise into fake samples and a *discriminator* (critic) that tries to tell
  real samples from fake ones. Training is a minimax game; at the optimum the
  generator produces samples indistinguishable from the real data.
- **Latent vector / latent space** — the low-dimensional random input the
  generator maps to a sample. The latent space is the domain of that input;
  moving smoothly through it produces smooth changes in the generated output.
- **Simple GAN** — the generator is pushed so the discriminator outputs `1` on
  fakes and the discriminator so it outputs `1` on real and `-1` on fake,
  both through a mean squared error.
- **Wasserstein GAN (WGAN)** — replaces the MSE objective with the
  Earth-Mover (Wasserstein) distance. The critic is not a classifier; it
  scores samples and is kept 1-Lipschitz by clipping its weights to
  `[-1, 1]`.
- **WGAN with gradient penalty (WGAN-GP)** — enforces the Lipschitz
  constraint by penalizing `(||grad|| - 1) ** 2` of the critic on points
  interpolated between real and fake batches, instead of clipping weights.
- **Convolutional generator / discriminator** — the generator upsamples a
  reshaped dense projection with `Conv2D` blocks to build an image; the
  discriminator downsamples the image with `Conv2D` / pooling blocks to a
  single score. Used here to generate small `16x16` faces.

## Files

| File | Description |
| --- | --- |
| `0-simple_gan.py` | `Simple_GAN` — MSE-based GAN. |
| `1-wgan_clip.py` | `WGAN_clip` — Wasserstein GAN with weight clipping. |
| `2-wgan_gp.py` | `WGAN_GP` — Wasserstein GAN with gradient penalty. |
| `3-generate_faces.py` | `convolutional_GenDiscr` — convolutional generator and discriminator. |
| `4-wgan_gp.py` | `WGAN_GP` with a `replace_weights` method to load pre-trained `.h5` weights. |

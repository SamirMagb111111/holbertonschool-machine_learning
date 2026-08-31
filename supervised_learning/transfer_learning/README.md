# Transfer Learning

Classify **CIFAR-10** by transferring features from an ImageNet-pretrained
network.

## Concepts

- **Transfer learning** — reuse a network trained on a large dataset
  (ImageNet) as a feature extractor for a new, smaller task instead of
  training from scratch.
- **Pretrained ImageNet model** — here `ResNet50` from `keras.applications`
  with `weights="imagenet"`, `include_top=False`.
- **Frozen layers** — the whole ResNet50 base is set `trainable = False`, so
  its ImageNet weights are kept fixed and only the new head learns.
- **Feature extraction** — the frozen base maps each image to a fixed
  embedding that the small classifier head is trained on.
- **Fine-tuning** — optionally unfreezing the top of the base and retraining
  it with a very small learning rate (not needed here to clear the bar).
- **Lambda resize layer** — CIFAR-10 images are `32x32`; the first model
  layer is a `Lambda` that bilinearly upsamples them to `224x224`, the size
  ResNet50 expects. Resizing lives in the model (not `preprocess_data`) so
  the saved model accepts raw `(32, 32, 3)` input.
- **Preprocessing** — `preprocess_data(X, Y)` applies
  `resnet50.preprocess_input` to the pixels and one-hot encodes the labels;
  the model is compiled with `categorical_crossentropy`.

## Files

| File | Description |
| --- | --- |
| `0-transfer.py` | `preprocess_data` + (run directly) trains and saves `cifar10.h5`. |

`cifar10.h5` is a generated artifact and is not committed.

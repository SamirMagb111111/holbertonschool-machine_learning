# Keras

Building, training, evaluating and serializing neural networks with
`tensorflow.keras`.

## Concepts

- **Keras** — the high-level neural-network API bundled with TensorFlow.
- **Sequential API** — stack layers in order (`0-sequential.py`); the first
  layer declares `input_shape`.
- **Functional API** — connect layers as a graph starting from `K.Input`
  (`1-input.py`); more flexible than Sequential.
- **Dense layers** — fully connected layers; here every one carries **L2
  regularization** (`kernel_regularizer=K.regularizers.l2(lambtha)`), so the
  model exposes one penalty term per Dense layer in `model.losses`.
- **Dropout** — randomly zeroes activations while training; Keras takes the
  *drop* rate, so `Dropout(1 - keep_prob)`, placed between layers only.
- **Adam** — adaptive optimizer configured with `learning_rate`, `beta_1`,
  `beta_2` (`2-optimize.py`).
- **Categorical crossentropy / accuracy** — loss and metric for one-hot
  multi-class classification.
- **One-hot encoding** — `K.utils.to_categorical` (`3-one_hot.py`).
- **`model.fit` / validation** — mini-batch training with an optional
  `validation_data` (`4-train.py`, `5-train.py`).
- **Early stopping** — `EarlyStopping(monitor="val_loss", patience=...)`,
  only with validation data (`6-train.py`).
- **Learning-rate decay** — `LearningRateScheduler` with inverse-time decay
  `alpha / (1 + decay_rate * epoch)`, stepwise per epoch, `verbose=1`
  (`7-train.py`).
- **ModelCheckpoint** — save the lowest-`val_loss` model during training
  (`8-train.py`).
- **Model save/load** — whole model to disk and back (`9-model.py`).
- **Weights save/load** — parameters only, into an existing architecture
  (`10-weights.py`).
- **JSON configuration** — architecture only, no weights (`11-config.py`).
- **Evaluation / prediction** — `evaluate` returns `[loss, accuracy]`
  (`12-test.py`); `predict` returns the full probability matrix
  (`13-predict.py`).

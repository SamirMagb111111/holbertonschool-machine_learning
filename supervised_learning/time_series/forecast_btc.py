#!/usr/bin/env python3
"""Train an RNN to forecast the next-hour BTC close price.

The preprocessed hourly archive produced by ``preprocess_data.py`` is loaded,
turned into ``(24, n_features)`` sliding windows through ``tf.data.Dataset``
pipelines and fed to a stacked GRU model trained with the mean squared error
loss.  The model is evaluated on the chronological validation split and a few
predictions are reported back in USD.
"""
import os

import numpy as np
import tensorflow as tf


np.random.seed(0)
tf.random.set_seed(0)

WINDOW = 24
BATCH_SIZE = 64
EPOCHS = 20
ARCHIVE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "btc_preprocessed.npz")


def make_dataset(data, target_index, window, batch_size, shuffle):
    """Build a ``tf.data.Dataset`` of sliding windows and next-hour targets.

    Sample ``i`` is ``data[i:i + window]`` and its target is the close price
    at row ``i + window``; no future value ever appears inside a window.
    """
    inputs = data[:-window]
    targets = data[window:, target_index]
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=inputs,
        targets=targets,
        sequence_length=window,
        sequence_stride=1,
        batch_size=None,
        shuffle=False,
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=10000, seed=0)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(window, n_features):
    """Return the compiled stacked-GRU forecasting model (MSE loss)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, n_features)),
        tf.keras.layers.GRU(64, return_sequences=True),
        tf.keras.layers.GRU(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    """Load the data, train the model and report validation results."""
    if not os.path.isfile(ARCHIVE):
        raise SystemExit(
            "Missing {}. Run ./preprocess_data.py first.".format(ARCHIVE))

    archive = np.load(ARCHIVE, allow_pickle=True)
    train_data = archive["train_data"].astype("float32")
    val_data = archive["val_data"].astype("float32")
    feature_mean = archive["feature_mean"]
    feature_std = archive["feature_std"]
    target_index = int(archive["target_index"])
    window = int(archive["window"])

    n_features = train_data.shape[1]
    train_ds = make_dataset(train_data, target_index, window,
                            BATCH_SIZE, shuffle=True)
    val_ds = make_dataset(val_data, target_index, window,
                          BATCH_SIZE, shuffle=False)

    model = build_model(window, n_features)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True)
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
              callbacks=[early_stop])

    val_mse, val_mae = model.evaluate(val_ds, verbose=0)
    print("\nvalidation MSE (scaled): {:.6f}".format(val_mse))
    print("validation MAE (scaled): {:.6f}".format(val_mae))

    t_mean = feature_mean[target_index]
    t_std = feature_std[target_index]
    val_mae_usd = val_mae * t_std
    val_rmse_usd = np.sqrt(val_mse) * t_std
    print("validation MAE  (USD):   {:.2f}".format(val_mae_usd))
    print("validation RMSE (USD):   {:.2f}".format(val_rmse_usd))

    preds = model.predict(val_ds, verbose=0).ravel()
    actual = np.concatenate([y.numpy() for _, y in val_ds]).ravel()
    preds_usd = preds * t_std + t_mean
    actual_usd = actual * t_std + t_mean

    print("\nexample next-hour forecasts:")
    for i in range(0, len(preds_usd), max(1, len(preds_usd) // 8))[:8]:
        print("  predicted close: ${:>10.2f}   actual close: ${:>10.2f}"
              .format(preds_usd[i], actual_usd[i]))


if __name__ == "__main__":
    main()

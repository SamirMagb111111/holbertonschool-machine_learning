#!/usr/bin/env python3
"""Transfer learning on CIFAR-10 with a frozen ResNet50 base."""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """Pre-process CIFAR-10 data for the ResNet50 application.

    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3) with CIFAR-10 images.
        Y: numpy.ndarray of shape (m,) or (m, 1) with the labels.

    Returns:
        X_p: the images run through ``resnet50.preprocess_input``.
        Y_p: the labels one-hot encoded over 10 classes.
    """
    X_p = K.applications.resnet50.preprocess_input(X.astype("float32"))
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


def build_model():
    """Return the compiled CIFAR-10 model (Lambda resize + ResNet50 + head)."""
    base = K.applications.ResNet50(weights="imagenet", include_top=False,
                                   input_shape=(224, 224, 3))
    base.trainable = False

    model = K.Sequential([
        K.layers.Lambda(
            lambda img: K.backend.resize_images(
                img, 7, 7, "channels_last", interpolation="bilinear"),
            input_shape=(32, 32, 3), output_shape=(224, 224, 3)),
        base,
        K.layers.GlobalAveragePooling2D(),
        K.layers.BatchNormalization(),
        K.layers.Dropout(0.3),
        K.layers.Dense(256, activation="relu"),
        K.layers.Dropout(0.3),
        K.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=K.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    model = build_model()
    callbacks = [
        K.callbacks.EarlyStopping(monitor="val_accuracy", patience=3,
                                  restore_best_weights=True),
        K.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.2,
                                      patience=1),
    ]
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
              batch_size=128, epochs=10, callbacks=callbacks)

    model.save("cifar10.h5")

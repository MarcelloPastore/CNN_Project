import keras
from keras import layers


def build_cnn_paper_t23(input_shape=(2, 128, 1), num_classes=10, dropout=0.5):
    """
    CNN paper-like (T23 style), backend-agnostic Keras.
    Usabile con JAX impostando:
        os.environ["KERAS_BACKEND"] = "jax"

    Architettura target (come da tabella paper):
    - Conv1: 256 filtri, kernel (1,3)
    - Conv2: 256 filtri, kernel (2,3)
    - Conv3: 80  filtri, kernel (1,3)
    - Conv4: 80  filtri, kernel (1,3)
    - Dense: 128
    - Output: num_classes (paper: 10)
    """
    inp = keras.Input(shape=input_shape, name="iq_input")

    # blocco 1
    x = layers.ZeroPadding2D((0, 2), name="pad1")(inp)
    x = layers.Conv2D(256, (1, 3), activation="relu", name="conv1")(x)
    x = layers.Dropout(dropout, name="drop1")(x)

    # blocco 2
    x = layers.ZeroPadding2D((0, 2), name="pad2")(x)
    x = layers.Conv2D(256, (2, 3), activation="relu", name="conv2")(x)
    x = layers.Dropout(dropout, name="drop2")(x)

    # blocco 3
    x = layers.ZeroPadding2D((0, 2), name="pad3")(x)
    x = layers.Conv2D(80, (1, 3), activation="relu", name="conv3")(x)
    x = layers.Dropout(dropout, name="drop3")(x)

    # blocco 4
    x = layers.ZeroPadding2D((0, 2), name="pad4")(x)
    x = layers.Conv2D(80, (1, 3), activation="relu", name="conv4")(x)
    x = layers.Dropout(dropout, name="drop4")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout, name="drop5")(x)
    out = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    return keras.Model(inp, out, name="cnn_paper_t23")
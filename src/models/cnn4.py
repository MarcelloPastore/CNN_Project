import keras
from keras import layers


def build_cnn4_v3(input_shape=(2, 128, 1), num_classes=11, dropout=0.45):
    """
    CNN4 v3:
    - backbone simile a v2
    - head più leggera (Dense 128) per ridurre overfit
    - BN + Dropout per stabilità
    """
    inp = keras.Input(shape=input_shape, name="iq_input")

    # Block 1
    x = layers.ZeroPadding2D((0, 2), name="pad1")(inp)
    x = layers.Conv2D(64, (1, 8), activation="relu", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(dropout, name="drop1")(x)

    # Block 2
    x = layers.ZeroPadding2D((0, 2), name="pad2")(x)
    x = layers.Conv2D(64, (2, 8), activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Dropout(dropout, name="drop2")(x)

    # Block 3
    x = layers.ZeroPadding2D((0, 1), name="pad3")(x)
    x = layers.Conv2D(128, (1, 4), activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Dropout(dropout, name="drop3")(x)

    # Block 4
    x = layers.ZeroPadding2D((0, 1), name="pad4")(x)
    x = layers.Conv2D(128, (1, 4), activation="relu", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Dropout(dropout, name="drop4")(x)

    # Head (lighter)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout, name="drop_dense")(x)
    out = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    return keras.Model(inp, out, name="cnn4_v3")
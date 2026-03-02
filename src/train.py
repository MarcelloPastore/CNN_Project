import os
os.environ["KERAS_BACKEND"] = "jax"

import pickle
import numpy as np
import keras
from keras import layers
from collections import defaultdict

DATA_PATH = "data/RML2016.pkl"
MODEL_OUT = "models/cnn2_model.keras"
SEED = 42
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3

np.random.seed(SEED)

# -------- Load data --------
with open(DATA_PATH, "rb") as f:
    Xd = pickle.load(f, encoding="latin1")

mods = sorted(list(set(k[0] for k in Xd.keys())))
snrs = sorted(list(set(k[1] for k in Xd.keys())))
mod_to_idx = {m: i for i, m in enumerate(mods)}

X_list, y_mod, y_snr = [], [], []
for m in mods:
    for s in snrs:
        arr = Xd[(m, s)].astype("float32")
        X_list.append(arr)
        y_mod.extend([m] * arr.shape[0])
        y_snr.extend([s] * arr.shape[0])

X = np.vstack(X_list)                      # (N,2,128)
y_mod = np.array(y_mod)
y_snr = np.array(y_snr)
y_idx = np.array([mod_to_idx[m] for m in y_mod], dtype=np.int32)
y_cat = keras.utils.to_categorical(y_idx, num_classes=len(mods))
X = X[..., np.newaxis]                     # (N,2,128,1)

# -------- Split stratified by (mod,snr): 50/50 then val from train --------
indices_by_key = defaultdict(list)
for i, (m, s) in enumerate(zip(y_mod, y_snr)):
    indices_by_key[(m, s)].append(i)

rng = np.random.default_rng(SEED)
train_idx, test_idx = [], []
for _, idxs in indices_by_key.items():
    idxs = np.array(idxs)
    rng.shuffle(idxs)
    n_train = int(len(idxs) * 0.5)
    train_idx.extend(idxs[:n_train].tolist())
    test_idx.extend(idxs[n_train:].tolist())

train_idx = np.array(train_idx, dtype=np.int32)
test_idx = np.array(test_idx, dtype=np.int32)

rng.shuffle(train_idx)
n_val = int(len(train_idx) * 0.1)
val_idx = train_idx[:n_val]
train_idx2 = train_idx[n_val:]

X_train, y_train = X[train_idx2], y_cat[train_idx2]
X_val, y_val = X[val_idx], y_cat[val_idx]
X_test, y_test = X[test_idx], y_cat[test_idx]

print("X_train:", X_train.shape, "X_val:", X_val.shape, "X_test:", X_test.shape)

# -------- Model (CNN2-like) --------
inp = keras.Input(shape=(2, 128, 1))
x = layers.ZeroPadding2D((0, 2))(inp)
x = layers.Conv2D(256, (1, 8), activation="relu")(x)
x = layers.Dropout(0.5)(x)

x = layers.ZeroPadding2D((0, 2))(x)
x = layers.Conv2D(80, (2, 8), activation="relu")(x)
x = layers.Dropout(0.5)(x)

x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(len(mods), activation="softmax")(x)

model = keras.Model(inp, out)
model.compile(
    optimizer=keras.optimizers.Adamax(learning_rate=LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-6
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=8, restore_best_weights=True
    ),
]

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"[RESULT] Test accuracy: {acc:.4f} | Test loss: {loss:.4f}")

model.save(MODEL_OUT)
print(f"Saved model -> {MODEL_OUT}")
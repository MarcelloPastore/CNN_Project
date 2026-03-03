import os
os.environ["KERAS_BACKEND"] = "jax"

import pickle
import numpy as np
import keras
from collections import defaultdict
from models.cnn4_v3 import build_cnn4_v3

DATA_PATH = "data/RML2016.pkl"
MODEL_OUT = "models/cnn4_v3_model.keras"

SEED = 42
BATCH_SIZE = 16
EPOCHS = 45
LR = 8e-4
VAL_FRAC_FROM_TRAIN = 0.1

np.random.seed(SEED)

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

X = np.vstack(X_list)
y_mod = np.array(y_mod)
y_snr = np.array(y_snr)

y_idx = np.array([mod_to_idx[m] for m in y_mod], dtype=np.int32)
y_cat = keras.utils.to_categorical(y_idx, num_classes=len(mods))
X = X[..., np.newaxis]

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
rng.shuffle(train_idx)

n_val = int(len(train_idx) * VAL_FRAC_FROM_TRAIN)
val_idx = train_idx[:n_val]
train_idx2 = train_idx[n_val:]

X_train, y_train = X[train_idx2], y_cat[train_idx2]
X_val, y_val = X[val_idx], y_cat[val_idx]
X_test, y_test = X[np.array(test_idx, dtype=np.int32)], y_cat[np.array(test_idx, dtype=np.int32)]

model = build_cnn4_v3(input_shape=(2, 128, 1), num_classes=len(mods), dropout=0.45)

loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss=loss_fn,
    metrics=["accuracy"],
)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1
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
print(f"[CNN4 v3] Test accuracy: {acc:.4f} | Test loss: {loss:.4f}")

os.makedirs("models", exist_ok=True)
model.save(MODEL_OUT)
print(f"Saved -> {MODEL_OUT}")
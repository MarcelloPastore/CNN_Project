import os
os.environ["KERAS_BACKEND"] = "jax"

import pickle
import numpy as np
import keras
from collections import defaultdict
from models.cnn_paper_t23 import build_cnn_paper_t23

DATA_PATH = "data/RML2016.10b.dat"
MODEL_OUT = "models/cnn_paper_t23_jax.keras"

SEED = 42
BATCH_SIZE = 1024   # paper testava 512/1024/2048; parti da 1024
EPOCHS = 60
LR = 1e-3
VAL_FRAC_FROM_TRAIN = 0.1
DROPOUT = 0.5

np.random.seed(SEED)

with open(DATA_PATH, "rb") as f:
    Xd = pickle.load(f, encoding="latin1")

mods = sorted(list(set(k[0] for k in Xd.keys())))
snrs = sorted(list(set(k[1] for k in Xd.keys())))

# Paper setup mostrato: 10 classi.
# Se il tuo pickle ha 11 classi (es. include AM-SSB), filtrale qui se vuoi replica fedele.
TARGET_MODS = mods[:10] if len(mods) >= 10 else mods
mod_to_idx = {m: i for i, m in enumerate(TARGET_MODS)}

X_list, y_mod, y_snr = [], [], []
for m in TARGET_MODS:
    for s in snrs:
        arr = Xd[(m, s)].astype("float32")
        X_list.append(arr)
        y_mod.extend([m] * arr.shape[0])
        y_snr.extend([s] * arr.shape[0])

X = np.vstack(X_list)            # (N,2,128)
y_mod = np.array(y_mod)
y_snr = np.array(y_snr)

y_idx = np.array([mod_to_idx[m] for m in y_mod], dtype=np.int32)
y_cat = keras.utils.to_categorical(y_idx, num_classes=len(TARGET_MODS))
X = X[..., np.newaxis]           # (N,2,128,1)

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
n_val = int(len(train_idx) * VAL_FRAC_FROM_TRAIN)
val_idx = train_idx[:n_val]
train_idx2 = train_idx[n_val:]

X_train, y_train = X[train_idx2], y_cat[train_idx2]
X_val, y_val = X[val_idx], y_cat[val_idx]
X_test, y_test = X[test_idx], y_cat[test_idx]

model = build_cnn_paper_t23(
    input_shape=(2, 128, 1),
    num_classes=len(TARGET_MODS),
    dropout=DROPOUT
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-6),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
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
print(f"[PAPER_T23_JAX] Test accuracy: {acc:.4f} | loss: {loss:.4f}")

os.makedirs("models", exist_ok=True)
model.save(MODEL_OUT)
print(f"Saved -> {MODEL_OUT}")
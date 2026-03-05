import os
# --- JAX backend / VRAM controls (set BEFORE importing keras) ---
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

import time
import json
import pickle
from collections import defaultdict

import numpy as np
import keras
from keras import layers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# mixed precision (JAX + Keras3)
keras.mixed_precision.set_global_policy("mixed_float16")

# -----------------------
# Config (CNN4 v3 - JAX optimized)
# -----------------------
DATA_PATH = "data/RML2016.10b.dat"
MODEL_OUT = "models/cnn4_v3_10c.keras"
TRAIN_METRICS_OUT = "outputs/tables/cnn4_v3_train_metrics.json"

SEED = 42
NUM_CLASSES = 10
BATCH_SIZE = 1024
EPOCHS = 45
LR = 8e-4
PATIENCE_ES = 8
PATIENCE_LR = 5
VAL_SPLIT = 0.2

np.random.seed(SEED)


def load_rml(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def build_stratified_splits(Xd, seed=42, num_classes=10, train_ratio=0.5):
    mods_all = sorted(list(set(k[0] for k in Xd.keys())))
    snrs = sorted(list(set(k[1] for k in Xd.keys())))
    target_mods = mods_all[:num_classes]
    mod_to_idx = {m: i for i, m in enumerate(target_mods)}

    X_list, y_mod, y_snr = [], [], []
    for m in target_mods:
        for s in snrs:
            arr = Xd[(m, s)].astype("float32")
            X_list.append(arr)
            y_mod.extend([m] * arr.shape[0])
            y_snr.extend([s] * arr.shape[0])

    X = np.vstack(X_list).astype("float32")[..., np.newaxis]  # (N,2,128,1)
    y_mod = np.array(y_mod)
    y_snr = np.array(y_snr)

    indices_by_key = defaultdict(list)
    for i, (m, s) in enumerate(zip(y_mod, y_snr)):
        indices_by_key[(m, s)].append(i)

    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for _, idxs in indices_by_key.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_train = int(len(idxs) * train_ratio)
        train_idx.extend(idxs[:n_train].tolist())
        test_idx.extend(idxs[n_train:].tolist())

    train_idx = np.array(train_idx, dtype=np.int32)
    test_idx = np.array(test_idx, dtype=np.int32)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train = X[train_idx]
    y_train = np.array([mod_to_idx[m] for m in y_mod[train_idx]], dtype=np.int32)
    X_test = X[test_idx]
    y_test = np.array([mod_to_idx[m] for m in y_mod[test_idx]], dtype=np.int32)

    return X_train, y_train, X_test, y_test, target_mods


def split_train_val_stratified(X_train, y_train, val_split=0.2, seed=42, num_classes=10):
    rng = np.random.default_rng(seed)
    train_sub_idx, val_sub_idx = [], []

    for c in range(num_classes):
        idx = np.where(y_train == c)[0]
        rng.shuffle(idx)
        n_val = int(len(idx) * val_split)
        val_sub_idx.extend(idx[:n_val].tolist())
        train_sub_idx.extend(idx[n_val:].tolist())

    train_sub_idx = np.array(train_sub_idx, dtype=np.int32)
    val_sub_idx = np.array(val_sub_idx, dtype=np.int32)
    rng.shuffle(train_sub_idx)
    rng.shuffle(val_sub_idx)

    X_tr, y_tr = X_train[train_sub_idx], y_train[train_sub_idx]
    X_val, y_val = X_train[val_sub_idx], y_train[val_sub_idx]
    return X_tr, y_tr, X_val, y_val


def build_cnn4_v3(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (1, 3), padding="same", activation="relu")(inp)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(128, (2, 3), padding="same", activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(128, (1, 3), padding="same", activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(256, (1, 3), padding="same", activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # VRAM-friendly head (instead of Flatten huge tensor)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # keep output in float32 for numeric stability
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    return keras.Model(inp, out, name="cnn4_v3")


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)

    Xd = load_rml(DATA_PATH)
    X_train, y_train, X_test, y_test, target_mods = build_stratified_splits(
        Xd, seed=SEED, num_classes=NUM_CLASSES, train_ratio=0.5
    )
    X_tr, y_tr, X_val, y_val = split_train_val_stratified(
        X_train, y_train, val_split=VAL_SPLIT, seed=SEED, num_classes=NUM_CLASSES
    )

    model = build_cnn4_v3(X_train.shape[1:], NUM_CLASSES)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    cbs = [
        ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=PATIENCE_LR, min_lr=1e-5, verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy", patience=PATIENCE_ES, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    t0 = time.time()
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        verbose=1,
        shuffle=True,
    )
    train_time_sec = time.time() - t0

    best_model = keras.models.load_model(MODEL_OUT)
    test_loss, test_acc = best_model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)

    with open(TRAIN_METRICS_OUT, "w") as f:
        json.dump({
            "model_name": "cnn4_v3",
            "batch_size": BATCH_SIZE,
            "best_val_accuracy": float(np.max(hist.history["val_accuracy"])),
            "final_test_accuracy": float(test_acc),
            "final_test_loss": float(test_loss),
            "epochs_ran": len(hist.history["loss"]),
            "train_time_sec": float(train_time_sec),
            "target_mods": target_mods
        }, f, indent=2)

    print(f"[CNN4_V3] Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")
    print(f"Saved -> {MODEL_OUT}")
    print(f"Saved -> {TRAIN_METRICS_OUT}")


if __name__ == "__main__":
    main()
import os
os.environ["KERAS_BACKEND"] = "jax"

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

DATA_PATH = "data/RML2016.pkl"
MODEL_PATH = "models/cnn2_model.keras"
BATCH_SIZE = 64
SEED = 42

np.random.seed(SEED)

# -------- Load data --------
with open(DATA_PATH, "rb") as f:
    Xd = pickle.load(f, encoding="latin1")

mods = sorted(list(set(k[0] for k in Xd.keys())))
snrs = sorted(list(set(k[1] for k in Xd.keys())))
mod_to_idx = {m: i for i, m in enumerate(mods)}
idx_to_mod = {i: m for m, i in mod_to_idx.items()}

X_list, y_mod, y_snr = [], [], []
for m in mods:
    for s in snrs:
        arr = Xd[(m, s)].astype("float32")
        X_list.append(arr)
        y_mod.extend([m] * arr.shape[0])
        y_snr.extend([s] * arr.shape[0])

X = np.vstack(X_list).astype("float32")
y_mod = np.array(y_mod)
y_snr = np.array(y_snr)
X = X[..., np.newaxis]

# same split logic as train.py
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

test_idx = np.array(test_idx, dtype=np.int32)
X_test = X[test_idx]
y_mod_test = y_mod[test_idx]
y_snr_test = y_snr[test_idx]

# -------- Predict --------
model = keras.models.load_model(MODEL_PATH)
probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
pred_idx = np.argmax(probs, axis=1)
pred_mod = np.array([idx_to_mod[int(i)] for i in pred_idx])

acc = np.mean(pred_mod == y_mod_test)
print(f"Test accuracy: {acc:.4f}")

# Accuracy per SNR
print("\nAccuracy per SNR:")
for s in sorted(np.unique(y_snr_test)):
    m = (y_snr_test == s)
    a = np.mean(pred_mod[m] == y_mod_test[m])
    print(f"SNR {int(s):>3}: {a:.4f}")

# Confusion matrix normalized
labels = mods
cm = confusion_matrix(y_mod_test, pred_mod, labels=labels)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True, values_format=".2f")
plt.title("Confusion Matrix (normalized)")
plt.tight_layout()
plt.savefig("outputs/figures/confusion_matrix_normalized.png", dpi=200)
print("Saved -> outputs/figures/confusion_matrix_normalized.png")

# save snr table
snr_rows = []
for s in sorted(np.unique(y_snr_test)):
    m = (y_snr_test == s)
    snr_rows.append({"snr_db": int(s), "accuracy": float(np.mean(pred_mod[m] == y_mod_test[m]))})
pd.DataFrame(snr_rows).to_csv("outputs/tables/accuracy_by_snr.csv", index=False)
print("Saved -> outputs/tables/accuracy_by_snr.csv")
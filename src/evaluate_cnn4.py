import os
os.environ["KERAS_BACKEND"] = "jax"  # se usi TF backend, metti "tensorflow"

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------
# Config
# -----------------------
DATA_PATH = "data/RML2016.pkl"
MODEL_PATH = "models/cnn4_v3_model.keras"
FIG_OUT = "outputs/figures/cnn4_v3_confusion_matrix_normalized.png"
SNR_TABLE_OUT = "outputs/tables/cnn4_v3_accuracy_by_snr.csv"

BATCH_SIZE = 64
SEED = 42
np.random.seed(SEED)

# -----------------------
# Load data
# -----------------------
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

X = np.vstack(X_list).astype("float32")   # (N,2,128)
y_mod = np.array(y_mod)
y_snr = np.array(y_snr)
X = X[..., np.newaxis]                    # (N,2,128,1)

# -----------------------
# Same split logic as training (stratified by mod+snr)
# -----------------------
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

# -----------------------
# Predict
# -----------------------
model = keras.models.load_model(MODEL_PATH)
probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)

pred_idx = np.argmax(probs, axis=1)
pred_mod = np.array([idx_to_mod[int(i)] for i in pred_idx])

acc = np.mean(pred_mod == y_mod_test)
print(f"[CNN4] Test accuracy: {acc:.4f}")

# -----------------------
# Accuracy per SNR
# -----------------------
rows = []
print("\n[CNN4] Accuracy per SNR:")
for s in sorted(np.unique(y_snr_test)):
    m = (y_snr_test == s)
    a = float(np.mean(pred_mod[m] == y_mod_test[m]))
    rows.append({"snr_db": int(s), "accuracy": a})
    print(f"SNR {int(s):>3}: {a:.4f}")

os.makedirs("outputs/tables", exist_ok=True)
pd.DataFrame(rows).to_csv(SNR_TABLE_OUT, index=False)
print(f"Saved -> {SNR_TABLE_OUT}")

# -----------------------
# Normalized confusion matrix
# -----------------------
cm = confusion_matrix(y_mod_test, pred_mod, labels=mods)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

os.makedirs("outputs/figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=mods)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True, values_format=".2f")
plt.title("CNN4 - Confusion Matrix (normalized)")
plt.tight_layout()
plt.savefig(FIG_OUT, dpi=200)
print(f"Saved -> {FIG_OUT}")
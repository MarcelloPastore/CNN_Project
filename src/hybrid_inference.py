import os
os.environ["KERAS_BACKEND"] = "jax"  # oppure "tensorflow"

import pickle
import numpy as np
import keras
from collections import defaultdict

# -----------------------
# Config
# -----------------------
DATA_PATH = "data/RML2016.pkl"
CNN2_MODEL_PATH = "models/cnn2_model.keras"
CNN4_MODEL_PATH = "models/cnn4_v3_model.keras"  # o cnn4_model.keras
SEED = 42
BATCH_SIZE = 128

# Soglia selector:
# se SNR <= THRESHOLD => CNN2, altrimenti CNN4
SNR_THRESHOLD_DB = -2

# -----------------------
# Load dataset
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

X = np.vstack(X_list)              # (N,2,128)
y_mod = np.array(y_mod)
y_snr = np.array(y_snr)
X = X[..., np.newaxis]             # (N,2,128,1)

# -----------------------
# Rebuild same test split used in training/eval
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
# Load models
# -----------------------
cnn2 = keras.models.load_model(CNN2_MODEL_PATH)
cnn4 = keras.models.load_model(CNN4_MODEL_PATH)

# -----------------------
# Predict both once
# -----------------------
p2 = cnn2.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
p4 = cnn4.predict(X_test, batch_size=BATCH_SIZE, verbose=1)

pred2_idx = np.argmax(p2, axis=1)
pred4_idx = np.argmax(p4, axis=1)

# -----------------------
# Hybrid selector by SNR
# -----------------------
use_cnn2 = (y_snr_test <= SNR_THRESHOLD_DB)
hyb_idx = np.where(use_cnn2, pred2_idx, pred4_idx)

pred2_mod = np.array([idx_to_mod[int(i)] for i in pred2_idx])
pred4_mod = np.array([idx_to_mod[int(i)] for i in pred4_idx])
hyb_mod = np.array([idx_to_mod[int(i)] for i in hyb_idx])

# -----------------------
# Global accuracies
# -----------------------
acc2 = np.mean(pred2_mod == y_mod_test)
acc4 = np.mean(pred4_mod == y_mod_test)
acch = np.mean(hyb_mod == y_mod_test)

print(f"\nCNN2 accuracy   : {acc2:.4f}")
print(f"CNN4 accuracy   : {acc4:.4f}")
print(f"HYBRID accuracy : {acch:.4f} (threshold {SNR_THRESHOLD_DB} dB)")

# -----------------------
# Accuracy by SNR
# -----------------------
print("\nAccuracy by SNR")
print("SNR   CNN2    CNN4    HYBRID")
for s in sorted(np.unique(y_snr_test)):
    m = (y_snr_test == s)
    a2 = np.mean(pred2_mod[m] == y_mod_test[m])
    a4 = np.mean(pred4_mod[m] == y_mod_test[m])
    ah = np.mean(hyb_mod[m] == y_mod_test[m])
    print(f"{int(s):>3}  {a2:>6.4f}  {a4:>6.4f}  {ah:>7.4f}")
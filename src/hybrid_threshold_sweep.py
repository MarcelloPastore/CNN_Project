import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   # evita prealloc aggressiva

import gc
import pickle
import numpy as np
import keras
from collections import defaultdict

DATA_PATH = "data/RML2016.pkl"
CNN2_MODEL_PATH = "models/cnn2_model.keras"
CNN4_MODEL_PATH = "models/cnn4_model.keras"
SEED = 42
BATCH_SIZE = 32   # << ridotto

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

X = np.vstack(X_list)[..., np.newaxis]
y_mod = np.array(y_mod)
y_snr = np.array(y_snr)

indices_by_key = defaultdict(list)
for i, (m, s) in enumerate(zip(y_mod, y_snr)):
    indices_by_key[(m, s)].append(i)

rng = np.random.default_rng(SEED)
test_idx = []
for _, idxs in indices_by_key.items():
    idxs = np.array(idxs)
    rng.shuffle(idxs)
    n_train = int(len(idxs) * 0.5)
    test_idx.extend(idxs[n_train:].tolist())

test_idx = np.array(test_idx, dtype=np.int32)
X_test = X[test_idx]
y_mod_test = y_mod[test_idx]
y_snr_test = y_snr[test_idx]

# predict CNN2
cnn2 = keras.models.load_model(CNN2_MODEL_PATH)
p2 = cnn2.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
pred2_idx = np.argmax(p2, axis=1)
del cnn2, p2
gc.collect()

# predict CNN4 (dopo cleanup)
cnn4 = keras.models.load_model(CNN4_MODEL_PATH)
p4 = cnn4.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
pred4_idx = np.argmax(p4, axis=1)
del cnn4, p4
gc.collect()

pred2_mod = np.array([idx_to_mod[int(i)] for i in pred2_idx])
pred4_mod = np.array([idx_to_mod[int(i)] for i in pred4_idx])

thresholds = [-8, -6, -4, -2, 0, 2, 4]
best_t, best_acc = None, -1.0

for t in thresholds:
    use_cnn2 = (y_snr_test <= t)
    hyb_idx = np.where(use_cnn2, pred2_idx, pred4_idx)
    hyb_mod = np.array([idx_to_mod[int(i)] for i in hyb_idx])
    acc = np.mean(hyb_mod == y_mod_test)
    print(f"threshold={t:>3} dB -> hybrid_acc={acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_t = t

print(f"\nBest threshold: {best_t} dB | accuracy={best_acc:.4f}")
import os
os.environ["KERAS_BACKEND"] = "jax"

import json
import pickle
import numpy as np
import pandas as pd
import keras
from collections import defaultdict

DATA_PATH = "data/RML2016.pkl"
MODEL_PATH = "models/cnn2_model.keras"
OUT_PATH = "outputs/tables/predictions.parquet"
BATCH_SIZE = 64
SEED = 42

np.random.seed(SEED)

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

# same split as train/eval
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

model = keras.models.load_model(MODEL_PATH)

# embedding = ultimo Dense relu
embedding_layer = None
for layer in model.layers[::-1]:
    if isinstance(layer, keras.layers.Dense) and layer.activation.__name__ == "relu":
        embedding_layer = layer
        break
if embedding_layer is None:
    raise RuntimeError("Dense relu layer not found for embedding.")

embedding_model = keras.Model(inputs=model.input, outputs=embedding_layer.output)

probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
emb = embedding_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)

pred_idx = np.argmax(probs, axis=1)
pred_mod = np.array([idx_to_mod[int(i)] for i in pred_idx])
conf = probs[np.arange(len(probs)), pred_idx]

top2_idx = np.argsort(-probs, axis=1)[:, :2]
top2_mod = np.array([idx_to_mod[int(i)] for i in top2_idx[:, 1]])
top2_conf = probs[np.arange(len(probs)), top2_idx[:, 1]]

rows = []
for i in range(len(X_test)):
    rows.append({
        "sample_id": int(i),
        "true_mod": str(y_mod_test[i]),
        "snr_db": int(y_snr_test[i]),
        "pred_mod": str(pred_mod[i]),
        "is_correct": bool(pred_mod[i] == y_mod_test[i]),
        "confidence": float(conf[i]),
        "top2_mod": str(top2_mod[i]),
        "top2_confidence": float(top2_conf[i]),
        "probs_json": json.dumps(probs[i].tolist()),
        "embedding_json": json.dumps(emb[i].tolist()),
    })

df = pd.DataFrame(rows)
df.to_parquet(OUT_PATH, index=False)
print(f"Saved -> {OUT_PATH} ({len(df)} rows)")
import os
os.environ["KERAS_BACKEND"] = "jax"   # << importante: prima di import keras

import json
import contextlib
import keras

MODEL_PATH = "models/cnn4_model.keras"
OUT_DIR = "outputs/model_reports"
BASENAME = os.path.splitext(os.path.basename(MODEL_PATH))[0]

os.makedirs(OUT_DIR, exist_ok=True)

model = keras.models.load_model(MODEL_PATH)

summary_path = os.path.join(OUT_DIR, f"{BASENAME}_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    with contextlib.redirect_stdout(f):
        model.summary()

json_path = os.path.join(OUT_DIR, f"{BASENAME}_architecture.json")
with open(json_path, "w", encoding="utf-8") as f:
    f.write(model.to_json())

config_path = os.path.join(OUT_DIR, f"{BASENAME}_config_pretty.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(model.get_config(), f, indent=2, ensure_ascii=False)

print("Export completato:")
print(summary_path)
print(json_path)
print(config_path)
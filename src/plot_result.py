import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = "tensorflow"

import json
import pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import keras

# -----------------------
# Paths
# -----------------------
DATA_PATH = "data/RML2016.10b.dat"

MODEL_PATHS = {
    "CNN2": "models/cnn2_10c.keras",
    "T23": "models/cnn_paper_t23_10c.keras",
    "CNN4_v3": "models/cnn4_v3_10c.keras",
}

METRICS_PATHS = {
    "CNN2": "outputs/tables/cnn2_train_metrics.json",
    "T23": "outputs/tables/cnn_paper_t23_train_metrics.json",
    "CNN4_v3": "outputs/tables/cnn4_v3_train_metrics.json",
}

OUT_DIR = "outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
NUM_CLASSES = 10
PRED_BATCH = 256  # più basso per evitare warning OOM durante predict


def load_rml(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def build_test_set_with_snr(Xd, seed=42, num_classes=10, train_ratio=0.5):
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
    test_idx = []
    for _, idxs in indices_by_key.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_train = int(len(idxs) * train_ratio)
        test_idx.extend(idxs[n_train:].tolist())

    test_idx = np.array(test_idx, dtype=np.int32)
    rng.shuffle(test_idx)

    X_test = X[test_idx]
    y_test = np.array([mod_to_idx[m] for m in y_mod[test_idx]], dtype=np.int32)
    snr_test = y_snr[test_idx]

    return X_test, y_test, snr_test, target_mods, snrs


def load_metrics(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def plot_confmat(cm, class_names, title, out_path):
    cm = cm.astype(np.float64)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot=False,   # senza numeri
        square=True
    )
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def accuracy_per_snr(y_true, y_pred, snr_values, snr_grid):
    out = []
    for s in snr_grid:
        idx = np.where(snr_values == s)[0]
        if len(idx) == 0:
            out.append(np.nan)
        else:
            out.append(float(np.mean(y_true[idx] == y_pred[idx])))
    return out


def main():
    Xd = load_rml(DATA_PATH)
    X_test, y_test, snr_test, class_names, snr_grid = build_test_set_with_snr(
        Xd, seed=SEED, num_classes=NUM_CLASSES
    )

    metrics = {k: load_metrics(v) for k, v in METRICS_PATHS.items()}

    names = []
    test_acc = []
    train_time = []
    epochs_ran = []
    snr_curves = {}

    for model_name in ["CNN2", "T23", "CNN4_v3"]:
        model_path = MODEL_PATHS[model_name]
        if not os.path.exists(model_path):
            print(f"[WARN] Model non trovato: {model_path}")
            continue

        print(f"[INFO] Loading model: {model_path}")
        model = keras.models.load_model(model_path, compile=False)

        print(f"[INFO] Predicting: {model_name}")
        y_prob = model.predict(X_test, batch_size=PRED_BATCH, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        # confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(NUM_CLASSES))
        out_cm = {
            "CNN2": f"{OUT_DIR}/confmat_cnn2.png",
            "T23": f"{OUT_DIR}/confmat_t23.png",
            "CNN4_v3": f"{OUT_DIR}/confmat_cnn4v3.png",
        }[model_name]
        plot_confmat(cm, class_names, f"Confusion Matrix - {model_name}", out_cm)
        print(f"[OK] Saved {out_cm}")

        # curve SNR
        snr_curves[model_name] = accuracy_per_snr(y_test, y_pred, snr_test, snr_grid)

        # metriche aggregate
        m = metrics.get(model_name, {})
        names.append(model_name)
        test_acc.append(float(m.get("final_test_accuracy", np.mean(y_pred == y_test))))
        train_time.append(float(m.get("train_time_sec", np.nan)))
        epochs_ran.append(float(m.get("epochs_ran", np.nan)))

        del model

    # accuracy bar
    if names:
        plt.figure(figsize=(7, 4.5))
        bars = plt.bar(names, test_acc)
        plt.ylabel("Test Accuracy")
        plt.ylim(0, 1.0)
        plt.title("Comparison of Test Accuracy")
        for b, v in zip(bars, test_acc):
            plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.3f}", ha="center")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/compare_accuracy.png", dpi=300)
        plt.close()
        print(f"[OK] Saved {OUT_DIR}/compare_accuracy.png")

    # SNR lines (TUTTE E 3)
    if snr_curves:
        plt.figure(figsize=(8.5, 5.2))
        style = {
            "CNN2": dict(marker="o", linewidth=2),
            "T23": dict(marker="s", linewidth=2),
            "CNN4_v3": dict(marker="^", linewidth=2),
        }
        for model_name in ["CNN2", "T23", "CNN4_v3"]:
            if model_name in snr_curves:
                plt.plot(snr_grid, snr_curves[model_name], label=model_name, **style[model_name])

        plt.xlabel("SNR (dB)")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.0)
        plt.title("Accuracy vs SNR")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/compare_accuracy_vs_snr.png", dpi=300)
        plt.close()
        print(f"[OK] Saved {OUT_DIR}/compare_accuracy_vs_snr.png")

    # final comparison
    if names:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        axes[0].bar(names, test_acc);   axes[0].set_title("Test Accuracy");      axes[0].set_ylim(0, 1.0)
        axes[1].bar(names, train_time); axes[1].set_title("Training Time (s)")
        axes[2].bar(names, epochs_ran); axes[2].set_title("Epochs Ran")
        for ax in axes:
            ax.tick_params(axis="x", rotation=20)
        fig.suptitle("Final Comparison of the 3 Networks", fontsize=13)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/final_comparison.png", dpi=300)
        plt.close()
        print(f"[OK] Saved {OUT_DIR}/final_comparison.png")

    print("\nDone.")
    print(f"Generated files in: {OUT_DIR}")


if __name__ == "__main__":
    main()
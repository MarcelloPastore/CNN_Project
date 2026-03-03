import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
CNN2_CSV = "outputs/tables/accuracy_by_snr.csv"
CNN4_CSV = "outputs/tables/cnn4_accuracy_by_snr.csv"

OUT_CSV = "outputs/tables/compare_cnn2_cnn4_by_snr.csv"
OUT_PNG = "outputs/figures/compare_cnn2_cnn4_by_snr.png"

# -----------------------
# Load
# -----------------------
df2 = pd.read_csv(CNN2_CSV).copy()
df4 = pd.read_csv(CNN4_CSV).copy()

# normalizza nomi colonne attesi: snr_db, accuracy
required = {"snr_db", "accuracy"}
if not required.issubset(df2.columns):
    raise ValueError(f"{CNN2_CSV} deve contenere colonne {required}, trovate: {set(df2.columns)}")
if not required.issubset(df4.columns):
    raise ValueError(f"{CNN4_CSV} deve contenere colonne {required}, trovate: {set(df4.columns)}")

df2 = df2.rename(columns={"accuracy": "cnn2_accuracy"})
df4 = df4.rename(columns={"accuracy": "cnn4_accuracy"})

# -----------------------
# Merge + delta
# -----------------------
cmp_df = pd.merge(df2[["snr_db", "cnn2_accuracy"]],
                  df4[["snr_db", "cnn4_accuracy"]],
                  on="snr_db",
                  how="inner").sort_values("snr_db")

cmp_df["delta_cnn4_minus_cnn2"] = cmp_df["cnn4_accuracy"] - cmp_df["cnn2_accuracy"]

# -----------------------
# Save table
# -----------------------
os.makedirs("outputs/tables", exist_ok=True)
cmp_df.to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CSV}")
print(cmp_df.to_string(index=False))

# -----------------------
# Plot
# -----------------------
os.makedirs("outputs/figures", exist_ok=True)
plt.figure(figsize=(9, 5))
plt.plot(cmp_df["snr_db"], cmp_df["cnn2_accuracy"], marker="o", label="CNN2")
plt.plot(cmp_df["snr_db"], cmp_df["cnn4_accuracy"], marker="s", label="CNN4")
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs SNR: CNN2 vs CNN4")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
print(f"Saved -> {OUT_PNG}")
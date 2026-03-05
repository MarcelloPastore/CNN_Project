# CNN Project (RML2016.10b)

Confronto tra tre reti per classificazione modulazioni:
- **CNN2** (baseline paper-style)
- **T23** (architettura da tabella paper)
- **CNN4_v3** (variante ottimizzata)

## Requisiti

- Python 3.12
- Virtual env `.venv312`
- Dataset: `data/RML2016.10b.dat`

## Training

### 1) CNN2 (TensorFlow)
```bash
python src/train.py
```

Output:
- `models/cnn2_10c.keras`
- `outputs/tables/cnn2_train_metrics.json`

### 2) T23 (TensorFlow)
```bash
python src/train_cnn_paper_t23.py
```

Output:
- `models/cnn_paper_t23_10c.keras`
- `outputs/tables/cnn_paper_t23_train_metrics.json`

### 3) CNN4_v3
```bash
python src/train_cnn4_v3.py
```

Output:
- `models/cnn4_v3_10c.keras`
- `outputs/tables/cnn4_v3_train_metrics.json`

## Plot finali e confronto

Generazione confusion matrix + grafici comparativi:

```bash
python src/plot_result.py
```

Output in `outputs/figures/`:
- `confmat_cnn2.png`
- `confmat_t23.png`
- `confmat_cnn4v3.png`
- `compare_accuracy.png`
- `compare_accuracy_vs_snr.png`
- `final_comparison.png`

## Note sperimentali

- Confusion matrix generate **senza valori numerici nelle celle** (stile paper).
- Confronto finale include:
  - test accuracy
  - training time
  - epochs effettive
  - accuracy vs SNR (curve delle 3 reti).
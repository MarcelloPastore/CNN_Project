# AMC Project (CNN2 baseline)

Progetto minimale per classificazione di modulazioni radio (Automatic Modulation Classification) su dataset **RML2016.10A** usando **Keras + JAX**.

## Struttura
```text
amc-project/
├── requirements.txt
├── data/RML2016.10a_dict.pkl
├── models/
├── outputs/
│   ├── figures/
│   └── tables/
└── src/
    ├── train.py
    ├── evaluate.py
    └── export_parquet.py
```

## Setup rapido
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## Esecuzione
### 1) Train + salvataggio modello
```bash
python src/train.py
```
Output atteso:
- `models/cnn2_model.keras`

### 2) Valutazione + confusion matrix
```bash
python src/evaluate.py
```
Output attesi:
- `outputs/figures/confusion_matrix_normalized.png`
- `outputs/tables/accuracy_by_snr.csv`

### 3) Export predizioni in parquet
```bash
python src/export_parquet.py
```
Output atteso:
- `outputs/tables/predictions.parquet`

## Note
- Il warning `VisibleDeprecationWarning` su `pickle.load(..., encoding="latin1")` è noto con NumPy recenti e in genere non blocca il workflow.
- Se hai problemi GPU/memoria, riduci `BATCH_SIZE` nei file in `src/`.
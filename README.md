# CNN Project - Automatic Modulation Classification (AMC)

Progetto di classificazione automatica delle modulazioni radio (AMC) su dataset **RML2016.10a**, con implementazioni multiple:

- **CNN2** (baseline)
- **CNN4 v2**
- **CNN4 v3** (head più leggera e training più stabile)
- **Hybrid inference** (selettore CNN2/CNN4 in base a SNR)

---

## 1) Obiettivo

Confrontare diverse architetture CNN per il riconoscimento di modulazione (11 classi) e valutare:

- accuratezza globale
- accuratezza per SNR
- robustezza in regime di rumore
- benefici di una strategia ibrida (model selection)

---

## 2) Dataset

- File atteso: `data/RML2016.10a_dict.pkl`
- Formato input usato: `(2, 128, 1)` (IQ)
- Classi (11):  
  `['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']`
- Split usato negli script:
  - stratificato per `(mod, snr)`
  - `50% train / 50% test`
  - validation estratta dal train

---

## 3) Struttura principale (script)

- `src/train.py` → training CNN2
- `src/evaluate.py` → evaluation CNN2 (accuracy + confusion matrix + accuracy/SNR)
- `src/models/cnn4.py` → definizione CNN4 v2
- `src/train_cnn4.py` → training CNN4 v2
- `src/models/cnn4_v3.py` → definizione CNN4 v3
- `src/train_cnn4_v3.py` → training CNN4 v3
- `src/evaluate_cnn4.py` → evaluation CNN4/CNN4_v3 (in base al model path)
- `src/compare_cnn2_to_cnn4.py` → confronto CNN2 vs CNN4 su SNR
- `src/hybrid_inference.py` → inferenza ibrida con soglia SNR
- `src/hybrid_threshold_sweep.py` → sweep della soglia per trovare il threshold migliore

Output principali:
- `outputs/tables/*.csv`
- `outputs/figures/*.png`
- `models/*.keras`

---

## 4) Setup ambiente

### 4.1 Creazione virtual env

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4.2 Install dipendenze
(se hai `requirements.txt`)

```bash
pip install -r requirements.txt
```

Se non presente, install minima:

```bash
pip install numpy pandas matplotlib scikit-learn keras
```

> Nota: se usi backend JAX o TensorFlow, installa anche i pacchetti relativi in base alla tua configurazione GPU/driver.

---

## 5) Backend (JAX / TensorFlow)

Negli script trovi:

```python
os.environ["KERAS_BACKEND"] = "jax"
```

Se hai problemi di stabilità/memoria, prova TensorFlow:

```python
os.environ["KERAS_BACKEND"] = "tensorflow"
```

Per JAX su GPU con VRAM limitata può aiutare:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

---

## 6) Esecuzione rapida

### 6.1 Train CNN2
```bash
./.venv/bin/python src/train.py
```

### 6.2 Train CNN4 v2
```bash
./.venv/bin/python src/train_cnn4.py
```

### 6.3 Train CNN4 v3
```bash
./.venv/bin/python src/train_cnn4_v3.py
```

### 6.4 Evaluate CNN4 (o CNN4_v3)
```bash
./.venv/bin/python src/evaluate_cnn4.py
```

### 6.5 Confronto CNN2 vs CNN4
```bash
./.venv/bin/python src/compare_cnn2_to_cnn4.py
```

### 6.6 Hybrid inference (soglia fissa)
```bash
./.venv/bin/python src/hybrid_inference.py
```

### 6.7 Sweep soglia hybrid
```bash
./.venv/bin/python src/hybrid_threshold_sweep.py
```

---

## 7) Risultati (stato attuale)

Dalle ultime run:

- **CNN2 accuracy**: `0.5394`
- **CNN4 accuracy**: `0.5314`
- **HYBRID accuracy**: `0.5451` con soglia `-2 dB`

Sweep soglia (estratto):
- `-8 dB -> 0.5539`
- `-6 dB -> 0.5567`
- `-4 dB -> 0.5590`
- `-2 dB -> 0.5596` ✅ best
- `0 dB -> 0.5582`
- `2 dB -> 0.5568`
- `4 dB -> 0.5543`

Conclusione pratica:
- CNN2 più robusta in parte del regime a basso SNR
- CNN4 migliore in molte condizioni medio-alte
- selezione ibrida CNN2/CNN4 migliora l’accuracy complessiva

---

## 8) Note su batch size e hardware

Nel paper possono comparire batch size alti (128+), ma non sono universalmente migliori.  
Nel nostro setup (GPU 6GB), batch elevati possono causare warning/OOM.

Suggerimento pratico:
- training: `batch 16/32` (più stabile)
- inferenza: aumentare gradualmente finché resta stabile

---

## 9) Troubleshooting

### Warning `VisibleDeprecationWarning` su `pickle.load(..., encoding="latin1")`
È legato a compatibilità formato/NumPy e in genere non blocca l’esecuzione.

### Warning OOM / rematerialization (JAX/XLA)
Azioni consigliate:
1. ridurre batch size
2. impostare `XLA_PYTHON_CLIENT_PREALLOCATE=false`

---

## 10) Prossimi step consigliati

- multi-run (3-5 seed) e media ± std su accuracy
- report automatico in markdown con tabelle e grafici
- soft-gating ibrido (peso continuo tra CNN2 e CNN4, non solo soglia hard)
- test su dati live SDR per validazione real-world
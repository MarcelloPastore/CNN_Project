import pickle

def load_rml_any(path):
    # prova latin1 (tipico RadioML), fallback default
    with open(path, "rb") as f:
        try:
            return pickle.load(f, encoding="latin1")
        except TypeError:
            f.seek(0)
            return pickle.load(f)
import os
import sys

# aggiunge .../src al path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from load_rml import load_rml_any  # <-- non "from utils.load_rml"

Xd = load_rml_any("data/RML2016.10b.dat")
mods = sorted(set(k[0] for k in Xd.keys()))
snrs = sorted(set(k[1] for k in Xd.keys()))

print("mods:", mods)
print("num mods:", len(mods))
print("snrs:", snrs)
print("sample shape:", Xd[(mods[0], snrs[0])].shape)
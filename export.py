"""Export a Nicu checkpoint (int8 weights) to big-endian files for the Mac OS 9 engine.
usage: python export.py HF_DIR OUT_DIR"""
import sys, json, struct, pickle, types
import numpy as np, torch

src, out = sys.argv[1], sys.argv[2]

class _Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): self.__dict__.update(s if isinstance(s, dict) else {})
class _U(pickle.Unpickler):
    def find_class(self, mod, name):
        try: return super().find_class(mod, name)
        except Exception: return type(name, (_Stub,), {})
pm = types.SimpleNamespace(Unpickler=_U, load=lambda f, **k: _U(f, **k).load(), __name__='p')
ck = torch.load(f"{src}/checkpoint.pt", map_location="cpu", weights_only=False, pickle_module=pm)
cfg = ck["cfg"].__dict__
sd = {k: v.numpy().astype(">f4") for k, v in ck["model_state"].items() if not k.endswith("attn.mask")}
tied = bool(np.array_equal(sd["tok_emb.weight"], sd["head.weight"]))

order = ["tok_emb.weight", "pos_emb.weight"]
for i in range(cfg["n_layer"]):
    p = f"blocks.{i}."
    order += [p + n for n in ["ln1.weight", "ln1.bias", "attn.c_attn.weight", "attn.c_attn.bias",
                              "attn.c_proj.weight", "attn.c_proj.bias", "ln2.weight", "ln2.bias",
                              "mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias"]]
order += ["ln_f.weight", "ln_f.bias"] + ([] if tied else ["head.weight"])
assert set(order) | {"head.weight"} >= set(sd), set(sd) - set(order)

# Weights are int8 (one float scale per row) for the big matrices, float32 for the rest.
QUANT = ("tok_emb.weight", "head.weight", "attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
with open(f"{out}/nicu.dat", "wb") as f:
    f.write(b"NICU" + struct.pack(">7i", 2, cfg["vocab_size"], cfg["block_size"], cfg["n_layer"],
                                  cfg["n_head"], cfg["n_embd"], int(tied)))
    for k in order:
        w = sd[k].astype(np.float32)
        if k.endswith(QUANT):
            scale = np.abs(w).max(1) / 127.0
            scale[scale == 0] = 1.0
            q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
            f.write(scale.astype(">f4").tobytes() + q.tobytes())
        else:
            f.write(w.astype(">f4").tobytes())

# Tokenizer: raw bytes per token, the 256 single-byte ids, merges by rank.
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}
u2b = bytes_to_unicode()
tj = json.load(open(f"{src}/tokenizer.json", encoding="utf-8"))
vocab = tj["model"]["vocab"]
special = {t["content"]: t["id"] for t in tj["added_tokens"]}
V = cfg["vocab_size"]
raw = [b""] * V
for s, i in vocab.items():
    raw[i] = s.encode() if s in special else bytes(u2b[c] for c in s)
byte_id = [vocab[[k for k, v in u2b.items() if v == b][0]] for b in range(256)]
merges = [m.split(" ") if isinstance(m, str) else m for m in tj["model"]["merges"]]
with open(f"{out}/nicutok.dat", "wb") as f:
    f.write(b"NTOK" + struct.pack(">i", V))
    for r in raw: f.write(struct.pack(">B", len(r)) + r)
    f.write(struct.pack(">256i", *byte_id))
    f.write(struct.pack(">i", len(merges)))
    for a, b in merges: f.write(struct.pack(">3i", vocab[a], vocab[b], vocab[a + b]))
print("cfg", cfg, "tied", tied, "merges", len(merges))

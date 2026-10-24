"""Numpy reference forward pass of the int8 export, checked against the official fp32 ONNX."""
import sys, struct, numpy as np, onnxruntime as ort
from math import erf
def load(path):
    b = open(path, "rb").read()
    V, T, L, H, C, tied = struct.unpack(">6i", b[8:32]); o = 32
    def take(*shape):          # float32 tensor
        nonlocal o; n = int(np.prod(shape)); a = np.frombuffer(b, ">f4", n, o).astype(np.float32).reshape(shape); o += 4*n; return a
    def takeq(r, c):           # int8 matrix with one scale per row
        nonlocal o; s = np.frombuffer(b, ">f4", r, o).astype(np.float32); o += 4*r
        q = np.frombuffer(b, np.int8, r*c, o).astype(np.float32).reshape(r, c); o += r*c; return q * s[:, None]
    P = {"wte": takeq(V, C), "wpe": take(T, C), "layers": []}
    for _ in range(L):
        P["layers"].append(dict(ln1w=take(C), ln1b=take(C), qkvw=takeq(3*C, C), qkvb=take(3*C), ow=takeq(C, C), ob=take(C),
                                ln2w=take(C), ln2b=take(C), fcw=takeq(4*C, C), fcb=take(4*C), pw=takeq(C, 4*C), pb=take(C)))
    P["lnfw"], P["lnfb"] = take(C), take(C)
    return P, H
def ln(x, w, b): m = x.mean(-1, keepdims=True); v = ((x-m)**2).mean(-1, keepdims=True); return (x-m)/np.sqrt(v+1e-5)*w+b
gelu_erf = np.vectorize(lambda x: 0.5*x*(1+erf(x/np.sqrt(2))))
gelu_tanh = lambda x: 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))
def forward(P, H, ids, gelu):
    x = P["wte"][ids] + P["wpe"][:len(ids)]; T, C = x.shape; hd = C // H
    for l in P["layers"]:
        h = ln(x, l["ln1w"], l["ln1b"]) @ l["qkvw"].T + l["qkvb"]
        q, k, v = np.split(h, 3, -1)
        q = q.reshape(T, H, hd).transpose(1, 0, 2); k = k.reshape(T, H, hd).transpose(1, 0, 2); v = v.reshape(T, H, hd).transpose(1, 0, 2)
        a = q @ k.transpose(0, 2, 1) / np.sqrt(hd); a = a + np.triu(np.full((T, T), -1e9), 1)
        a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
        y = (a @ v).transpose(1, 0, 2).reshape(T, C)
        x = x + y @ l["ow"].T + l["ob"]
        x = x + gelu(ln(x, l["ln2w"], l["ln2b"]) @ l["fcw"].T + l["fcb"]) @ l["pw"].T + l["pb"]
    return ln(x, P["lnfw"], P["lnfb"]) @ P["wte"].T
if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "../strumenti/sito"; hf = sys.argv[2] if len(sys.argv) > 2 else "hf20"
    P, H = load(f"{d}/nicu.dat")
    ids = np.array([5, 99, 1234, 42, 7, 3000, 11], dtype=np.int64)
    s = ort.InferenceSession(f"{hf}/model.onnx")
    print([(i.name, i.shape) for i in s.get_inputs()], [o.name for o in s.get_outputs()])
    ref = s.run(None, {s.get_inputs()[0].name: ids[None]})[0][0]
    out = forward(P, H, ids, gelu_erf)
    print("max abs logit diff vs fp32 ONNX:", np.abs(out - ref).max(), "| top-1 agreement:", (out.argmax(-1) == ref.argmax(-1)).mean())

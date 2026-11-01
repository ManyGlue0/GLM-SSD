"""Write quantization test vectors decoded by gguf-py (llama.cpp's numpy reference).

Usage: <python with numpy> -I tools/qref.py <path-to-llama.cpp/gguf-py> <out_dir> [model.gguf ...]

For each type: random blocks with sane f16 scale fields -> <out>/rand_<TYPE>.bin + .f32.
If GGUF shards are given, also dumps a slice of the first real tensor of each type
(only tensors whose bytes are actually present/complete) -> <out>/real_<TYPE>.bin + .f32.
"""
import os
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
from gguf import GGMLQuantizationType as T  # noqa: E402
from gguf import quants  # noqa: E402

OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# type -> (block elements, block bytes, f16 field offsets, which of them are "mins")
TYPES = {
    T.Q8_0: (32, 34, [0]),
    T.Q2_K: (256, 84, [80, 82]),
    T.Q3_K: (256, 110, [108]),
    T.Q4_K: (256, 144, [0, 2]),
    T.Q5_K: (256, 176, [0, 2]),
    T.Q6_K: (256, 210, [208]),
    T.IQ2_XXS: (256, 66, [0]),
    T.IQ2_S: (256, 82, [0]),
    T.IQ3_XXS: (256, 98, [0]),
    T.IQ4_XS: (256, 136, [0]),
}

rng = np.random.default_rng(1234)
N_BLOCKS = 64

for t, (be, bb, halves) in TYPES.items():
    raw = rng.integers(0, 256, size=(N_BLOCKS, bb), dtype=np.uint8)
    for off in halves:
        v = (rng.uniform(1e-3, 5e-2, size=N_BLOCKS) * rng.choice([-1, 1], size=N_BLOCKS)).astype(np.float16)
        raw[:, off:off + 2] = v.view(np.uint8).reshape(N_BLOCKS, 2)
    y = quants.dequantize(raw.reshape(-1), t).astype(np.float32).reshape(-1)
    assert y.size == N_BLOCKS * be, (t.name, y.size)
    raw.tofile(os.path.join(OUT, f"rand_{t.name}.bin"))
    y.tofile(os.path.join(OUT, f"rand_{t.name}.f32"))
    print(f"rand {t.name:8s} {N_BLOCKS} blocks  absmax {np.abs(y).max():.4f}")

# ---- real tensors ---------------------------------------------------------------------
if len(sys.argv) > 3:
    from gguf import GGUFReader  # noqa: E402

    done = set()
    for path in sys.argv[3:]:
        state = path + ".part.json"
        if os.path.exists(path + ".part"):
            continue  # incomplete file
        r = GGUFReader(path)
        for tensor in r.tensors:
            t = tensor.tensor_type
            if t not in TYPES or t in done:
                continue
            be, bb, _ = TYPES[t]
            n_blocks = min(4096, tensor.data.nbytes // bb)
            raw = np.asarray(tensor.data).reshape(-1).view(np.uint8)[: n_blocks * bb]
            y = quants.dequantize(raw, t).astype(np.float32).reshape(-1)
            raw.tofile(os.path.join(OUT, f"real_{t.name}.bin"))
            y.tofile(os.path.join(OUT, f"real_{t.name}.f32"))
            done.add(t)
            print(f"real {t.name:8s} from {tensor.name} ({n_blocks} blocks)  absmax {np.abs(y).max():.4f}")

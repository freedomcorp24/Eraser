"""Build-time proof (Dockerfile RUN): the baked model loads on the CPU and erases. A broken image fails the BUILD, never a
user's request (the Lustify worker's rule, ARCHITECTURE "Self-hosted media"). Also usable on any machine:
    python3 verify.py [path/to/big-lama.pt]
A 96x96 picture: a smooth blue-to-green gradient with a red square in the middle; the mask covers the square. After the
erase the square must be gone (no red left inside it) and every pixel outside the mask must be untouched.
"""

import base64
import io
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

import lama

path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("LAMA_MODEL_PATH", "/models/big-lama.pt")
t = time.time()
model = torch.jit.load(path, map_location="cpu").eval()
print("load: %.1f s" % (time.time() - t), flush=True)

h = w = 96
y, x = np.mgrid[0:h, 0:w]
pic = np.stack([np.zeros_like(x), (x * 255 // (w - 1)), (255 - y * 255 // (h - 1))], axis=-1).astype(np.uint8)
pic[36:60, 36:60] = (230, 20, 20)
mask = np.zeros((h, w), np.uint8)
mask[32:64, 32:64] = 255


def b64png(arr, mode):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


t = time.time()
out = lama.handle({"image": b64png(pic, "RGB"), "mask": b64png(mask, "L")}, lama.torch_runner(model, torch, torch.device("cpu")))
print("erase 96x96: %.2f s" % (time.time() - t), flush=True)
assert "error" not in out, out
res = np.asarray(Image.open(io.BytesIO(base64.b64decode(out["image_base64"]))).convert("RGB"))
assert res.shape == pic.shape, res.shape
outside = mask <= 127
assert np.array_equal(res[outside], pic[outside]), "pixels outside the mask changed"
square = res[36:60, 36:60].astype(int)
redness = float((square[..., 0] - np.maximum(square[..., 1], square[..., 2])).mean())
print("red left in the square: %.1f (was %.1f)" % (redness, 230 - 20), flush=True)
assert redness < 40, "the red square was not erased"
print("verify ok", flush=True)

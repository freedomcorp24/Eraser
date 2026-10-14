"""GramsAI magic eraser: RunPod serverless worker (docs/features/DESIGN_CANVAS_SPEC.md 19). The same shape as the Z-Image
worker (github.com/freedomcorp24/ZImage): the model loads once, at start, outside the handler; every job is one erase.

The weights are BAKED INTO THE IMAGE at build time (Dockerfile, md5-checked) and proved to load and erase at build time
(verify.py), per ARCHITECTURE "Self-hosted media": no download at boot, no network volume pinning the endpoint to one
datacenter. Endpoint env: none needed. GPU: torch is the cu124 build, so enable only Ampere / Ada cards (PITFALLS #17).
"""

import os

import runpod
import torch

import lama

MODEL_PATH = os.environ.get("LAMA_MODEL_PATH", "/models/big-lama.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading LaMa from %s on %s" % (MODEL_PATH, DEVICE), flush=True)
model = torch.jit.load(MODEL_PATH, map_location="cpu").to(DEVICE).eval()
run = lama.torch_runner(model, torch, DEVICE)
print("Model ready.", flush=True)


def handler(event):
    return lama.handle((event or {}).get("input"), run)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

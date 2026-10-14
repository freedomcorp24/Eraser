# runpod/eraser: the canvas's magic eraser worker

LaMa inpainting (advimman/lama, Apache-2.0) on a RunPod serverless GPU, for the image editor's Magic eraser
(docs/features/DESIGN_CANVAS_SPEC.md 19). The same pattern as the Z-Image worker (github.com/freedomcorp24/ZImage):
RunPod builds this folder's Dockerfile from GitHub; the model loads once per worker.

| file | what |
|---|---|
| `Dockerfile` | CUDA 12.4 + torch 2.5.1 cu124 (same as Z-Image); bakes `big-lama.pt` (196 MB, md5-checked); runs `verify.py` at BUILD time |
| `lama.py` | the job: decode, pad to 8, run, keep every pixel outside the mask, encode. IOPaint 1.6.0's own LaMa steps, without IOPaint |
| `handler.py` | loads the model on start, `runpod.serverless.start` |
| `verify.py` | loads the model on the CPU and erases a red square; a broken image fails the build, never a user |
| `test_lama.py` | unit tests, no GPU or model: `python3 -m unittest -v test_lama` |

Contract (the gateway, `m1/app/internal/proxy/canvas_erase.go`, is the only caller):
`{"input": {"image": "<base64>", "mask": "<base64 PNG, white = erase>"}}` ->
`{"image_base64": "<PNG>", "width": W, "height": H}` or `{"error": "<why>"}`. The app sends a crop of at most 1280 px.

## Measured (sandbox CPU, 2 threads, 2026-10-13; a GPU is the target and is NOT measured yet)
- Model load 1.1 s; 96x96 erase 1.7 s (first call); 512x384 3.2 s; 1280x960 21.1 s; peak RSS 2.0 GB.
- On M2 the IOPaint stack was OOM-killed at 6.2 GB while loading (spec 18); this worker's own load peaks at 672 MB.

## Setting it up (John, after the release that carries the gateway route)
1. RunPod console, Serverless, New Endpoint, source GitHub: this repository (if RunPod's GitHub app can see it) with
   Dockerfile path `runpod/eraser/Dockerfile` and build context `runpod/eraser`; or a public mirror repository holding
   only this folder's files, the way ZImage is.
2. GPUs: Ampere / Ada only (RTX A4000 / A4500 / A5000, RTX 4090, L4, A40). Never Blackwell (PITFALLS #17) and never
   a MIG slice. Execution timeout at least 120 s. Max workers as for Z-Image.
3. Create the endpoint's own API key (RunPod keys are per endpoint, PITFALLS #114).
4. M1, in `/etc/gramsai/gramsai.env`: `RUNPOD_ENDPOINT_ERASE=<endpoint id>` and `RUNPOD_KEY_ERASE=<key>`; once the
   console shows the GPU price, `RUNPOD_RATE_ERASE_MICROS_PER_SEC=<micro-dollars per second>` (default 339, the
   never-undercharge Z-Image rate). Then `systemctl restart gramsai-api`.
5. Until step 4 the app's eraser answers "The magic eraser is not set up yet" and nothing else changes.

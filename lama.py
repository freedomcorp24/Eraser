"""GramsAI magic eraser: the pure part of the RunPod worker (docs/features/DESIGN_CANVAS_SPEC.md 19).

LaMa ("Resolution-robust Large Mask Inpainting with Fourier Convolutions", advimman/lama, Apache-2.0) as the TorchScript
file IOPaint ships (big-lama.pt, 196 MB, md5 e3aa4aaa15225a33ec84f9f4bc47e500). The inference steps are IOPaint 1.6.0's
own for LaMa (iopaint/model/lama.py forward + model/base.py _pad_forward, Apache-2.0): pad to a multiple of 8 by
mirroring, image and mask as float 0..1, mask > 0 = paint here, run, crop back. Without IOPaint itself: its import pulls
diffusers, transformers, gradio and friends (and on M2's CPU it was killed at 6.2 GB while loading, spec 18); this needs
torch, numpy and Pillow.

THE CONTRACT (the gateway, m1/app/internal/proxy/canvas_erase.go, is the only caller):
  input  {"image": base64 PNG/JPEG/WebP (a data: URL is accepted), "mask": base64 PNG, same size, white = erase}
  output {"image_base64": base64 PNG of the same size, "width": W, "height": H}   or   {"error": "<why>"}
Outside the mask the result IS the input, pixel for pixel (composited here, not trusted to the model). The app sends a
crop around what was brushed, at most 1280 px on its long side (IOPaint's own defaults: crop margin 128, resize limit
1280), and pastes the brushed pixels back into the full picture; anything larger than MAX_SIDE is refused, never resized
here, so the worker's memory and the 10 MB RunPod /run payload stay bounded.
"""

import base64
import binascii
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_SIDE = 2048
PAD_MOD = 8

Image.MAX_IMAGE_PIXELS = MAX_SIDE * MAX_SIDE  # Pillow's decompression-bomb guard, at our own limit


class EraseError(ValueError):
    """A request the worker refuses; its message goes back to the user."""


def decode_image(b64, mode):
    if not isinstance(b64, str) or not b64:
        raise EraseError("missing image data")
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[-1]
    try:
        raw = base64.b64decode(b64, validate=False)
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise EraseError("unreadable image (%s)" % (type(e).__module__ + "." + type(e).__name__))
    if max(img.size) > MAX_SIDE:
        raise EraseError("image too large: %dx%d (max %d px a side)" % (img.size[0], img.size[1], MAX_SIDE))
    if mode == "RGB" and img.mode in ("RGBA", "LA", "P"):
        # Transparent pixels have no meaningful colour; white keeps them neutral for the model. The app keeps the
        # original alpha and only takes the brushed pixels' colour from the result.
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert(mode)


def pad_to_modulo(arr, mod=PAD_MOD):
    """IOPaint pad_img_to_modulo: mirror-pad the bottom and right edges up to a multiple of `mod`."""
    h, w = arr.shape[:2]
    ph = (mod - h % mod) % mod
    pw = (mod - w % mod) % mod
    if arr.ndim == 2:
        return np.pad(arr, ((0, ph), (0, pw)), mode="symmetric")
    return np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="symmetric")


def erase(run, rgb, mask):
    """rgb uint8 HxWx3, mask uint8 HxW (> 127 = erase). `run(image, mask)` takes float32 arrays 1x3xHxW / 1x1xHxW in
    0..1 (sizes a multiple of 8) and returns 1x3xHxW in 0..1. Returns uint8 HxWx3: the model's pixels inside the mask,
    the input's everywhere else."""
    h, w = rgb.shape[:2]
    hole = mask > 127
    if not hole.any():
        return rgb.copy()
    img = pad_to_modulo(rgb).astype(np.float32) / 255.0
    msk = pad_to_modulo(hole.astype(np.float32))
    out = run(np.transpose(img, (2, 0, 1))[None], msk[None, None])
    out = np.asarray(out, dtype=np.float32)[0].transpose(1, 2, 0)[:h, :w]
    res = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.where(hole[:, :, None], res, rgb)


def encode_png(rgb):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(buf, format="PNG", compress_level=6)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def handle(job_input, run):
    """The whole job, minus the model: validate, erase, encode. Never raises for a bad request."""
    if not isinstance(job_input, dict):
        return {"error": "input must be an object"}
    try:
        image = decode_image(job_input.get("image"), "RGB")
        mask = decode_image(job_input.get("mask"), "L")
        if image.size != mask.size:
            raise EraseError("image is %dx%d but mask is %dx%d" % (image.size + mask.size))
        out = erase(run, np.asarray(image, dtype=np.uint8), np.asarray(mask, dtype=np.uint8))
    except EraseError as e:
        return {"error": str(e)}
    return {"image_base64": encode_png(out), "width": out.shape[1], "height": out.shape[0]}


def torch_runner(model, torch, device):
    """The model call for `erase`, on `device`, without autograd."""

    def run(image, mask):
        with torch.inference_mode():
            t_img = torch.from_numpy(np.ascontiguousarray(image)).to(device)
            t_msk = torch.from_numpy(np.ascontiguousarray(mask)).to(device)
            return model(t_img, t_msk).float().cpu().numpy()

    return run

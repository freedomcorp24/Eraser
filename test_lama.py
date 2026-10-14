"""Unit tests for lama.py (docs/features/DESIGN_CANVAS_SPEC.md 19, tests E1-E6). No GPU and no model: a fake runner
records what it is given. The real model is proved by verify.py (build time and any machine). Run: python3 -m unittest -v
"""

import base64
import io
import unittest

import numpy as np
from PIL import Image

import lama


def b64(arr, mode, fmt="PNG"):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def unb64(s):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB"))


class Fake:
    """Answers 0.5 everywhere and remembers the shapes and mask values it was given."""

    def __init__(self):
        self.calls = []

    def __call__(self, image, mask):
        self.calls.append((image.shape, mask.shape, set(np.unique(mask).tolist()), image.dtype, float(image.max())))
        return np.full(image.shape, 0.5, np.float32)


class TestPad(unittest.TestCase):
    def test_e1_pads_bottom_right_to_a_multiple_of_8_by_mirroring(self):
        a = np.arange(13 * 21 * 3, dtype=np.uint8).reshape(13, 21, 3)
        p = lama.pad_to_modulo(a)
        self.assertEqual(p.shape, (16, 24, 3))
        self.assertTrue(np.array_equal(p[:13, :21], a))
        self.assertTrue(np.array_equal(p[13, :21], a[12]))  # mirrored edge row
        self.assertTrue(np.array_equal(p[:13, 21], a[:, 20]))  # mirrored edge column
        self.assertEqual(lama.pad_to_modulo(np.zeros((16, 8), np.uint8)).shape, (16, 8))


class TestErase(unittest.TestCase):
    def test_e2_model_pixels_inside_the_mask_input_pixels_outside(self):
        rgb = np.random.default_rng(1).integers(0, 255, (13, 21, 3), dtype=np.uint8)
        mask = np.zeros((13, 21), np.uint8)
        mask[3:9, 5:15] = 255
        fake = Fake()
        out = lama.erase(fake, rgb, mask)
        self.assertEqual(out.shape, rgb.shape)
        self.assertTrue((out[3:9, 5:15] == 128).all())
        keep = mask <= 127
        self.assertTrue(np.array_equal(out[keep], rgb[keep]))
        (ishape, mshape, mvals, dtype, imax) = fake.calls[0]
        self.assertEqual(ishape, (1, 3, 16, 24))
        self.assertEqual(mshape, (1, 1, 16, 24))
        self.assertEqual(mvals, {0.0, 1.0})
        self.assertEqual(dtype, np.float32)
        self.assertLessEqual(imax, 1.0)

    def test_e3_threshold_is_above_127_and_an_empty_mask_never_runs_the_model(self):
        rgb = np.zeros((8, 8, 3), np.uint8)
        mask = np.full((8, 8), 127, np.uint8)
        fake = Fake()
        self.assertTrue(np.array_equal(lama.erase(fake, rgb, mask), rgb))
        self.assertEqual(fake.calls, [])
        mask[0, 0] = 128
        out = lama.erase(fake, rgb, mask)
        self.assertEqual(out[0, 0].tolist(), [128, 128, 128])
        self.assertEqual(out[0, 1].tolist(), [0, 0, 0])


class TestHandle(unittest.TestCase):
    def setUp(self):
        self.rgb = np.random.default_rng(2).integers(0, 255, (40, 30, 3), dtype=np.uint8)
        self.mask = np.zeros((40, 30), np.uint8)
        self.mask[10:20, 10:20] = 255

    def test_e4_a_job_comes_back_as_a_png_of_the_same_size(self):
        out = lama.handle({"image": "data:image/png;base64," + b64(self.rgb, "RGB"), "mask": b64(self.mask, "L")}, Fake())
        self.assertEqual((out["width"], out["height"]), (30, 40))
        res = unb64(out["image_base64"])
        self.assertTrue((res[10:20, 10:20] == 128).all())
        self.assertTrue(np.array_equal(res[:10], self.rgb[:10]))

    def test_e4b_a_jpeg_image_is_accepted(self):
        out = lama.handle({"image": b64(self.rgb, "RGB", "JPEG"), "mask": b64(self.mask, "L")}, Fake())
        self.assertNotIn("error", out)

    def test_e5_bad_requests_are_refused_with_a_reason_never_raised(self):
        big = np.zeros((lama.MAX_SIDE + 1, 8, 3), np.uint8)
        cases = {
            "input must be an object": "nope",
            "missing image data": {"mask": b64(self.mask, "L")},
            "unreadable image": {"image": "bm90IGFuIGltYWdl", "mask": b64(self.mask, "L")},
            "image is 30x40 but mask is 10x10": {"image": b64(self.rgb, "RGB"), "mask": b64(np.zeros((10, 10), np.uint8), "L")},
            "image too large": {"image": b64(big, "RGB"), "mask": b64(np.zeros((lama.MAX_SIDE + 1, 8), np.uint8), "L")},
        }
        for want, job in cases.items():
            out = lama.handle(job, Fake())
            self.assertIn("error", out, want)
            self.assertTrue(out["error"].startswith(want), (want, out["error"]))

    def test_e6_transparent_pixels_reach_the_model_as_white(self):
        rgba = np.zeros((8, 8, 4), np.uint8)
        rgba[..., 0] = 200
        rgba[:4, :, 3] = 255  # top half opaque red, bottom half fully transparent
        seen = {}

        def run(image, mask):
            seen["img"] = image
            return np.full(image.shape, 0.5, np.float32)

        mask = np.full((8, 8), 255, np.uint8)
        out = lama.handle({"image": b64(rgba, "RGBA"), "mask": b64(mask, "L")}, run)
        self.assertNotIn("error", out)
        img = seen["img"][0]
        self.assertAlmostEqual(float(img[0, 0, 0]), 200 / 255, places=3)
        self.assertAlmostEqual(float(img[1, 7, 0]), 1.0, places=3)  # transparent -> white


if __name__ == "__main__":
    unittest.main()

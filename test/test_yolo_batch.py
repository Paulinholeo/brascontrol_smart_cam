from detection.yolov9 import YOLOv9
import cv2
from tinygrad import Tensor
from utils.helpers import jit_infer
import numpy as np

def top(preds, thresh=0.7): return [list(p) for p in preds if p[4] > thresh]

if __name__ == "__main__":
  img = cv2.imread("test/clip_images/front.png")
  small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
  model = YOLOv9("t", 960)

  # old single-frame path, kept for VOD
  jit_cache = {}
  for _ in range(3): ref = jit_infer(model, Tensor(img), jit_cache).numpy()
  ref = top(ref)
  assert len(ref) == 1
  np.testing.assert_allclose([511.69214, 367.81128, 1294.1671, 899.5974, 0.8302731, 2.0], ref[0], rtol=1e-4)

  # batched path: mixed resolutions and a non power of 2 batch (pads to 4)
  for _ in range(3): out = model.detect_batch([img, small, img])
  for i in (0, 2):
    got = top(out[i])
    assert len(got) == 1, got
    # the old path interpolates in uint8 and wraps ~0.35% of pixels; cv2 matches tinygrad's float interpolate.
    # same box within 3 px, score may differ (0.83 old vs 0.89 cv2 on this image)
    np.testing.assert_allclose(got[0][:4], ref[0][:4], atol=3)
    assert abs(got[0][4] - ref[0][4]) < 0.1 and got[0][5] == ref[0][5]
  got = top(out[1])
  assert len(got) == 1 and got[0][5] == 2.0, got
  np.testing.assert_allclose(np.array(got[0][:4]) * 2, ref[0][:4], atol=8)
  print("ok")

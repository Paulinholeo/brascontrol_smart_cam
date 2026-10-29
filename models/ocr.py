# language: Python, file: models/ocr.py, target: Windows 11 / Cross-platform
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


PLATE_REGEX_MERCOSUL = re.compile(r"[A-Z]{3}[0-9][A-Z0-9][0-9]{2}")

DIGIT_FIX = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
LETTER_FIX = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}


def find_tesseract_binary(custom_path: Optional[str] = None) -> Optional[str]:
    if custom_path and os.path.isfile(custom_path):
        return custom_path
    which_bin = shutil.which("tesseract")
    if which_bin:
        return which_bin
    win_candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for cand in win_candidates:
        if os.path.isfile(cand):
            return cand
    return None


def normalize_br_plate(raw_text: str) -> Optional[str]:
    """Extracts and normalizes Brazilian Mercosul (ABC1D23) or legacy (ABC1234) license plates."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text.upper())
    if len(cleaned) < 7:
        return None
    for i in range(len(cleaned) - 6):
        sub = list(cleaned[i : i + 7])
        # First 3 positions must be letters
        for pos in (0, 1, 2):
            if sub[pos].isdigit() and sub[pos] in LETTER_FIX:
                sub[pos] = LETTER_FIX[sub[pos]]
        # Position 3, 5, 6 must be digits
        for pos in (3, 5, 6):
            if sub[pos].isalpha() and sub[pos] in DIGIT_FIX:
                sub[pos] = DIGIT_FIX[sub[pos]]
        candidate = "".join(sub)
        if PLATE_REGEX_MERCOSUL.fullmatch(candidate):
            return candidate
    return None


class MultiOCREngine:
    """
    Unified OCR Analytics Engine for Clearcam supporting:
      - 'rapidocr'  (RapidAI/RapidOCR PaddleOCR PP-OCRv4 ONNX - fast, zero GPU bottleneck)
      - 'easyocr'   (JaidedAI/EasyOCR PyTorch CRNN/CRAFT)
      - 'tesseract' (UB-Mannheim / Google Tesseract OCR via pytesseract)
      - 'ensemble'  (RapidOCR + Tesseract/EasyOCR fallback voting)
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._rapid = None
        self._easy = None
        self._easy_langs = None
        self._tesseract_ready = False

    def _preprocess(self, img_bgr: np.ndarray, plate_mode: bool = False) -> Tuple[np.ndarray, float]:
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr, 1.0
        h, w = img_bgr.shape[:2]
        scale = 1.0
        if max(h, w) < 320:
            scale = 320.0 / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5 if plate_mode else 1.8, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        return enhanced, scale

    def _init_rapid(self):
        if self._rapid is None:
            from rapidocr_onnxruntime import RapidOCR
            self._rapid = RapidOCR()

    def _init_easy(self, langs: List[str]):
        if self._easy is None or self._easy_langs != tuple(langs):
            import sys
            prev_utils = {k: v for k, v in sys.modules.items() if k == "utils" or k.startswith("utils.")}
            prev_path = list(sys.path)
            try:
                import easyocr
                use_gpu = False
                try:
                    import torch
                    use_gpu = torch.cuda.is_available()
                except Exception:
                    pass
                self._easy = easyocr.Reader(langs, gpu=use_gpu, verbose=False)
                self._easy_langs = tuple(langs)
            finally:
                for k in [m for m in list(sys.modules.keys()) if m == "utils" or m.startswith("utils.")]:
                    if k not in prev_utils:
                        sys.modules.pop(k, None)
                sys.modules.update(prev_utils)
                sys.path[:] = prev_path

    def _init_tesseract(self, custom_path: Optional[str] = None) -> str:
        import pytesseract
        bin_path = find_tesseract_binary(custom_path)
        if not bin_path:
            raise RuntimeError("Tesseract executable not found on system")
        pytesseract.pytesseract.tesseract_cmd = bin_path
        self._tesseract_ready = True
        return bin_path

    def available_engines(self) -> Dict[str, bool]:
        import importlib.util
        return {
            "tesseract": find_tesseract_binary() is not None,
            "rapidocr": importlib.util.find_spec("rapidocr_onnxruntime") is not None,
            "easyocr": importlib.util.find_spec("easyocr") is not None,
        }

    def read_image(
        self,
        img_bgr: np.ndarray,
        engine: str = "rapidocr",
        plate_mode: bool = False,
        languages: str = "pt,en",
        min_conf: float = 0.35,
        tesseract_path: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """
        Returns list of dicts: [{"text": str, "plate": str|None, "conf": float, "engine": str, "box": [x1,y1,x2,y2]}]
        """
        if img_bgr is None or img_bgr.size == 0:
            return []

        orig_h, orig_w = img_bgr.shape[:2]
        proc, scale = self._preprocess(img_bgr, plate_mode=plate_mode)
        engine = (engine or "rapidocr").lower()
        results: List[Dict[str, object]] = []

        def _unscale_box(bx1: int, by1: int, bx2: int, by2: int) -> List[int]:
            inv = 1.0 / scale if scale > 0 else 1.0
            ox1 = max(0, min(orig_w - 1, int(round(bx1 * inv))))
            oy1 = max(0, min(orig_h - 1, int(round(by1 * inv))))
            ox2 = max(ox1 + 1, min(orig_w, int(round(bx2 * inv))))
            oy2 = max(oy1 + 1, min(orig_h, int(round(by2 * inv))))
            return [ox1, oy1, ox2, oy2]

        with self._lock:
            if engine in ("rapidocr", "ensemble"):
                try:
                    self._init_rapid()
                    raw, _ = self._rapid(proc)
                    if raw:
                        for item in raw:
                            box_pts, txt, score = item[0], str(item[1]).strip(), float(item[2])
                            if score < min_conf or not txt:
                                continue
                            xs = [int(p[0]) for p in box_pts]
                            ys = [int(p[1]) for p in box_pts]
                            plate = normalize_br_plate(txt)
                            if plate_mode and not plate:
                                continue
                            results.append({
                                "text": plate if plate else txt,
                                "raw_text": txt,
                                "plate": plate,
                                "conf": round(score, 3),
                                "engine": "rapidocr",
                                "box": _unscale_box(min(xs), min(ys), max(xs), max(ys)),
                            })
                except Exception as exc:
                    print(f"[OCR RapidOCR] warning: {exc}")

            if engine == "easyocr" or (engine == "ensemble" and not results):
                try:
                    langs = [x.strip() for x in languages.split(",") if x.strip()] or ["pt", "en"]
                    self._init_easy(langs)
                    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" if plate_mode else None
                    raw = self._easy.readtext(proc, allowlist=allowlist)
                    for box_pts, txt, score in raw:
                        txt = str(txt).strip()
                        score = float(score)
                        if score < min_conf or not txt:
                            continue
                        xs = [int(p[0]) for p in box_pts]
                        ys = [int(p[1]) for p in box_pts]
                        plate = normalize_br_plate(txt)
                        if plate_mode and not plate:
                            continue
                        results.append({
                            "text": plate if plate else txt,
                            "raw_text": txt,
                            "plate": plate,
                            "conf": round(score, 3),
                            "engine": "easyocr",
                            "box": _unscale_box(min(xs), min(ys), max(xs), max(ys)),
                        })
                except Exception as exc:
                    print(f"[OCR EasyOCR] warning: {exc}")

            if engine == "tesseract" or (engine == "ensemble" and not results):
                try:
                    import pytesseract
                    self._init_tesseract(tesseract_path)
                    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
                    _, thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    cfg = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" if plate_mode else "--oem 3 --psm 6"
                    tess_lang = "eng"
                    data = pytesseract.image_to_data(thresh_img, lang=tess_lang, config=cfg, output_type=pytesseract.Output.DICT)
                    words = []
                    confs = []
                    x1, y1, x2, y2 = 99999, 99999, 0, 0
                    for i, w_txt in enumerate(data.get("text", [])):
                        w_txt = str(w_txt).strip()
                        try:
                            c = float(data["conf"][i]) / 100.0
                        except Exception:
                            c = 0.0
                        if w_txt and c >= min_conf * 0.7:
                            words.append(w_txt)
                            confs.append(c)
                            lx, ly, lw, lh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                            x1, y1 = min(x1, lx), min(y1, ly)
                            x2, y2 = max(x2, lx + lw), max(y2, ly + lh)
                    if words:
                        joined = " ".join(words)
                        avg_conf = sum(confs) / len(confs)
                        plate = normalize_br_plate(joined)
                        if not (plate_mode and not plate):
                            b_coords = _unscale_box(x1, y1, x2, y2) if x2 > x1 else [0, 0, orig_w, orig_h]
                            results.append({
                                "text": plate if plate else joined,
                                "raw_text": joined,
                                "plate": plate,
                                "conf": round(avg_conf, 3),
                                "engine": "tesseract",
                                "box": b_coords,
                            })
                except Exception as exc:
                    print(f"[OCR Tesseract] warning: {exc}")

        results.sort(key=lambda r: (1 if r.get("plate") else 0, float(r.get("conf", 0.0))), reverse=True)
        return results


ocr_engine = MultiOCREngine()

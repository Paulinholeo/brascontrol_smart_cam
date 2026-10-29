# language: Python 3, file: models/vehicle_attributes.py, target: Windows 11
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np

VEHICLE_COLORS: Dict[str, Dict[str, str]] = {
    "branca": {"label": "Branca", "hex": "#f8fafc", "border": "#cbd5e1"},
    "preta": {"label": "Preta", "hex": "#18181b", "border": "#3f3f46"},
    "prata": {"label": "Prata / Cinza", "hex": "#9ca3af", "border": "#6b7280"},
    "vermelha": {"label": "Vermelha", "hex": "#dc2626", "border": "#991b1b"},
    "azul": {"label": "Azul", "hex": "#2563eb", "border": "#1d4ed8"},
    "amarela": {"label": "Amarela", "hex": "#eab308", "border": "#ca8a04"},
    "verde": {"label": "Verde", "hex": "#16a34a", "border": "#15803d"},
    "marrom": {"label": "Marrom / Bege", "hex": "#92400e", "border": "#78350f"},
    "indefinida": {"label": "Não Identificada", "hex": "#64748b", "border": "#475569"},
}

YOLO_CLASS_CATEGORY: Dict[int, str] = {
    1: "Bicicleta",
    2: "Automóvel",
    3: "Motocicleta",
    5: "Ônibus",
    7: "Caminhão / Utilitário",
}

VALID_BR_PLATE_RE = re.compile(r"^[A-Z]{3}[0-9][A-Z0-9][0-9]{2}$")

# Classes automotivas ImageNet-1000 para MobileNetV2 ONNX (restrito ao domínio veicular)
IMAGENET_VEHICLE_MAP: Dict[int, Tuple[str, str]] = {
    407: ("FURGÃO / UTILITÁRIO", "Furgão / Utilitário"),
    436: ("SUV / CROSSOVER", "SUV / Crossover"),
    468: ("PASSEIO / SEDAN", "Sedan / Passeio"),
    511: ("PASSEIO / COUPÉ", "Sedan / Passeio"),
    569: ("CAMINHÃO / PESADO", "Caminhão"),
    609: ("SUV / UTILITÁRIO 4X4", "SUV / Crossover"),
    627: ("PASSEIO / SEDAN EXECUTIVO", "Sedan / Passeio"),
    654: ("VAN / MICRO-ÔNIBUS", "Ônibus / Van"),
    656: ("HATCH / MONOVOLUME", "Hatch / Passeio"),
    665: ("MOTOCICLETA / URBANA", "Motocicleta"),
    670: ("MOTOCICLETA / SCOOTER", "Motocicleta"),
    675: ("FURGÃO / BAÚ", "Furgão / Utilitário"),
    717: ("PICAPE / CABINE DUPLA", "Picape / Utilitário"),
    734: ("FURGÃO / UTILITÁRIO", "Furgão / Utilitário"),
    751: ("PASSEIO / HATCH", "Hatch / Passeio"),
    779: ("ÔNIBUS / COLETIVO", "Ônibus"),
    817: ("PASSEIO / SEDAN", "Sedan / Passeio"),
    864: ("CAMINHÃO / VUC", "Caminhão"),
    867: ("CAMINHÃO / CARRETA", "Caminhão"),
    874: ("ÔNIBUS / URBANO", "Ônibus"),
}

# Detecção de emblemas de marca/modelo lidos pelo OCR na tampa traseira / grade
BADGE_MODEL_KEYWORDS: List[Tuple[str, str, str]] = [
    ("STRADA", "FIAT/STRADA", "Picape / Utilitário"),
    ("TORO", "FIAT/TORO", "Picape / Utilitário"),
    ("ARGO", "FIAT/ARGO", "Hatch / Passeio"),
    ("MOBI", "FIAT/MOBI", "Hatch / Passeio"),
    ("FIORINO", "FIAT/FIORINO", "Furgão / Utilitário"),
    ("CRONOS", "FIAT/CRONOS", "Sedan / Passeio"),
    ("PULSE", "FIAT/PULSE", "SUV / Crossover"),
    ("FASTBACK", "FIAT/FASTBACK", "SUV / Crossover"),
    ("SIENA", "FIAT/SIENA", "Sedan / Passeio"),
    ("PALIO", "FIAT/PALIO", "Hatch / Passeio"),
    ("UNO", "FIAT/UNO", "Hatch / Passeio"),
    ("SAVEIRO", "VW/SAVEIRO", "Picape / Utilitário"),
    ("AMAROK", "VW/AMAROK", "Picape / Utilitário"),
    ("TCROSS", "VW/T-CROSS", "SUV / Crossover"),
    ("NIVUS", "VW/NIVUS", "SUV / Crossover"),
    ("VIRTUS", "VW/VIRTUS", "Sedan / Passeio"),
    ("VOYAGE", "VW/VOYAGE", "Sedan / Passeio"),
    ("POLO", "VW/POLO", "Hatch / Passeio"),
    ("GOL", "VW/GOL", "Hatch / Passeio"),
    ("ONIX", "CHEVROLET/ONIX", "Hatch / Passeio"),
    ("PRISMA", "CHEVROLET/PRISMA", "Sedan / Passeio"),
    ("TRACKER", "CHEVROLET/TRACKER", "SUV / Crossover"),
    ("SPIN", "CHEVROLET/SPIN", "Hatch / Monovolume"),
    ("MONTANA", "CHEVROLET/MONTANA", "Picape / Utilitário"),
    ("S10", "CHEVROLET/S10", "Picape / Utilitário"),
    ("CRUZE", "CHEVROLET/CRUZE", "Sedan / Passeio"),
    ("HB20", "HYUNDAI/HB20", "Hatch / Passeio"),
    ("CRETA", "HYUNDAI/CRETA", "SUV / Crossover"),
    ("HILUX", "TOYOTA/HILUX", "Picape / Utilitário"),
    ("COROLLA", "TOYOTA/COROLLA", "Sedan / Passeio"),
    ("YARIS", "TOYOTA/YARIS", "Hatch / Passeio"),
    ("ETIOS", "TOYOTA/ETIOS", "Hatch / Passeio"),
    ("SW4", "TOYOTA/HILUX SW4", "SUV / Crossover"),
    ("CIVIC", "HONDA/CIVIC", "Sedan / Passeio"),
    ("CITY", "HONDA/CITY", "Sedan / Passeio"),
    ("HRV", "HONDA/HR-V", "SUV / Crossover"),
    ("FIT", "HONDA/FIT", "Hatch / Passeio"),
    ("KWID", "RENAULT/KWID", "Hatch / Passeio"),
    ("SANDERO", "RENAULT/SANDERO", "Hatch / Passeio"),
    ("LOGAN", "RENAULT/LOGAN", "Sedan / Passeio"),
    ("DUSTER", "RENAULT/DUSTER", "SUV / Crossover"),
    ("OROCH", "RENAULT/OROCH", "Picape / Utilitário"),
    ("MASTER", "RENAULT/MASTER", "Furgão / Utilitário"),
    ("RENEGADE", "JEEP/RENEGADE", "SUV / Crossover"),
    ("COMPASS", "JEEP/COMPASS", "SUV / Crossover"),
    ("COMMANDER", "JEEP/COMMANDER", "SUV / Crossover"),
    ("RANGER", "FORD/RANGER", "Picape / Utilitário"),
    ("ECOSPORT", "FORD/ECOSPORT", "SUV / Crossover"),
    ("KICKS", "NISSAN/KICKS", "SUV / Crossover"),
    ("VERSA", "NISSAN/VERSA", "Sedan / Passeio"),
    ("FRONTIER", "NISSAN/FRONTIER", "Picape / Utilitário"),
]

# Faixas alfabéticas DENATRAN (3 letras iniciais -> UF de emplacamento original)
DENATRAN_UF_RANGES: List[Tuple[str, str, str]] = [
    ("AAA", "BEZ", "PR"),
    ("BFA", "GKI", "SP"),
    ("GKJ", "HOK", "MG"),
    ("HOL", "HQE", "MA"),
    ("HQF", "HTW", "MS"),
    ("HTX", "HZA", "CE"),
    ("HZB", "IAP", "SE"),
    ("IAQ", "JDO", "RS"),
    ("JDP", "JKR", "DF"),
    ("JKS", "JSZ", "BA"),
    ("JTA", "JWE", "PA"),
    ("JWF", "JXY", "AM"),
    ("JXZ", "KAU", "MT"),
    ("KAV", "KFC", "GO"),
    ("KFD", "KME", "PE"),
    ("KMF", "LVE", "RJ"),
    ("LVF", "LWQ", "PI"),
    ("LWR", "MMM", "SC"),
    ("MMN", "MOW", "PB"),
    ("MOX", "MTZ", "ES"),
    ("MUA", "MVK", "AL"),
    ("MVL", "MXG", "TO"),
    ("MXH", "MZM", "RN"),
    ("MZN", "NAG", "AC"),
    ("NAH", "NBA", "RR"),
    ("NBB", "NEH", "RO"),
    ("NEI", "NFB", "AP"),
    ("NFC", "NGZ", "GO"),
    ("NHA", "NHT", "MA"),
    ("NHU", "NIX", "PI"),
    ("NIY", "NJW", "MT"),
    ("NJX", "NLU", "GO"),
    ("NLV", "NMO", "AL"),
    ("NMP", "NNI", "MA"),
    ("NNJ", "NOH", "RN"),
    ("NOI", "NPB", "AM"),
    ("NPC", "NPQ", "MT"),
    ("NPR", "NQK", "PB"),
    ("NQL", "NRE", "CE"),
    ("NRF", "NSD", "MS"),
    ("NSE", "NTC", "PA"),
    ("NTD", "NTW", "BA"),
    ("NTX", "NUG", "MT"),
    ("NUH", "NUL", "RR"),
    ("NUM", "NVF", "CE"),
    ("NVG", "NVN", "SE"),
    ("NVO", "NWR", "GO"),
    ("NWS", "NXQ", "MA"),
    ("NXR", "NXT", "AC"),
    ("NXU", "NXW", "PE"),
    ("NXX", "NYG", "MG"),
    ("NYH", "NZZ", "BA"),
    ("OAA", "OAO", "AM"),
    ("OAP", "OBS", "MT"),
    ("OBT", "OCA", "PA"),
    ("OCB", "OCU", "CE"),
    ("OCV", "ODT", "ES"),
    ("ODU", "OEI", "PI"),
    ("OEJ", "OES", "SE"),
    ("OET", "OFH", "PB"),
    ("OFI", "OFW", "PA"),
    ("OFX", "OGG", "PB"),
    ("OGH", "OHA", "GO"),
    ("OHB", "OHK", "AL"),
    ("OHL", "OHW", "RO"),
    ("OHX", "OIQ", "CE"),
    ("OIR", "OJQ", "MA"),
    ("OJR", "OKC", "RN"),
    ("OKD", "OKH", "SC"),
    ("OKI", "OLG", "BA"),
    ("OLH", "OLN", "TO"),
    ("OLO", "OMH", "MG"),
    ("OMI", "OOF", "GO"),
    ("OOG", "OOU", "MS"),
    ("OOV", "ORC", "MG"),
    ("ORD", "ORM", "AL"),
    ("ORN", "OSV", "CE"),
    ("OSW", "OTZ", "PA"),
    ("OUA", "OUE", "PI"),
    ("OUF", "OVD", "BA"),
    ("OVE", "OVF", "ES"),
    ("OVG", "OVG", "AC"),
    ("OVH", "OVL", "ES"),
    ("OVM", "OVV", "DF"),
    ("OVW", "OVY", "PI"),
    ("OVZ", "OWG", "RN"),
    ("OWH", "OXK", "MG"),
    ("OXL", "OXL", "RO"),
    ("OXM", "OXM", "AM"),
    ("OXN", "OXN", "AL"),
    ("OXO", "OXO", "PB"),
    ("OXP", "OXP", "AC"),
    ("OXQ", "OXZ", "MA"),
    ("OYA", "OYC", "TO"),
    ("OYD", "OYK", "ES"),
    ("OYL", "OYZ", "PE"),
    ("OZA", "OZA", "CE"),
    ("OZB", "OZB", "SE"),
    ("OZC", "OZV", "BA"),
    ("OZW", "PBZ", "DF"),
    ("PCA", "PGZ", "PE"),
    ("PHA", "PHZ", "AM"),
    ("PIA", "PIZ", "PI"),
    ("PJA", "PLZ", "BA"),
    ("PMA", "POZ", "CE"),
    ("PPA", "PPZ", "ES"),
    ("PQA", "PRZ", "GO"),
    ("PSA", "PTZ", "MA"),
    ("PUA", "PZZ", "MG"),
    ("QAA", "QAZ", "MS"),
    ("QBA", "QCZ", "MT"),
    ("QDA", "QEZ", "PA"),
    ("QFA", "QFZ", "PB"),
    ("QGA", "QGZ", "RN"),
    ("QHA", "QJZ", "SC"),
    ("QKA", "QKM", "TO"),
    ("QKN", "QKZ", "SE"),
    ("QLA", "QLM", "AL"),
    ("QLN", "QLT", "AP"),
    ("QLU", "QLZ", "AC"),
    ("QMA", "QMP", "SE"),
    ("QMQ", "QQZ", "MG"),
    ("QRA", "QRA", "RO"),
    ("QRB", "QRM", "ES"),
    ("QRN", "QRZ", "PI"),
    ("QSA", "QSM", "PB"),
    ("QSN", "QSZ", "SP"),
    ("QTA", "QTJ", "RO"),
    ("QTK", "QTS", "SC"),
    ("QTT", "QTT", "AL"),
    ("QTU", "QTZ", "BA"),
    ("QUA", "QUZ", "MG"),
    ("QVA", "QVZ", "PA"),
    ("QWA", "QWF", "TO"),
    ("QWG", "QWL", "AL"),
    ("QWM", "QWQ", "AC"),
    ("QWR", "QXZ", "MG"),
    ("QYA", "QYZ", "PE"),
    ("QZA", "QZZ", "AM"),
    ("RAA", "RAJ", "SC"),
    ("RAK", "RAZ", "MT"),
    ("RBA", "RBJ", "ES"),
    ("RBK", "RCN", "GO"),
    ("RCO", "RDR", "BA"),
    ("RDS", "REB", "SC"),
    ("REC", "REZ", "DF"),
    ("RFA", "RGD", "MG"),
    ("RGE", "RGN", "RN"),
    ("RGO", "RGU", "AL"),
    ("RGV", "RHZ", "PR"),
    ("RIA", "RIN", "CE"),
    ("RIO", "RKV", "RJ"),
    ("RKW", "RLP", "SC"),
    ("RLQ", "RMC", "PB"),
    ("RMD", "RNZ", "MG"),
    ("ROA", "ROZ", "MA"),
    ("RPA", "RQL", "BA"),
    ("RQM", "RQV", "ES"),
    ("RQW", "RRH", "SE"),
    ("RRI", "RRZ", "MT"),
    ("RSA", "RSF", "TO"),
    ("RSG", "RST", "PI"),
    ("RSU", "RSZ", "RO"),
    ("RTA", "RVZ", "MG"),
    ("RWA", "RWJ", "MS"),
    ("RWK", "RXD", "PA"),
    ("RXE", "RXJ", "RR"),
    ("RXK", "RYI", "SC"),
    ("RYJ", "RYZ", "PB"),
    ("RZA", "RZD", "RR"),
    ("RZE", "RZZ", "PE"),
    ("SAA", "SAJ", "AL"),
    ("SAK", "SAM", "AP"),
    ("SAN", "SBV", "CE"),
    ("SBW", "SDS", "GO"),
    ("SDT", "ZZZ", "SP"),
]

# Modelos da frota brasileira para classificação Zero-Shot OpenCLIP
FLEET_MODEL_PROMPTS: List[Tuple[str, str, str]] = [
    ("FIAT/STRADA", "Picape / Utilitário", "a photo of a Fiat Strada pickup truck"),
    ("FIAT/TORO", "Picape / Utilitário", "a photo of a Fiat Toro pickup truck"),
    ("FIAT/ARGO", "Hatch / Passeio", "a photo of a Fiat Argo compact hatchback car"),
    ("FIAT/MOBI", "Hatch / Passeio", "a photo of a Fiat Mobi city hatchback car"),
    ("FIAT/FIORINO", "Furgão / Utilitário", "a photo of a Fiat Fiorino white cargo van"),
    ("FIAT/SIENA", "Sedan / Passeio", "a photo of a Fiat Grand Siena sedan car"),
    ("VW/GOL", "Hatch / Passeio", "a photo of a Volkswagen Gol hatchback car"),
    ("VW/POLO", "Hatch / Passeio", "a photo of a Volkswagen Polo hatchback car"),
    ("VW/VIRTUS", "Sedan / Passeio", "a photo of a Volkswagen Virtus sedan car"),
    ("VW/SAVEIRO", "Picape / Utilitário", "a photo of a Volkswagen Saveiro pickup truck"),
    ("VW/T-CROSS", "SUV / Crossover", "a photo of a Volkswagen T-Cross SUV"),
    ("CHEVROLET/ONIX", "Hatch / Passeio", "a photo of a Chevrolet Onix hatchback car"),
    ("CHEVROLET/PRISMA", "Sedan / Passeio", "a photo of a Chevrolet Onix Plus Prisma sedan car"),
    ("CHEVROLET/S10", "Picape / Utilitário", "a photo of a Chevrolet S10 pickup truck"),
    ("CHEVROLET/TRACKER", "SUV / Crossover", "a photo of a Chevrolet Tracker SUV"),
    ("HYUNDAI/HB20", "Hatch / Passeio", "a photo of a Hyundai HB20 hatchback car"),
    ("HYUNDAI/CRETA", "SUV / Crossover", "a photo of a Hyundai Creta SUV"),
    ("TOYOTA/HILUX", "Picape / Utilitário", "a photo of a Toyota Hilux pickup truck"),
    ("TOYOTA/COROLLA", "Sedan / Passeio", "a photo of a Toyota Corolla sedan car"),
    ("HONDA/FIT", "Hatch / Passeio", "a photo of a Honda Fit compact minivan hatchback"),
    ("HONDA/CIVIC", "Sedan / Passeio", "a photo of a Honda Civic sedan car"),
    ("HONDA/HR-V", "SUV / Crossover", "a photo of a Honda HR-V SUV"),
    ("RENAULT/KWID", "Hatch / Passeio", "a photo of a Renault Kwid compact car"),
    ("RENAULT/DUSTER", "SUV / Crossover", "a photo of a Renault Duster SUV"),
    ("JEEP/RENEGADE", "SUV / Crossover", "a photo of a Jeep Renegade boxy SUV"),
    ("JEEP/COMPASS", "SUV / Crossover", "a photo of a Jeep Compass SUV"),
    ("FORD/KA", "Hatch / Passeio", "a photo of a Ford Ka hatchback car"),
    ("FORD/RANGER", "Picape / Utilitário", "a photo of a Ford Ranger pickup truck"),
    ("NISSAN/KICKS", "SUV / Crossover", "a photo of a Nissan Kicks SUV"),
    ("HONDA/CG 160", "Motocicleta", "a photo of a motorcycle"),
    ("MERCEDES/CAMINHÃO", "Caminhão", "a photo of a commercial cargo truck"),
    ("MERCEDES/ÔNIBUS", "Ônibus", "a photo of a transit bus"),
]

CLIP_COLOR_PROMPTS: List[Tuple[str, str]] = [
    ("branca", "a white painted car vehicle"),
    ("preta", "a black painted car vehicle"),
    ("prata", "a silver or grey metallic car vehicle"),
    ("vermelha", "a red painted car vehicle"),
    ("azul", "a blue painted car vehicle"),
    ("amarela", "a yellow painted car vehicle"),
    ("verde", "a green painted car vehicle"),
    ("marrom", "a brown or beige car vehicle"),
]


def normalize_plate_key(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def infer_plate_uf(plate: Optional[str]) -> str:
    clean = normalize_plate_key(plate)
    if not VALID_BR_PLATE_RE.match(clean):
        return ""
    prefix = clean[:3]
    for start_p, end_p, uf in DENATRAN_UF_RANGES:
        if start_p <= prefix <= end_p:
            return uf
    return ""


def detect_badge_from_text(ocr_text: Optional[str]) -> Optional[Tuple[str, str]]:
    if not ocr_text:
        return None
    norm = re.sub(r"[^A-Z0-9 ]", " ", str(ocr_text).upper())
    compact = norm.replace(" ", "")
    for kw, model_lbl, cat_lbl in BADGE_MODEL_KEYWORDS:
        if kw in compact:
            return model_lbl, cat_lbl
    return None


class VehicleAttributeAnalyzer:
    def __init__(self) -> None:
        self._clip_model_embs: Optional[np.ndarray] = None
        self._clip_color_embs: Optional[np.ndarray] = None
        self._ort_session: Any = None
        self._ort_checked: bool = False
        self._veh_indices = sorted(IMAGENET_VEHICLE_MAP.keys())

    def _get_onnx_session(self) -> Any:
        if self._ort_checked:
            return self._ort_session
        self._ort_checked = True
        try:
            import onnxruntime as ort
            model_fp = Path(__file__).resolve().parent.parent / "data" / "models" / "mobilenetv2-12.onnx"
            if model_fp.is_file():
                self._ort_session = ort.InferenceSession(str(model_fp), providers=["CPUExecutionProvider"])
        except Exception:
            self._ort_session = None
        return self._ort_session

    def classify_with_onnx(self, veh_roi_bgr: np.ndarray, cls_id: int = 2) -> Optional[Tuple[str, str]]:
        sess = self._get_onnx_session()
        if sess is None or veh_roi_bgr is None or veh_roi_bgr.size == 0:
            return None
        try:
            rgb = cv2.cvtColor(veh_roi_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            norm = (resized - mean) / std
            chw = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]
            inp_name = sess.get_inputs()[0].name
            logits = sess.run(None, {inp_name: chw})[0].reshape(-1)

            if int(cls_id) == 3:
                return "MOTOCICLETA / URBANA", "Motocicleta"
            if int(cls_id) == 5:
                return "ÔNIBUS / COLETIVO", "Ônibus"

            sub_scores = [(idx, float(logits[idx])) for idx in self._veh_indices]
            best_idx, _ = max(sub_scores, key=lambda kv: kv[1])
            return IMAGENET_VEHICLE_MAP[best_idx]
        except Exception:
            return None

    def _extract_body_patches(
        self,
        clean_bgr: np.ndarray,
        plate_box: Optional[List[int]] = None,
        crop_offset: Optional[List[int]] = None,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        h, w = clean_bgr.shape[:2]
        patches: List[np.ndarray] = []

        if crop_offset is not None and len(crop_offset) == 4:
            vx1, vy1, vx2, vy2 = [int(v) for v in crop_offset]
            vx1, vy1 = max(0, vx1), max(0, vy1)
            vx2, vy2 = min(w, vx2), min(h, vy2)
            veh_roi = clean_bgr[vy1:vy2, vx1:vx2] if (vx2 - vx1 > 24 and vy2 - vy1 > 24) else clean_bgr
        elif plate_box is not None and len(plate_box) == 4:
            px1, py1, px2, py2 = [int(v) for v in plate_box]
            pw = max(16, px2 - px1)
            ph = max(10, py2 - py1)
            vx1 = max(0, int(px1 - pw * 2.2))
            vx2 = min(w, int(px2 + pw * 2.2))
            vy1 = max(0, int(py1 - ph * 6.5))
            vy2 = min(h, int(py2 + ph * 1.4))
            veh_roi = clean_bgr[vy1:vy2, vx1:vx2] if (vx2 - vx1 > 28 and vy2 - vy1 > 28) else clean_bgr
        else:
            veh_roi = clean_bgr

        if plate_box is not None and len(plate_box) == 4:
            px1, py1, px2, py2 = [int(v) for v in plate_box]
            pw = max(12, px2 - px1)
            ph = max(8, py2 - py1)
            if pw < w * 0.78 and ph < h * 0.65:
                uy1 = max(0, int(py1 - ph * 3.4))
                uy2 = max(0, int(py1 - ph * 0.25))
                ux1 = max(0, int(px1 - pw * 0.9))
                ux2 = min(w, int(px2 + pw * 0.9))
                if uy2 - uy1 >= 8 and ux2 - ux1 >= 14:
                    patches.append(clean_bgr[uy1:uy2, ux1:ux2])

                lx1 = max(0, int(px1 - pw * 1.35))
                lx2 = max(0, int(px1 - pw * 0.12))
                ly1 = max(0, int(py1 - ph * 0.8))
                ly2 = min(h, int(py2 + ph * 0.35))
                if ly2 - ly1 >= 8 and lx2 - lx1 >= 10:
                    patches.append(clean_bgr[ly1:ly2, lx1:lx2])

                rx1 = min(w, int(px2 + pw * 0.12))
                rx2 = min(w, int(px2 + pw * 1.35))
                ry1 = max(0, int(py1 - ph * 0.8))
                ry2 = min(h, int(py2 + ph * 0.35))
                if ry2 - ry1 >= 8 and rx2 - rx1 >= 10:
                    patches.append(clean_bgr[ry1:ry2, rx1:rx2])

        if not patches:
            vh, vw = veh_roi.shape[:2]
            cy1, cy2 = int(vh * 0.22), int(vh * 0.72)
            cx1, cx2 = int(vw * 0.18), int(vw * 0.82)
            if cy2 > cy1 and cx2 > cx1:
                patches.append(veh_roi[cy1:cy2, cx1:cx2])
            else:
                patches.append(veh_roi)

        return patches, veh_roi

    def detect_vehicle_color(
        self,
        clean_bgr: Optional[np.ndarray],
        plate_box: Optional[List[int]] = None,
        crop_offset: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        if clean_bgr is None or clean_bgr.size == 0:
            meta = VEHICLE_COLORS["indefinida"]
            return {
                "color_key": "indefinida",
                "vehicle_color": meta["label"],
                "color_hex": meta["hex"],
                "color_conf": 0.0,
            }

        patches, _ = self._extract_body_patches(clean_bgr, plate_box=plate_box, crop_offset=crop_offset)
        hsv_pixels = []
        lab_pixels = []
        for patch in patches:
            if patch is None or patch.size == 0:
                continue
            small = cv2.resize(patch, (min(96, max(8, patch.shape[1])), min(96, max(8, patch.shape[0]))), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
            hsv_pixels.append(hsv.reshape(-1, 3))
            lab_pixels.append(lab.reshape(-1, 3))

        if not hsv_pixels:
            meta = VEHICLE_COLORS["indefinida"]
            return {
                "color_key": "indefinida",
                "vehicle_color": meta["label"],
                "color_hex": meta["hex"],
                "color_conf": 0.0,
            }

        hsv_all = np.vstack(hsv_pixels)
        lab_all = np.vstack(lab_pixels)
        H = hsv_all[:, 0].astype(np.int32)
        S = hsv_all[:, 1].astype(np.int32)
        V = hsv_all[:, 2].astype(np.int32)
        L = lab_all[:, 0].astype(np.int32)
        A = lab_all[:, 1].astype(np.int32) - 128
        B = lab_all[:, 2].astype(np.int32) - 128
        chroma = np.sqrt(A * A + B * B)

        valid_mask = (V >= 18) & ~((V >= 250) & (S <= 10))
        if np.count_nonzero(valid_mask) > 40:
            H, S, V, L, chroma = H[valid_mask], S[valid_mask], V[valid_mask], L[valid_mask], chroma[valid_mask]

        total_px = max(1, len(H))
        votes: Dict[str, float] = {k: 0.0 for k in VEHICLE_COLORS.keys() if k != "indefinida"}

        chromatic = (S >= 58) & (V >= 42) & (chroma >= 14)
        if np.any(chromatic):
            Hc = H[chromatic]
            Sc = S[chromatic]
            Vc = V[chromatic]

            red_m = ((Hc <= 10) | (Hc >= 168)) & (Sc >= 70)
            brown_m = (Hc > 8) & (Hc <= 22) & (Vc < 150) & (Sc >= 55)
            yellow_m = (Hc > 16) & (Hc <= 36) & (Vc >= 135) & (Sc >= 68)
            green_m = (Hc > 36) & (Hc <= 86) & (Sc >= 58)
            blue_m = (Hc > 88) & (Hc <= 136) & (Sc >= 62)

            votes["vermelha"] += float(np.count_nonzero(red_m)) * 1.65
            votes["marrom"] += float(np.count_nonzero(brown_m)) * 1.45
            votes["amarela"] += float(np.count_nonzero(yellow_m)) * 1.55
            votes["verde"] += float(np.count_nonzero(green_m)) * 1.50
            votes["azul"] += float(np.count_nonzero(blue_m)) * 1.45

        achromatic = ~chromatic
        if np.any(achromatic):
            Sa = S[achromatic]
            Va = V[achromatic]
            La = L[achromatic]

            white_m = (Va >= 182) & (Sa <= 52) & (La >= 175)
            black_m = (Va <= 68) & (La <= 72)
            silver_m = (~white_m) & (~black_m) & (Sa <= 62)

            votes["branca"] += float(np.count_nonzero(white_m)) * 1.15
            votes["preta"] += float(np.count_nonzero(black_m)) * 1.05
            votes["prata"] += float(np.count_nonzero(silver_m)) * 1.00

        best_key = max(votes.items(), key=lambda kv: kv[1])[0]
        total_votes = sum(votes.values()) or float(total_px)
        conf = round(min(0.98, max(0.35, votes[best_key] / total_votes)), 2)
        meta = VEHICLE_COLORS[best_key]
        return {
            "color_key": best_key,
            "vehicle_color": meta["label"],
            "color_hex": meta["hex"],
            "color_conf": conf,
        }

    def detect_geometry_category(
        self,
        veh_roi: np.ndarray,
        cls_id: int = 2,
    ) -> Tuple[str, str]:
        if int(cls_id) == 3:
            return "MOTOCICLETA / URBANA", "Motocicleta"
        if int(cls_id) == 5:
            return "ÔNIBUS / COLETIVO", "Ônibus"
        if int(cls_id) == 7:
            return "CAMINHÃO / VUC", "Caminhão / Utilitário"

        h, w = veh_roi.shape[:2]
        ratio = float(w) / float(max(1, h))
        if ratio >= 1.52:
            return "PASSEIO / SEDAN", "Sedan / Passeio"
        if ratio <= 1.05:
            return "SUV / UTILITÁRIO", "SUV / Crossover"
        return "PASSEIO / HATCH", "Hatch / Passeio"

    def classify_with_clip_mainthread(self, object_finder: Any, veh_roi_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """Executa exclusivamente na thread principal (via add_to_queue) para segurança OpenCL."""
        if not getattr(object_finder, "clip", False) or getattr(object_finder, "model", None) is None:
            return None
        from tinygrad import Tensor
        from utils.helpers import jit_infer

        model = object_finder.model
        if self._clip_model_embs is None:
            m_list = []
            for _, _, prompt in FLEET_MODEL_PROMPTS:
                emb_t = model._encode_text(prompt, realize=True)
                m_list.append(emb_t.reshape(-1))
            self._clip_model_embs = np.vstack(m_list)

        if self._clip_color_embs is None:
            c_list = []
            for _, prompt in CLIP_COLOR_PROMPTS:
                emb_t = model._encode_text(prompt, realize=True)
                c_list.append(emb_t.reshape(-1))
            self._clip_color_embs = np.vstack(c_list)

        rgb = cv2.cvtColor(veh_roi_bgr, cv2.COLOR_BGR2RGB)
        inp = object_finder.preprocess(rgb)
        img_emb = jit_infer(model.precompute_embedding, Tensor(inp).unsqueeze(0), jit_cache=object_finder.jit_cache).numpy()
        img_vec = img_emb.reshape(1, -1)

        model_sims = (img_vec @ self._clip_model_embs.T).reshape(-1)
        best_m_idx = int(np.argmax(model_sims))
        best_model_name, best_model_cat, _ = FLEET_MODEL_PROMPTS[best_m_idx]

        color_sims = (img_vec @ self._clip_color_embs.T).reshape(-1)
        best_c_idx = int(np.argmax(color_sims))
        best_clip_color = CLIP_COLOR_PROMPTS[best_c_idx][0]

        return {
            "vehicle_model": best_model_name,
            "vehicle_type": best_model_cat,
            "model_score": round(float(model_sims[best_m_idx]), 3),
            "clip_color_key": best_clip_color,
            "clip_color_score": round(float(color_sims[best_c_idx]), 3),
        }

    def analyze(
        self,
        clean_bgr: Optional[np.ndarray],
        cls_id: int = 2,
        plate_code: Optional[str] = None,
        raw_ocr_text: Optional[str] = None,
        plate_box: Optional[List[int]] = None,
        crop_offset: Optional[List[int]] = None,
        registry_entry: Optional[Dict[str, Any]] = None,
        clip_runner: Optional[Callable[[np.ndarray], Optional[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        color_info = self.detect_vehicle_color(clean_bgr, plate_box=plate_box, crop_offset=crop_offset)
        _, veh_roi = (
            self._extract_body_patches(clean_bgr, plate_box=plate_box, crop_offset=crop_offset)
            if (clean_bgr is not None and clean_bgr.size > 0)
            else ([], np.zeros((32, 32, 3), dtype=np.uint8))
        )
        vehicle_model, vehicle_type = self.detect_geometry_category(veh_roi, cls_id=cls_id)

        # Classificador neural ONNX MobileNetV2 (CPU ~3ms)
        onnx_pred = self.classify_with_onnx(veh_roi, cls_id=cls_id)
        if onnx_pred is not None:
            vehicle_model, vehicle_type = onnx_pred

        # Emblema de marca/modelo detectado via OCR na cena
        badge_hit = detect_badge_from_text(raw_ocr_text)
        if badge_hit is not None:
            vehicle_model, vehicle_type = badge_hit

        # OpenCLIP Zero-Shot quando habilitado nas configurações globais
        if clip_runner is not None and clean_bgr is not None and clean_bgr.size > 0 and veh_roi.shape[0] >= 36 and veh_roi.shape[1] >= 36:
            try:
                clip_res = clip_runner(veh_roi)
                if isinstance(clip_res, dict):
                    if int(cls_id) == 3:
                        vehicle_model = "HONDA/CG 160"
                        vehicle_type = "Motocicleta"
                    elif int(cls_id) == 5:
                        vehicle_model = "MERCEDES/ÔNIBUS"
                        vehicle_type = "Ônibus"
                    else:
                        vehicle_model = clip_res["vehicle_model"]
                        vehicle_type = clip_res["vehicle_type"]

                    clip_ck = clip_res.get("clip_color_key")
                    if clip_ck in VEHICLE_COLORS and color_info["color_conf"] < 0.48 and clip_res.get("clip_color_score", 0) >= 0.23:
                        cmeta = VEHICLE_COLORS[clip_ck]
                        color_info["color_key"] = clip_ck
                        color_info["vehicle_color"] = cmeta["label"]
                        color_info["color_hex"] = cmeta["hex"]
            except Exception:
                pass

        uf = infer_plate_uf(plate_code)
        year_str = ""
        list_status = "sem_restricao"
        notes = ""

        if isinstance(registry_entry, dict):
            reg_color = str(registry_entry.get("color_key") or "").strip().lower()
            if reg_color in VEHICLE_COLORS and reg_color != "indefinida":
                cmeta = VEHICLE_COLORS[reg_color]
                color_info["color_key"] = reg_color
                color_info["vehicle_color"] = cmeta["label"]
                color_info["color_hex"] = cmeta["hex"]
                color_info["color_conf"] = 1.0
            if registry_entry.get("vehicle_model"):
                vehicle_model = str(registry_entry["vehicle_model"]).strip().upper()
            if registry_entry.get("vehicle_type"):
                vehicle_type = str(registry_entry["vehicle_type"]).strip()
            if registry_entry.get("vehicle_year"):
                year_str = str(registry_entry["vehicle_year"]).strip()
            if registry_entry.get("vehicle_uf"):
                uf = str(registry_entry["vehicle_uf"]).strip().upper()
            if registry_entry.get("list_status") in ("sem_restricao", "blacklist", "whitelist"):
                list_status = str(registry_entry["list_status"])
            if registry_entry.get("notes"):
                notes = str(registry_entry["notes"])

        return {
            "color_key": color_info["color_key"],
            "vehicle_color": color_info["vehicle_color"],
            "color_hex": color_info["color_hex"],
            "color_conf": color_info["color_conf"],
            "vehicle_model": vehicle_model,
            "vehicle_type": vehicle_type,
            "vehicle_year": year_str,
            "vehicle_uf": uf,
            "list_status": list_status,
            "notes": notes,
        }

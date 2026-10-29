# language: Python, file: protocols/pumatronix.py, target: Windows 11 / Cross-platform
import base64
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Generator, Optional, Tuple
import cv2
import numpy as np


FLASH_MODES = {
    1: "Desativado",
    2: "Modo unico",
    3: "Modo unico com delay",
    4: "Continuo",
    5: "Automatico",
    6: "Automatico com delay",
    7: "Continuo no modo Night",
}

SUPPORTED_RESOLUTIONS = [
    "160x120",
    "240x180",
    "320x240",
    "480x360",
    "640x480",
    "752x480",
    "1280x720",
    "1280x960",
    "1636x1220",
]


def parse_pumatronix_uri(uri: str) -> Dict[str, object]:
    """
    Parses either:
      - pumatronix://[user:pass@]ip[:port][?Quality=80&Resolution=640x480&FrameRate=15]
      - http://[user:pass@]ip[:port]/api/mjpegvideo.cgi?Quality=80&Resolution=640x480&FrameRate=0
    """
    parsed = urllib.parse.urlparse(uri)
    qs = urllib.parse.parse_qs(parsed.query)
    quality = int(qs.get("Quality", qs.get("quality", qs.get("qualidade", ["80"])))[0])
    resolution = qs.get("Resolution", qs.get("resolution", ["640x480"]))[0]
    framerate = int(qs.get("FrameRate", qs.get("framerate", ["15"]))[0])
    flash_mode = int(qs.get("ModoFlash", ["0"])[0]) if "ModoFlash" in qs else None
    watchdog = qs.get("watchdog", ["false"])[0].lower() in ("1", "true", "yes")

    return {
        "ip": parsed.hostname or "",
        "port": parsed.port or 80,
        "user": urllib.parse.unquote(parsed.username) if parsed.username else None,
        "password": urllib.parse.unquote(parsed.password) if parsed.password else None,
        "quality": max(1, min(100, quality)),
        "resolution": resolution,
        "framerate": framerate,
        "flash_mode": flash_mode,
        "watchdog": watchdog,
    }


def is_pumatronix_source(src: str) -> bool:
    if not isinstance(src, str):
        return False
    lower = src.lower()
    return (
        lower.startswith("pumatronix://")
        or "/api/mjpegvideo.cgi" in lower
        or "/api/snapshot.cgi" in lower
        or "/api/lastframe.cgi" in lower
    )


class PumatronixClient:
    """Client for Pumatronix ITSCAM HTTP CGI APIs (mjpegvideo, snapshot, trigger, config, watchdog)."""

    def __init__(
        self,
        ip: str,
        port: int = 80,
        user: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 8.0,
    ):
        self.ip = ip
        self.port = int(port or 80)
        self.user = user
        self.password = password
        self.timeout = timeout

    def _make_request(self, path: str) -> urllib.request.Request:
        url = f"http://{self.ip}:{self.port}{path}"
        req = urllib.request.Request(url, headers={"User-Agent": "Clearcam-ITSCAM/1.0"})
        if self.user:
            pwd = self.password or ""
            token = base64.b64encode(f"{self.user}:{pwd}".encode("utf-8")).decode("ascii")
            req.add_header("Authorization", f"Basic {token}")
        return req

    def mjpeg_url(
        self,
        quality: int = 80,
        resolution: str = "640x480",
        framerate: int = 15,
        include_auth: bool = True,
    ) -> str:
        auth = ""
        if include_auth and self.user:
            u = urllib.parse.quote(self.user, safe="")
            p = urllib.parse.quote(self.password or "", safe="")
            auth = f"{u}:{p}@"
        q = max(1, min(100, int(quality)))
        return (
            f"http://{auth}{self.ip}:{self.port}/api/mjpegvideo.cgi"
            f"?Quality={q}&Resolution={resolution}&FrameRate={int(framerate)}"
        )

    def get_bytes(self, path: str, timeout: Optional[float] = None) -> bytes:
        req = self._make_request(path)
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            return resp.read()

    def snapshot(self, quality: int = 90, resolution: Optional[str] = None) -> bytes:
        q = max(1, min(100, int(quality)))
        path = f"/api/snapshot.cgi?qualidade={q}"
        if resolution:
            path += f"&Resolution={urllib.parse.quote(resolution)}"
        return self.get_bytes(path)

    def snapshot_bgr(self, quality: int = 90, resolution: Optional[str] = None) -> Optional[np.ndarray]:
        raw = self.snapshot(quality=quality, resolution=resolution)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def last_frame(self) -> bytes:
        return self.get_bytes("/api/lastFrame.cgi")

    def trigger_snapshot(self, timeout: float = 15.0) -> bytes:
        return self.get_bytes("/api/trigger.cgi", timeout=timeout)

    def get_config(self, param: str = "tudo") -> Dict[str, str]:
        raw = self.get_bytes(f"/api/config.cgi?{urllib.parse.quote(param)}")
        text = raw.decode("latin-1", errors="replace")
        out: Dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
        if not out and text.strip():
            out["_raw"] = text.strip()
        return out

    def set_config(self, params: Dict[str, object]) -> str:
        query = "&".join(
            f"{urllib.parse.quote(str(k))}={urllib.parse.quote(str(v))}"
            for k, v in params.items()
        )
        raw = self.get_bytes(f"/api/config.cgi?{query}")
        return raw.decode("latin-1", errors="replace").strip()

    def set_flash(self, modo_flash: int, delay_flash_steps: int = 5000) -> Dict[str, str]:
        res = {"ModoFlash": self.set_config({"ModoFlash": int(modo_flash)})}
        if int(modo_flash) in (3, 6):
            res["DelayFlash"] = self.set_config({"DelayFlash": int(delay_flash_steps)})
        return res

    def get_audit_configs(self) -> str:
        raw = self.get_bytes("/api/configs.cgi")
        return raw.decode("latin-1", errors="replace")

    def ping_watchdog(self) -> str:
        raw = self.get_bytes("/api/watchdog.cgi", timeout=5.0)
        return raw.decode("latin-1", errors="replace").strip()

    def reboot(self) -> str:
        raw = self.get_bytes("/api/reboot.cgi", timeout=5.0)
        return raw.decode("latin-1", errors="replace").strip()

    def iter_mjpeg_frames(
        self,
        quality: int = 80,
        resolution: str = "640x480",
        framerate: int = 15,
    ) -> Generator[np.ndarray, None, None]:
        """Reads multipart/x-mixed-replace JPEG frames directly from /api/mjpegvideo.cgi."""
        q = max(1, min(100, int(quality)))
        path = f"/api/mjpegvideo.cgi?Quality={q}&Resolution={resolution}&FrameRate={int(framerate)}"
        req = self._make_request(path)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            buf = bytearray()
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                buf.extend(chunk)
                start = buf.find(b"\xff\xd8")
                if start != -1:
                    end = buf.find(b"\xff\xd9", start + 2)
                    if end != -1:
                        jpg_bytes = bytes(buf[start : end + 2])
                        del buf[: end + 2]
                        frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            yield frame
                elif len(buf) > 2 * 1024 * 1024:
                    buf.clear()


class PumatronixWatchdogManager:
    """Maintains optional 60-second /api/watchdog.cgi heartbeats per Pumatronix camera."""

    def __init__(self):
        self._stops: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def start(self, cam_name: str, client: PumatronixClient, interval_sec: float = 60.0):
        self.stop(cam_name)
        stop_ev = threading.Event()
        with self._lock:
            self._stops[cam_name] = stop_ev

        def _loop():
            while not stop_ev.is_set():
                try:
                    client.ping_watchdog()
                except Exception as exc:
                    print(f"[Pumatronix Watchdog] {cam_name} ({client.ip}): {exc}")
                stop_ev.wait(interval_sec)

        threading.Thread(target=_loop, daemon=True, name=f"PumaWatchdog-{cam_name}").start()

    def stop(self, cam_name: str):
        with self._lock:
            ev = self._stops.pop(cam_name, None)
        if ev:
            ev.set()

    def is_running(self, cam_name: str) -> bool:
        with self._lock:
            return cam_name in self._stops and not self._stops[cam_name].is_set()

# language: Python, file: protocols/isup.py, target: Windows 11 / Cross-platform
import ctypes
import os
import re
import socket
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth, HTTPBasicAuth


def is_isup_or_isapi_source(src: str) -> bool:
    if not isinstance(src, str):
        return False
    lower = src.lower()
    return lower.startswith("isup://") or lower.startswith("isapi://")


def parse_isup_uri(uri: str) -> Dict[str, object]:
    """
    Supported URI formats:
      - isup://device_id:isup_key@0.0.0.0:7660?cam_ip=192.168.1.49&user=admin&pass=xxx&channel=1
      - isapi://admin:pass@192.168.1.49:80/101
    """
    parsed = urllib.parse.urlparse(uri)
    qs = urllib.parse.parse_qs(parsed.query)
    scheme = parsed.scheme.lower()
    path_clean = (parsed.path or "").strip("/")
    channel = int(qs.get("channel", [path_clean if path_clean.isdigit() else "101"])[0])
    if channel < 100:
        channel = channel * 100 + 1

    if scheme == "isapi":
        return {
            "mode": "isapi",
            "host": parsed.hostname or "",
            "port": parsed.port or 80,
            "user": urllib.parse.unquote(parsed.username) if parsed.username else "admin",
            "password": urllib.parse.unquote(parsed.password) if parsed.password else "",
            "channel": channel,
            "device_id": qs.get("device_id", [""])[0],
            "isup_key": qs.get("key", [""])[0],
        }

    return {
        "mode": "isup",
        "listen_host": parsed.hostname or "0.0.0.0",
        "listen_port": parsed.port or 7660,
        "device_id": urllib.parse.unquote(parsed.username) if parsed.username else qs.get("device_id", [""])[0],
        "isup_key": urllib.parse.unquote(parsed.password) if parsed.password else qs.get("key", [""])[0],
        "cam_ip": qs.get("cam_ip", qs.get("ip", [""]))[0],
        "user": qs.get("user", ["admin"])[0],
        "password": qs.get("pass", qs.get("password", [""]))[0],
        "http_port": int(qs.get("http_port", ["80"])[0]),
        "rtsp_port": int(qs.get("rtsp_port", ["554"])[0]),
        "channel": channel,
    }


class HikvisionISAPIClient:
    """HTTP Digest client for Hikvision ISAPI (deviceInfo, snapshots, ANPR/LPR plates, RTSP bridge)."""

    def __init__(self, host: str, port: int = 80, user: str = "admin", password: str = "", timeout: float = 6.0):
        self.host = host
        self.port = int(port or 80)
        self.user = user or "admin"
        self.password = password or ""
        self.timeout = timeout
        self.base_url = f"http://{self.host}:{self.port}"

    def _get(self, path: str, stream: bool = False) -> requests.Response:
        url = f"{self.base_url}{path}"
        resp = requests.get(
            url,
            auth=HTTPDigestAuth(self.user, self.password),
            timeout=self.timeout,
            stream=stream,
        )
        if resp.status_code == 401:
            resp = requests.get(
                url,
                auth=HTTPBasicAuth(self.user, self.password),
                timeout=self.timeout,
                stream=stream,
            )
        resp.raise_for_status()
        return resp

    def get_device_info(self) -> Dict[str, str]:
        resp = self._get("/ISAPI/System/deviceInfo")
        root = ET.fromstring(resp.content)
        info = {}
        for el in root.iter():
            tag = el.tag.split("}")[-1]
            if el.text and el.text.strip() and tag not in ("DeviceInfo",):
                info[tag] = el.text.strip()
        return info

    def snapshot_bgr(self, channel: int = 101) -> Optional[np.ndarray]:
        resp = self._get(f"/ISAPI/Streaming/channels/{int(channel)}/picture")
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def rtsp_url(self, channel: int = 101, rtsp_port: int = 554) -> str:
        u = urllib.parse.quote(self.user, safe="")
        p = urllib.parse.quote(self.password, safe="")
        auth = f"{u}:{p}@" if self.user else ""
        return f"rtsp://{auth}{self.host}:{int(rtsp_port)}/Streaming/Channels/{int(channel)}"


class ISUPGateway:
    """
    Hikvision ISUP 5.0 (EHome) Gateway & Listener on TCP/UDP 7660:
      1. Checks `sdk/isup/HCEHomeCMS.dll` (or system PATH) to initialize official Hikvision EHome CMS SDK if present.
      2. Otherwise runs a pure-Python TCP + UDP listener on port 7660 to capture incoming ISUP camera registrations,
         extract source IP/session metadata, and bridge to RTSP / ISAPI when routable.
    """

    def __init__(self, sdk_dir: Optional[Path] = None):
        self.sdk_dir = sdk_dir or (Path(__file__).parent.parent / "sdk" / "isup")
        self.sdk_dir.mkdir(parents=True, exist_ok=True)
        self.sdk_loaded = False
        self.cms_dll = None
        self.listening_port: Optional[int] = None
        self.isup_key: str = ""
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.registered_devices: Dict[str, Dict[str, object]] = {}
        self._try_load_hik_sdk()

    def _try_load_hik_sdk(self):
        candidates = [
            self.sdk_dir / "HCEHomeCMS.dll",
            self.sdk_dir / "libHCEHOMESDK.so",
        ]
        for cand in candidates:
            if cand.exists():
                try:
                    if os.name == "nt":
                        os.add_dll_directory(str(self.sdk_dir.resolve()))
                    self.cms_dll = ctypes.CDLL(str(cand.resolve()))
                    if hasattr(self.cms_dll, "NET_ECMS_Init"):
                        self.cms_dll.NET_ECMS_Init()
                        self.sdk_loaded = True
                        print(f"[ISUP] Loaded official Hikvision EHome CMS SDK: {cand}")
                        return
                except Exception as exc:
                    print(f"[ISUP] Failed loading SDK {cand}: {exc}")

    def start_listener(self, host: str = "0.0.0.0", port: int = 7660, isup_key: str = "") -> Dict[str, object]:
        self.isup_key = isup_key or self.isup_key
        if self.listening_port == port and not self._stop_event.is_set():
            return self.status()

        self.stop_listener()
        self._stop_event.clear()
        self.listening_port = int(port)

        threading.Thread(
            target=self._udp_listener,
            args=(host, int(port)),
            daemon=True,
            name=f"ISUP-UDP-{port}",
        ).start()
        threading.Thread(
            target=self._tcp_listener,
            args=(host, int(port)),
            daemon=True,
            name=f"ISUP-TCP-{port}",
        ).start()
        return self.status()

    def stop_listener(self):
        self._stop_event.set()
        self.listening_port = None

    def _record_packet(self, proto: str, addr: tuple, payload: bytes):
        ip, remote_port = addr[0], addr[1]
        ascii_tokens = re.findall(rb"[A-Za-z0-9_\-\.]{4,32}", payload)
        decoded_tokens = [t.decode("ascii", errors="ignore") for t in ascii_tokens[:6]]
        dev_id = decoded_tokens[0] if decoded_tokens else f"isup_{ip.replace('.', '_')}"
        with self._lock:
            self.registered_devices[ip] = {
                "ip": ip,
                "remote_port": remote_port,
                "transport": proto,
                "device_hint": dev_id,
                "tokens": decoded_tokens,
                "bytes": len(payload),
                "hex_head": payload[:32].hex(),
                "last_seen": time.strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp": time.time(),
            }

    def _udp_listener(self, host: str, port: int):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.5)
        try:
            sock.bind((host, port))
        except OSError as exc:
            print(f"[ISUP] UDP {host}:{port} bind warning: {exc}")
            sock.close()
            return
        while not self._stop_event.is_set():
            try:
                data, addr = sock.recvfrom(8192)
                if data:
                    self._record_packet("UDP", addr, data)
            except socket.timeout:
                continue
            except Exception:
                break
        sock.close()

    def _tcp_listener(self, host: str, port: int):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.settimeout(1.5)
        try:
            srv.bind((host, port))
            srv.listen(16)
        except OSError as exc:
            print(f"[ISUP] TCP {host}:{port} bind warning: {exc}")
            srv.close()
            return
        while not self._stop_event.is_set():
            try:
                conn, addr = srv.accept()
                threading.Thread(target=self._handle_tcp_client, args=(conn, addr), daemon=True).start()
            except socket.timeout:
                continue
            except Exception:
                break
        srv.close()

    def _handle_tcp_client(self, conn: socket.socket, addr: tuple):
        conn.settimeout(5.0)
        try:
            data = conn.recv(8192)
            if data:
                self._record_packet("TCP", addr, data)
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def resolve_stream_source(self, uri: str) -> Optional[str]:
        """
        Resolves an isup:// or isapi:// URI into an active RTSP stream URL when the camera IP is known
        (either passed via ?cam_ip=... or auto-detected via incoming ISUP registration on port 7660).
        """
        cfg = parse_isup_uri(uri)
        if cfg["mode"] == "isapi":
            client = HikvisionISAPIClient(
                host=str(cfg["host"]),
                port=int(cfg["port"]),
                user=str(cfg["user"]),
                password=str(cfg["password"]),
            )
            return client.rtsp_url(channel=int(cfg["channel"]))

        # Ensure ISUP listener is active
        self.start_listener(
            host=str(cfg["listen_host"]),
            port=int(cfg["listen_port"]),
            isup_key=str(cfg["isup_key"]),
        )
        cam_ip = str(cfg.get("cam_ip") or "")
        dev_id = str(cfg.get("device_id") or "")
        if not cam_ip:
            with self._lock:
                for ip, meta in self.registered_devices.items():
                    if not dev_id or dev_id in str(meta.get("tokens", "")) or dev_id == meta.get("device_hint"):
                        cam_ip = ip
                        break
        if cam_ip:
            u = urllib.parse.quote(str(cfg["user"]), safe="")
            p = urllib.parse.quote(str(cfg["password"]), safe="")
            auth = f"{u}:{p}@" if cfg.get("user") else ""
            return f"rtsp://{auth}{cam_ip}:{int(cfg['rtsp_port'])}/Streaming/Channels/{int(cfg['channel'])}"
        return None

    def status(self) -> Dict[str, object]:
        with self._lock:
            devs = list(self.registered_devices.values())
        return {
            "listening": self.listening_port is not None and not self._stop_event.is_set(),
            "port": self.listening_port,
            "sdk_loaded": self.sdk_loaded,
            "sdk_dir": str(self.sdk_dir),
            "devices": devs,
        }


isup_gateway = ISUPGateway()

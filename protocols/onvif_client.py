# language: Python, file: protocols/onvif_client.py, target: Windows 11 / Cross-platform
import base64
import datetime
import hashlib
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
import requests
from requests.auth import HTTPDigestAuth, HTTPBasicAuth


def is_onvif_source(src: str) -> bool:
    return isinstance(src, str) and src.lower().startswith("onvif://")


def parse_onvif_uri(uri: str) -> Dict[str, object]:
    """
    Format: onvif://[user:pass@]host[:port][/profile_index_or_token][?subtype=0&channel=1]
    """
    parsed = urllib.parse.urlparse(uri)
    qs = urllib.parse.parse_qs(parsed.query)
    path_clean = (parsed.path or "").strip("/")
    if path_clean.lower() == "onvif/device_service":
        path_clean = ""
    profile_idx = int(qs.get("profile", [path_clean if path_clean.isdigit() else "0"])[0])
    channel = int(qs.get("channel", ["1"])[0])
    return {
        "host": parsed.hostname or "",
        "port": parsed.port or 80,
        "user": urllib.parse.unquote(parsed.username) if parsed.username else "",
        "password": urllib.parse.unquote(parsed.password) if parsed.password else "",
        "profile_index": profile_idx,
        "profile_token": None if (not path_clean or path_clean.isdigit()) else path_clean,
        "channel": channel,
    }


class ONVIFDirectClient:
    """
    Zero-WSDL SOAP 1.2 ONVIF client with WS-UsernameToken PasswordDigest + HTTP Digest fallback.
    Works reliably on Windows with Intelbras, Hikvision, Dahua, Axis, and generic Profile S/T cameras.
    """

    def __init__(self, host: str, port: int = 80, user: str = "admin", password: str = "", timeout: float = 6.0):
        self.host = host
        self.port = int(port or 80)
        self.user = user or "admin"
        self.password = password or ""
        self.timeout = timeout
        self.device_url = f"http://{self.host}:{self.port}/onvif/device_service"
        self.media_url: Optional[str] = None
        self.ptz_url: Optional[str] = None
        self._time_offset = datetime.timedelta(0)

    def _wsse_header(self) -> str:
        if not self.user:
            return ""
        nonce_raw = os.urandom(16)
        created = (datetime.datetime.now(datetime.timezone.utc) + self._time_offset).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        digest = base64.b64encode(
            hashlib.sha1(nonce_raw + created.encode("utf-8") + self.password.encode("utf-8")).digest()
        ).decode("ascii")
        nonce_b64 = base64.b64encode(nonce_raw).decode("ascii")
        return f"""
  <s:Header>
    <Security s:mustUnderstand="1" xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd">
      <UsernameToken>
        <Username>{self._xml_escape(self.user)}</Username>
        <Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{digest}</Password>
        <Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">{nonce_b64}</Nonce>
        <Created xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">{created}</Created>
      </UsernameToken>
    </Security>
  </s:Header>"""

    @staticmethod
    def _xml_escape(val: str) -> str:
        return (
            val.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    def _soap_call(self, url: str, body_xml: str, auth_header: bool = True) -> ET.Element:
        envelope = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tds="http://www.onvif.org/ver10/device/wsdl"
            xmlns:trt="http://www.onvif.org/ver10/media/wsdl"
            xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl"
            xmlns:tt="http://www.onvif.org/ver10/schema">{self._wsse_header() if auth_header else ""}
  <s:Body>
    {body_xml}
  </s:Body>
</s:Envelope>"""
        headers = {"Content-Type": "application/soap+xml; charset=utf-8"}
        resp = requests.post(url, data=envelope.encode("utf-8"), headers=headers, timeout=self.timeout)
        if resp.status_code == 401 and self.user:
            resp = requests.post(
                url,
                data=envelope.encode("utf-8"),
                headers=headers,
                auth=HTTPDigestAuth(self.user, self.password),
                timeout=self.timeout,
            )
            if resp.status_code == 401:
                resp = requests.post(
                    url,
                    data=envelope.encode("utf-8"),
                    headers=headers,
                    auth=HTTPBasicAuth(self.user, self.password),
                    timeout=self.timeout,
                )
        if resp.status_code >= 400 and b"NotAuthorized" not in resp.content and b"Fault" not in resp.content:
            resp.raise_for_status()
        root = ET.fromstring(resp.content)
        fault = root.find(".//{http://www.w3.org/2003/05/soap-envelope}Fault")
        if fault is not None:
            reason_el = fault.find(".//{http://www.w3.org/2003/05/soap-envelope}Text")
            reason = reason_el.text if reason_el is not None else ET.tostring(fault, encoding="unicode")
            raise RuntimeError(f"ONVIF SOAP Fault: {reason}")
        return root

    def sync_time(self):
        try:
            root = self._soap_call(self.device_url, "<tds:GetSystemDateAndTime/>", auth_header=False)
            ns = "{http://www.onvif.org/ver10/schema}"
            utc = root.find(f".//{ns}UTCDateTime")
            if utc is not None:
                y = int(utc.findtext(f".//{ns}Year", "0"))
                m = int(utc.findtext(f".//{ns}Month", "0"))
                d = int(utc.findtext(f".//{ns}Day", "0"))
                hh = int(utc.findtext(f".//{ns}Hour", "0"))
                mm = int(utc.findtext(f".//{ns}Minute", "0"))
                ss = int(utc.findtext(f".//{ns}Second", "0"))
                if y > 2000:
                    cam_dt = datetime.datetime(y, m, d, hh, mm, ss, tzinfo=datetime.timezone.utc)
                    self._time_offset = cam_dt - datetime.datetime.now(datetime.timezone.utc)
        except Exception:
            pass

    def get_device_info(self) -> Dict[str, str]:
        self.sync_time()
        root = self._soap_call(self.device_url, "<tds:GetDeviceInformation/>")
        info = {}
        for field in ("Manufacturer", "Model", "FirmwareVersion", "SerialNumber", "HardwareId"):
            el = root.find(f".//{{http://www.onvif.org/ver10/device/wsdl}}{field}")
            if el is not None and el.text:
                info[field] = el.text.strip()
        return info

    def discover_service_urls(self) -> Tuple[str, Optional[str]]:
        self.sync_time()
        try:
            root = self._soap_call(self.device_url, "<tds:GetCapabilities><tds:Category>All</tds:Category></tds:GetCapabilities>")
            ns = "{http://www.onvif.org/ver10/schema}"
            media_el = root.find(f".//{ns}Media/{ns}XAddr")
            ptz_el = root.find(f".//{ns}PTZ/{ns}XAddr")
            if media_el is not None and media_el.text:
                self.media_url = self._rewrite_host(media_el.text.strip())
            if ptz_el is not None and ptz_el.text:
                self.ptz_url = self._rewrite_host(ptz_el.text.strip())
        except Exception:
            pass
        if not self.media_url:
            self.media_url = f"http://{self.host}:{self.port}/onvif/media_service"
        return self.media_url, self.ptz_url

    def _rewrite_host(self, url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        netloc = f"{self.host}:{parsed.port}" if parsed.port else (
            f"{self.host}:{self.port}" if self.port != 80 else self.host
        )
        return urllib.parse.urlunparse(parsed._replace(netloc=netloc))

    def get_profiles(self) -> List[Dict[str, object]]:
        if not self.media_url:
            self.discover_service_urls()
        root = self._soap_call(self.media_url, "<trt:GetProfiles/>")
        ns_trt = "{http://www.onvif.org/ver10/media/wsdl}"
        ns_tt = "{http://www.onvif.org/ver10/schema}"
        profiles = []
        for prof in root.findall(f".//{ns_trt}Profiles"):
            token = prof.attrib.get("token", "")
            name = prof.findtext(f"{ns_tt}Name", token)
            w = prof.findtext(f".//{ns_tt}VideoEncoderConfiguration/{ns_tt}Resolution/{ns_tt}Width", "")
            h = prof.findtext(f".//{ns_tt}VideoEncoderConfiguration/{ns_tt}Resolution/{ns_tt}Height", "")
            enc = prof.findtext(f".//{ns_tt}VideoEncoderConfiguration/{ns_tt}Encoding", "")
            profiles.append({
                "token": token,
                "name": name,
                "encoding": enc,
                "resolution": f"{w}x{h}" if w and h else "",
            })
        return profiles

    def get_stream_uri(self, profile_token: Optional[str] = None, profile_index: int = 0) -> str:
        profiles = self.get_profiles()
        if not profiles:
            raise RuntimeError("No ONVIF media profiles returned by camera")
        token = profile_token
        if not token:
            idx = max(0, min(profile_index, len(profiles) - 1))
            token = str(profiles[idx]["token"])
        body = f"""<trt:GetStreamUri>
      <trt:StreamSetup>
        <tt:Stream>RTP-Unicast</tt:Stream>
        <tt:Transport><tt:Protocol>RTSP</tt:Protocol></tt:Transport>
      </trt:StreamSetup>
      <trt:ProfileToken>{self._xml_escape(token)}</trt:ProfileToken>
    </trt:GetStreamUri>"""
        root = self._soap_call(self.media_url, body)
        uri_el = root.find(".//{http://www.onvif.org/ver10/schema}Uri")
        if uri_el is None or not uri_el.text:
            raise RuntimeError("ONVIF GetStreamUri returned empty URI")
        raw_uri = uri_el.text.strip()
        return self._inject_rtsp_credentials(raw_uri)

    def get_snapshot_uri(self, profile_token: Optional[str] = None, profile_index: int = 0) -> str:
        profiles = self.get_profiles()
        if not profiles:
            raise RuntimeError("No ONVIF media profiles returned")
        token = profile_token or str(profiles[max(0, min(profile_index, len(profiles) - 1))]["token"])
        body = f"<trt:GetSnapshotUri><trt:ProfileToken>{self._xml_escape(token)}</trt:ProfileToken></trt:GetSnapshotUri>"
        root = self._soap_call(self.media_url, body)
        uri_el = root.find(".//{http://www.onvif.org/ver10/schema}Uri")
        if uri_el is None or not uri_el.text:
            raise RuntimeError("ONVIF GetSnapshotUri returned empty URI")
        return self._rewrite_host(uri_el.text.strip())

    def fetch_snapshot_bytes(self, profile_index: int = 0) -> bytes:
        snap_url = self.get_snapshot_uri(profile_index=profile_index)
        resp = requests.get(snap_url, auth=HTTPDigestAuth(self.user, self.password), timeout=self.timeout)
        if resp.status_code == 401:
            resp = requests.get(snap_url, auth=HTTPBasicAuth(self.user, self.password), timeout=self.timeout)
        resp.raise_for_status()
        return resp.content

    def ptz_continuous_move(self, pan: float = 0.0, tilt: float = 0.0, zoom: float = 0.0, profile_index: int = 0):
        profiles = self.get_profiles()
        if not self.ptz_url or not profiles:
            raise RuntimeError("PTZ service not supported on this ONVIF device")
        token = str(profiles[max(0, min(profile_index, len(profiles) - 1))]["token"])
        body = f"""<tptz:ContinuousMove>
      <tptz:ProfileToken>{self._xml_escape(token)}</tptz:ProfileToken>
      <tptz:Velocity>
        <tt:PanTilt x="{float(pan):.2f}" y="{float(tilt):.2f}"/>
        <tt:Zoom x="{float(zoom):.2f}"/>
      </tptz:Velocity>
    </tptz:ContinuousMove>"""
        self._soap_call(self.ptz_url, body)

    def ptz_stop(self, profile_index: int = 0):
        profiles = self.get_profiles()
        if not self.ptz_url or not profiles:
            return
        token = str(profiles[max(0, min(profile_index, len(profiles) - 1))]["token"])
        body = f"""<tptz:Stop>
      <tptz:ProfileToken>{self._xml_escape(token)}</tptz:ProfileToken>
      <tptz:PanTilt>true</tptz:PanTilt>
      <tptz:Zoom>true</tptz:Zoom>
    </tptz:Stop>"""
        self._soap_call(self.ptz_url, body)

    def _inject_rtsp_credentials(self, rtsp_url: str) -> str:
        parsed = urllib.parse.urlparse(rtsp_url)
        host = self.host
        port = parsed.port or 554
        if self.user:
            u = urllib.parse.quote(self.user, safe="")
            p = urllib.parse.quote(self.password, safe="")
            netloc = f"{u}:{p}@{host}:{port}"
        else:
            netloc = f"{host}:{port}"
        return urllib.parse.urlunparse(parsed._replace(netloc=netloc))


def resolve_onvif_to_rtsp(uri: str) -> str:
    """
    Resolves onvif://user:pass@ip:port/profile into an authenticated rtsp:// URL.
    Falls back to vendor-standard RTSP paths (Intelbras/Dahua & Hikvision) if SOAP port is blocked.
    """
    cfg = parse_onvif_uri(uri)
    client = ONVIFDirectClient(
        host=str(cfg["host"]),
        port=int(cfg["port"]),
        user=str(cfg["user"]),
        password=str(cfg["password"]),
    )
    try:
        return client.get_stream_uri(
            profile_token=cfg["profile_token"],
            profile_index=int(cfg["profile_index"]),
        )
    except Exception as exc:
        err_s = str(exc)
        if ("not authorized" in err_s.lower() or "401" in err_s) and not cfg["password"]:
            raise RuntimeError("401 Unauthorized (informe usuario/senha)") from exc
        print(f"[ONVIF] SOAP GetStreamUri failed for {cfg['host']} ({exc}), probing vendor RTSP fallback...")
        u = urllib.parse.quote(str(cfg["user"]), safe="")
        p = urllib.parse.quote(str(cfg["password"]), safe="")
        auth = f"{u}:{p}@" if cfg["user"] else ""
        ch = int(cfg["channel"])
        sub = 1 if int(cfg["profile_index"]) > 0 else 0
        # Check Intelbras/Dahua port 37777 vs Hikvision 8000
        import socket
        try:
            with socket.create_connection((str(cfg["host"]), 37777), timeout=0.6):
                return f"rtsp://{auth}{cfg['host']}:554/cam/realmonitor?channel={ch}&subtype={sub}"
        except Exception:
            pass
        hik_ch = ch * 100 + (2 if sub else 1)
        return f"rtsp://{auth}{cfg['host']}:554/Streaming/Channels/{hik_ch}"

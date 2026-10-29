# language: Python, file: protocols/discovery.py, target: Windows 11 / Cross-platform
import concurrent.futures
import ipaddress
import re
import socket
import struct
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Dict, List, Optional


CAMERA_PORTS = [80, 554, 8000, 8080, 8554, 8899, 37777, 7660]


def get_local_ipv4_subnets() -> List[str]:
    subnets = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        parts = ip.split(".")
        if len(parts) == 4 and not ip.startswith("127."):
            subnets.append(f"{parts[0]}.{parts[1]}.{parts[2]}.0/24")
    except Exception:
        pass
    if not subnets:
        subnets.append("192.168.1.0/24")
    return subnets


def _get_primary_ip() -> Optional[str]:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def ws_discovery_scan(timeout: float = 2.0) -> Dict[str, Dict[str, object]]:
    found: Dict[str, Dict[str, object]] = {}
    probe = f"""<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
            xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
            xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <e:Header>
    <w:MessageID>uuid:{uuid.uuid4()}</w:MessageID>
    <w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
    <w:Action a:mustUnderstand="true" xmlns:a="http://www.w3.org/2005/08/addressing">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
  </e:Header>
  <e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>
</e:Envelope>""".encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    sock.settimeout(timeout)
    bind_ip = _get_primary_ip()
    if bind_ip:
        try:
            sock.bind((bind_ip, 0))
        except Exception:
            pass
    try:
        sock.sendto(probe, ("239.255.255.250", 3702))
        while True:
            data, addr = sock.recvfrom(8192)
            txt = data.decode("utf-8", errors="ignore")
            xaddrs = re.findall(r"<[^:>]*:?XAddrs>([^<]+)</", txt)
            scopes = " ".join(re.findall(r"<[^:>]*:?Scopes>([^<]+)</", txt))
            hw = re.findall(r"onvif://www\.onvif\.org/hardware/([^\s<]+)", scopes)
            name = re.findall(r"onvif://www\.onvif\.org/name/([^\s<]+)", scopes)
            found[addr[0]] = {
                "onvif_xaddrs": xaddrs,
                "model": urllib.parse.unquote(hw[0]) if hw else "",
                "vendor": urllib.parse.unquote(name[0]) if name else "ONVIF",
            }
    except Exception:
        pass
    finally:
        sock.close()
    return found


def hikvision_sadp_scan(timeout: float = 2.0) -> Dict[str, Dict[str, str]]:
    found: Dict[str, Dict[str, str]] = {}
    probe = f'<?xml version="1.0" encoding="utf-8"?><Probe><Uuid>{str(uuid.uuid4()).upper()}</Uuid><Types>inquiry</Types></Probe>'.encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(timeout)
    bind_ip = _get_primary_ip()
    if bind_ip:
        try:
            sock.bind((bind_ip, 0))
        except Exception:
            pass
    try:
        sock.sendto(probe, ("239.255.255.250", 37020))
        while True:
            data, addr = sock.recvfrom(8192)
            txt = data.decode("utf-8", errors="ignore")
            model = re.findall(r"<DeviceDescription>([^<]+)</DeviceDescription>", txt)
            sn = re.findall(r"<DeviceSN>([^<]+)</DeviceSN>", txt)
            http_p = re.findall(r"<HttpPort>([^<]+)</HttpPort>", txt)
            found[addr[0]] = {
                "vendor": "Hikvision",
                "model": model[0] if model else "Hikvision IP Cam",
                "serial": sn[0] if sn else "",
                "http_port": http_p[0] if http_p else "80",
            }
    except Exception:
        pass
    finally:
        sock.close()
    return found


def intelbras_dhip_scan(timeout: float = 2.0) -> Dict[str, Dict[str, str]]:
    found: Dict[str, Dict[str, str]] = {}
    body = b'{"method":"DHDiscover.search","params":{"mac":"","uni":1}}\n'
    pkt = struct.pack("<I4sIIIIII", 0x20, b"DHIP", 0, 0, len(body), 0, len(body), 0) + body
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(timeout)
    bind_ip = _get_primary_ip()
    if bind_ip:
        try:
            sock.bind((bind_ip, 0))
        except Exception:
            pass
    try:
        sock.sendto(pkt, ("239.255.255.251", 37810))
        sock.sendto(pkt, ("255.255.255.255", 37810))
        while True:
            data, addr = sock.recvfrom(8192)
            txt = data[32:].decode("utf-8", errors="ignore")
            model = re.findall(r'"DeviceType"\s*:\s*"([^"]+)"', txt)
            sn = re.findall(r'"SerialNo"\s*:\s*"([^"]+)"', txt)
            vendor = re.findall(r'"Vendor"\s*:\s*"([^"]+)"', txt)
            found[addr[0]] = {
                "vendor": vendor[0] if vendor else "Intelbras",
                "model": model[0] if model else "Intelbras/Dahua",
                "serial": sn[0] if sn else "",
            }
    except Exception:
        pass
    finally:
        sock.close()
    return found


def _probe_single_ip(ip: str, timeout: float = 0.35) -> Optional[Dict[str, object]]:
    open_ports = []
    for port in CAMERA_PORTS:
        try:
            with socket.create_connection((ip, port), timeout=timeout):
                open_ports.append(port)
        except Exception:
            pass
    if not open_ports:
        return None

    vendor = None
    model = ""
    protocols = []
    auth_required = True

    if 554 in open_ports or 8554 in open_ports:
        protocols.append("RTSP")
    if 7660 in open_ports:
        protocols.append("ISUP-CMS")

    # Probe HTTP signatures on 80 / 8080
    for hp in [p for p in open_ports if p in (80, 8080)]:
        # 1. Check Pumatronix ITSCAM
        try:
            req = urllib.request.Request(f"http://{ip}:{hp}/api/config.cgi?Modelo", headers={"User-Agent": "Clearcam/1.0"})
            with urllib.request.urlopen(req, timeout=0.8) as r:
                body = r.read(256).decode("latin-1", errors="replace").strip()
                if "Modelo" in body and "ITSCAM" in body.upper():
                    vendor = "Pumatronix"
                    model = body.split("=", 1)[-1].strip()
                    protocols.append("Pumatronix-CGI")
                    # Check if mjpegvideo.cgi is open without auth
                    try:
                        mreq = urllib.request.Request(f"http://{ip}:{hp}/api/mjpegvideo.cgi?Quality=75&Resolution=640x480&FrameRate=5")
                        with urllib.request.urlopen(mreq, timeout=1.0) as mr:
                            ctype = mr.headers.get("Content-Type", "")
                            if "multipart" in ctype.lower():
                                auth_required = False
                    except Exception:
                        auth_required = True
                    break
        except urllib.error.HTTPError as e:
            realm = e.headers.get("WWW-Authenticate", "")
            if "pumatronix" in realm.lower() or "itscam" in realm.lower():
                vendor = "Pumatronix"
                model = "ITSCAM"
                protocols.append("Pumatronix-CGI")
                break
        except Exception:
            pass

        # 2. Check Hikvision ISAPI
        if 8000 in open_ports or 554 in open_ports:
            try:
                req = urllib.request.Request(f"http://{ip}:{hp}/ISAPI/System/deviceInfo")
                with urllib.request.urlopen(req, timeout=0.7) as r:
                    txt = r.read(300).decode("utf-8", errors="ignore")
                    if "DeviceInfo" in txt:
                        vendor = "Hikvision"
                        protocols.append("ISAPI")
            except urllib.error.HTTPError as e:
                if e.code == 401 and (8000 in open_ports or "IP Camera" in e.headers.get("WWW-Authenticate", "")):
                    vendor = "Hikvision"
                    protocols.append("ISAPI")
            except Exception:
                pass

        # 3. Check Intelbras / Dahua
        if not vendor and 37777 in open_ports:
            try:
                req = urllib.request.Request(f"http://{ip}:{hp}/cgi-bin/magicBox.cgi?action=getSystemInfo")
                with urllib.request.urlopen(req, timeout=0.7):
                    vendor = "Intelbras"
                    protocols.append("Intelbras-CGI")
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    vendor = "Intelbras"
                    protocols.append("Intelbras-CGI")
            except Exception:
                pass

    if not vendor:
        if 37777 in open_ports and 554 in open_ports:
            vendor = "Intelbras"
        elif 8000 in open_ports and 554 in open_ports:
            vendor = "Hikvision"
        elif 554 in open_ports or 8554 in open_ports or 7660 in open_ports:
            vendor = "Generic-IPCam"
        else:
            return None  # Ignore ordinary routers/PCs with only port 80/443

    return {
        "ip": ip,
        "ports": open_ports,
        "vendor": vendor,
        "model": model,
        "protocols": protocols,
        "auth_required": auth_required,
    }


def build_camera_url(
    cam: Dict[str, object],
    user: str = "",
    password: str = "",
    preferred_protocol: str = "auto",
    channel: int = 1,
    subtype: int = 0,
) -> str:
    ip = str(cam["ip"])
    vendor = str(cam.get("vendor", "")).lower()
    protocols = [str(p).upper() for p in cam.get("protocols", [])]
    ports = cam.get("ports", [])
    u = urllib.parse.quote(user, safe="") if user else ""
    p = urllib.parse.quote(password, safe="") if password else ""
    auth = f"{u}:{p}@" if user else ""

    proto = preferred_protocol.lower()
    if proto == "pumatronix" or (proto == "auto" and "pumatronix" in vendor):
        hp = 80 if 80 in ports else (8080 if 8080 in ports else 80)
        res = "1280x720" if "411" in str(cam.get("model", "")) else "640x480"
        return f"http://{auth}{ip}:{hp}/api/mjpegvideo.cgi?Quality=80&Resolution={res}&FrameRate=15"

    if proto == "onvif" or (proto == "auto" and "ONVIF" in protocols and not user):
        return f"onvif://{auth}{ip}:80/{subtype}?channel={channel}"

    if proto == "isapi":
        return f"isapi://{auth}{ip}:80/{channel * 100 + (2 if subtype else 1)}"

    if proto == "isup":
        return f"isup://{user or 'device'}:{password or 'key'}@0.0.0.0:7660?cam_ip={ip}&channel={channel * 100 + 1}"

    # Default RTSP per vendor
    rtsp_port = 554 if 554 in ports else (8554 if 8554 in ports else 554)
    if "intelbras" in vendor or "dahua" in vendor:
        return f"rtsp://{auth}{ip}:{rtsp_port}/cam/realmonitor?channel={channel}&subtype={subtype}"
    if "hikvision" in vendor:
        hik_ch = channel * 100 + (2 if subtype else 1)
        return f"rtsp://{auth}{ip}:{rtsp_port}/Streaming/Channels/{hik_ch}"
    if "ONVIF" in protocols:
        return f"onvif://{auth}{ip}:80/{subtype}?channel={channel}"
    return f"rtsp://{auth}{ip}:{rtsp_port}/stream1"


def discover_cameras(subnet_or_range: Optional[str] = None, user: str = "", password: str = "") -> List[Dict[str, object]]:
    """
    Runs multicast (WS-Discovery ONVIF, Hikvision SADP, Intelbras DHIP) + parallel TCP/HTTP scan
    and merges all discovered cameras with suggested names & connection URLs.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as mexec:
        f_wsd = mexec.submit(ws_discovery_scan)
        f_sadp = mexec.submit(hikvision_sadp_scan)
        f_dhip = mexec.submit(intelbras_dhip_scan)
        wsd = f_wsd.result()
        sadp = f_sadp.result()
        dhip = f_dhip.result()

    targets: List[str] = []
    local_ip = _get_primary_ip()
    ranges = [subnet_or_range] if subnet_or_range else get_local_ipv4_subnets()
    for rng in ranges:
        if not rng:
            continue
        rng = rng.strip()
        try:
            if "/" in rng:
                net = ipaddress.ip_network(rng, strict=False)
                targets.extend(str(h) for h in net.hosts() if str(h) != local_ip)
            elif rng.endswith("."):
                targets.extend(f"{rng}{i}" for i in range(1, 255) if f"{rng}{i}" != local_ip)
            else:
                targets.append(rng)
        except Exception:
            pass

    for extra_ip in set(list(wsd.keys()) + list(sadp.keys()) + list(dhip.keys())):
        if extra_ip not in targets:
            targets.append(extra_ip)

    merged: Dict[str, Dict[str, object]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=96) as ex:
        for res in ex.map(_probe_single_ip, targets):
            if res:
                merged[str(res["ip"])] = res

    for ip, winfo in wsd.items():
        entry = merged.setdefault(ip, {"ip": ip, "ports": [80, 554], "vendor": winfo.get("vendor", "ONVIF"), "model": "", "protocols": [], "auth_required": True})
        if "ONVIF" not in entry["protocols"]:
            entry["protocols"].append("ONVIF")
        if not entry.get("model") and winfo.get("model"):
            entry["model"] = winfo["model"]
        if entry.get("vendor") in (None, "Generic-IPCam", "Unknown") and winfo.get("vendor"):
            entry["vendor"] = winfo["vendor"]

    for ip, sinfo in sadp.items():
        entry = merged.setdefault(ip, {"ip": ip, "ports": [80, 554, 8000], "vendor": "Hikvision", "model": "", "protocols": ["RTSP", "ISAPI"], "auth_required": True})
        entry["vendor"] = "Hikvision"
        entry["model"] = sinfo.get("model") or entry.get("model", "")
        entry["serial"] = sinfo.get("serial", "")
        if "SADP" not in entry["protocols"]:
            entry["protocols"].append("SADP")

    for ip, dinfo in dhip.items():
        entry = merged.setdefault(ip, {"ip": ip, "ports": [80, 554, 37777], "vendor": "Intelbras", "model": "", "protocols": ["RTSP", "ONVIF"], "auth_required": True})
        entry["vendor"] = dinfo.get("vendor", "Intelbras")
        entry["model"] = dinfo.get("model") or entry.get("model", "")
        entry["serial"] = dinfo.get("serial", "")

    cameras_list = []
    for ip in sorted(merged.keys(), key=lambda x: tuple(int(p) for p in x.split(".") if p.isdigit())):
        cam = merged[ip]
        last_octet = ip.split(".")[-1]
        clean_vendor = re.sub(r"[^A-Za-z0-9]", "", str(cam.get("vendor", "Cam")))
        clean_model = re.sub(r"[^A-Za-z0-9]", "", str(cam.get("model", "")))[:12]
        cam["suggested_name"] = f"{clean_vendor}_{clean_model}_{last_octet}".strip("_")
        cam["suggested_url"] = build_camera_url(cam, user=user, password=password)
        cam["onvif_url"] = build_camera_url(cam, user=user, password=password, preferred_protocol="onvif") if "ONVIF" in cam.get("protocols", []) else ""
        cameras_list.append(cam)

    return cameras_list

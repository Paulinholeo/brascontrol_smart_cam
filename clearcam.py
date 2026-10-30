import os as _os, sys as _sys, builtins as _builtins, types as _types
if _sys.platform == "win32":
  _os.environ.setdefault("DEV", "CL")
  if not hasattr(_builtins, "io_uring"):
    _builtins.io_uring = _types.SimpleNamespace(struct_io_uring_params=object)
from tinygrad.tensor import Tensor
from tinygrad.helpers import fetch
from tinygrad.dtype import dtypes
from detection.yolov9 import YOLOv9
from llm.qwen3vl import Qwen3VL
import numpy as np
from pathlib import Path
import cv2
from collections import defaultdict, deque
import time, sys
import json
import http
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import threading
import shutil
from datetime import datetime, time as time_obj
import uuid
import urllib
from urllib.parse import urlparse, parse_qs
import struct
from urllib.parse import unquote, quote
import zlib
from utils.db import db
import multiprocessing
import re
import base64
from utils.helpers import send_notif, find_ffmpeg, export_clip, upload_file, encrypt_file, export_and_upload, jit_infer
import pickle
from protocols.pumatronix import (
    PumatronixClient, PumatronixWatchdogManager, is_pumatronix_source,
    parse_pumatronix_uri, FLASH_MODES, SUPPORTED_RESOLUTIONS
)
from protocols.onvif_client import (
    ONVIFDirectClient, is_onvif_source, parse_onvif_uri, resolve_onvif_to_rtsp
)
from protocols.isup import (
    isup_gateway, HikvisionISAPIClient, is_isup_or_isapi_source, parse_isup_uri
)
from protocols.discovery import discover_cameras, build_camera_url
from models.ocr import ocr_engine, find_tesseract_binary
from models.vehicle_attributes import (
    VehicleAttributeAnalyzer, VEHICLE_COLORS, normalize_plate_key, infer_plate_uf
)

# RTSP URL
# Video capture thread
import subprocess
import threading
import time
import numpy as np
import cv2
from datetime import datetime
import os
import queue as _std_queue
from utils.helpers import BASE_DIR
APP_VERSION = (Path(__file__).resolve().parent / "VERSION").read_text().strip() if (Path(__file__).resolve().parent / "VERSION").is_file() else "dev"
from ocsort_tracker import ocsort

(BASE_DIR / "cameras").mkdir(parents=True, exist_ok=True)
models = {1: "t", 2: "s", 3: "m", 4: "c", 5: "e", 6: "nano", 7: "small", 8:"medium", 9:"large"}
puma_watchdog = PumatronixWatchdogManager()
ocr_queue = _std_queue.Queue(maxsize=256)
cam_runtime_status = {}
last_ocr_by_cam = {}
ocr_by_image = {}
vehicle_analyzer = VehicleAttributeAnalyzer()
_main_loop_ready = False

_PROFILE = _os.environ.get("CLEARCAM_PROFILE") == "1"

class LoopProfiler:
  # CLEARCAM_PROFILE=1: prints main-loop time per stage every 10 s
  def __init__(self, every=10.0):
    self.every, self.t, self.n, self.acc = every, time.time(), 0, defaultdict(float)
    self.frame_counts, self.prev_counts = None, {} # frame_counts: dict cam -> frames captured so far
  def add(self, **stages):
    for k, v in stages.items(): self.acc[k] += v
    self.n += 1
    if (dt := time.time() - self.t) >= self.every:
      ms = " ".join(f"{k}={v / self.n * 1000:.1f}ms" for k, v in self.acc.items() if k != "frames")
      print(f"\n[profile] {self.n / dt:.1f} loops/s, {self.acc['frames'] / dt:.1f} frames/s, per loop: {ms}", flush=True)
      if self.frame_counts is not None:
        counts = dict(self.frame_counts)
        print("[profile] capture fps: " + " ".join(f"{c}={(n - self.prev_counts.get(c, n)) / dt:.1f}" for c, n in counts.items()), flush=True)
        self.prev_counts = counts
      self.t, self.n, self.acc = time.time(), 0, defaultdict(float)

loop_prof = LoopProfiler()

class RollingClassCounter:
  def __init__(self, window_seconds=None, max=None, classes=None, sched=[[0,86399],True,True,True,True,True,True,True],cam_name=None, desc=None, threshold=0.28):
    self.window = window_seconds
    self.data = defaultdict(deque)
    self.max = max
    self.classes = classes
    self.last_det = 0
    self.sched = sched
    self.cam_name = cam_name
    self.is_on = True
    self.is_notif = True
    self.zone = True
    self.reset = False
    self.new = True
    self.desc = desc
    self.desc_emb = None
    self.threshold = threshold

  def add(self, class_id):
    if self.classes is not None and class_id not in self.classes: return
    now = time.time()
    self.data[class_id].append(now)
    self.cleanup(class_id, now)

  def cleanup(self, class_id, now):
    q = self.data[class_id]
    window = self.window if self.window else (60 if self.is_notif else 1)
    while window and q and now - q[0] > window:
      q.popleft()

  def reset_counts(self):
    for class_id, _ in self.data.items():
       self.data[class_id] = deque() # todo, use in reset endpoint?
    self.reset = True

  def get_counts(self):
    window = self.window if self.window else (60 if self.is_notif else 1)
    max_reached = False
    now = time.time()
    counts = {}
    for class_id, q in self.data.items():
      while window and q and now - q[0] > window:
        q.popleft()
      if q:
        counts[class_id] = len(q)
        if self.max and len(q) >= self.max: max_reached = True
    return counts, max_reached
  
  def is_active(self, offset=0):
    if not alerts_on[self.cam_name]: return False
    if not getattr(self, "is_on", False): return False
    if not self.sched: return True
    now = time.localtime()
    time_of_day = now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec
    if not self.sched[time.localtime().tm_wday + 1]: return False
    window = self.window if self.window else (60 if self.is_notif else 1)
    return time_of_day < self.sched[0][1] and time_of_day > ((self.sched[0][0] - window) + offset)

def write_png(filename, array):
    array = array[..., ::-1]  # BGR to RGB
    height, width, _ = array.shape
    png_signature = b"\x89PNG\r\n\x1a\n"
    def chunk(chunk_type, data):
        return (struct.pack("!I", len(data)) +
                chunk_type +
                data +
                struct.pack("!I", zlib.crc32(chunk_type + data) & 0xffffffff))
    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw_data = b"".join(b"\x00" + array[y].tobytes() for y in range(height))
    compressed = zlib.compress(raw_data, 9)
    png_bytes = (
        png_signature +
        chunk(b"IHDR", ihdr) +
        chunk(b"IDAT", compressed) +
        chunk(b"IEND", b"")
    )
    with open(filename, "wb") as f:
        f.write(png_bytes)

import numpy as np

def draw_rectangle_numpy(img, pt1, pt2, color, thickness=1):
    x1, y1 = pt1
    x2, y2 = pt2
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1]-1, x2), min(img.shape[0]-1, y2)
    if thickness == -1:  # fill
        img[y1:y2+1, x1:x2+1] = color
    else:
        img[y1:y1+thickness, x1:x2+1] = color
        img[y2-thickness+1:y2+1, x1:x2+1] = color
        img[y1:y2+1, x1:x1+thickness] = color
        img[y1:y2+1, x2-thickness+1:x2+1] = color
    return img


def is_vod(cam_name): return Path("data/cameras", cam_name, "streams", "video").is_dir()

def _detect_protocol_name(src: str) -> str:
  if not isinstance(src, str): return "Unknown"
  low = src.lower()
  if is_pumatronix_source(src): return "Pumatronix-CGI"
  if is_onvif_source(src): return "ONVIF"
  if low.startswith("isup://"): return "ISUP-5.0"
  if low.startswith("isapi://"): return "Hikvision-ISAPI"
  if low.startswith("rtsp"): return "RTSP"
  if low.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')): return "VOD"
  return "HTTP/Stream"

def _redact_url(src: str) -> str:
  if not isinstance(src, str): return ""
  return re.sub(r"(://[^:/@]+):([^@]+)@", r"\1:***@", src)

def _get_stream_resolution(src):
  if not isinstance(src, str):
    return 1280, 720
  if is_pumatronix_source(src):
    cfg = parse_pumatronix_uri(src)
    res_str = str(cfg.get("resolution", "640x480"))
    if "x" in res_str:
      try:
        w, h = map(int, res_str.lower().split("x", 1))
        return w, h
      except Exception:
        pass
    return 640, 480
  if is_onvif_source(src) or is_isup_or_isapi_source(src):
    return 1280, 720
  ffmpeg_path = find_ffmpeg()
  is_rtsp = src.lower().startswith("rtsp")
  command = [ffmpeg_path, *(["-rtsp_transport", "tcp"] if is_rtsp else []), "-i", src]
  try:
    result = subprocess.run(
        command,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=4
    )
    output = result.stderr or ""
    match = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", output)
    if match:
      width, height = map(int, match.groups())
      return width, height
    return 1280, 720
  except Exception:
    return 1280, 720

class VideoCapture:
  def __init__(self):
    self.output_dir_raw = {}
    self.frame_num = {}
    self.last_frame_num = {}
    self.vod = {}
    # objects in scene count
    self.counter = {}
    self.object_set = {}
    self.object_set_zone = {}

    self.src = {}
    self.resolved_src = {}
    self.mode = {}
    self.width, self.height = {}, {}
    self.proc = {}
    self.hls_proc = {}
    self.running = {}

    self.raw_frame = {}
    self.last_preds = {}
    self.last_frames = {}

    self.settings = {}
    self.count = {}
    self.prev_time = {}
    self.current_stream_dir_raw = {}
    self.pred_occs = {}
    self.tracker = {}
    self.cap = {}
    self.src_fps = {}
    self.last_det = {}
    self.last_live_check = {}
    self.last_counter_update = {}
    self.last_preview_time = {}
    self.last_live_seg = {}
    self.start_time = {}
    self.filename = {}
    self.alert_counters = {}
    self.live_link = {}
    self.live_link_lock = {}
    self.last_ocr_ts = {}
    self.active_links = set() # camera names in the links table, refreshed by the main loop every 5 s

  def _kill_procs(self, cam_name):
    for store in (self.proc, self.hls_proc):
      p = store.get(cam_name)
      if p is not None:
        try:
          if p.stdin:
            p.stdin.close()
        except Exception:
          pass
        try:
          p.kill()
        except Exception:
          pass
        store[cam_name] = None

  def init_cam(self, cam_name, src):
    self.counter[cam_name] = RollingClassCounter(cam_name=cam_name, window_seconds=float('inf'))
    self.active_links.add(cam_name)
    self.src[cam_name] = src
    self.resolved_src[cam_name] = None
    self.mode[cam_name] = "init"
    self.last_frames[cam_name] = deque(maxlen=2)
    self.vod[cam_name] = isinstance(src, str) and src.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
    self.frame_num[cam_name] = -1
    self.last_frame_num[cam_name] = -1
    self.object_set[cam_name] = set()
    self.object_set_zone[cam_name] = set()
    self.running[cam_name] = True
    self.output_dir_raw[cam_name] = BASE_DIR / "cameras" / f'{cam_name}' / "streams"
    self.last_preds[cam_name] = []
    self.raw_frame[cam_name] = None
    self.width[cam_name], self.height[cam_name] = _get_stream_resolution(src)
    self.settings[cam_name] = None
    self.start_time[cam_name] = time.time()
    self.last_ocr_ts[cam_name] = 0.0

    cam_runtime_status[cam_name] = {
      "protocol": _detect_protocol_name(src),
      "state": "connecting",
      "url": _redact_url(src),
      "resolution": f"{self.width[cam_name]}x{self.height[cam_name]}",
      "fps": 0.0,
      "last_error": "",
    }
    
    self.alert_counters[cam_name] = database.run_get("alerts",cam_name)
    if not self.alert_counters[cam_name]:
      self.alert_counters[cam_name] = dict()
      id, alert_counter = str(uuid.uuid4()), RollingClassCounter(window_seconds=None, max=1, classes={0,1,2,3,5,7},cam_name=cam_name)
      self.alert_counters[cam_name][id] = alert_counter
      database.run_put("alerts", cam_name, alert_counter, id=id)

    self.last_det[cam_name] = -1
    self.last_live_check[cam_name] = time.time()
    self.last_live_seg[cam_name] = time.time()
    self.last_preview_time[cam_name] = None
    self.last_counter_update[cam_name] = time.time()
    self.pred_occs[cam_name] = {}
    self.tracker[cam_name] = ocsort.OCSort(max_age=100)
    self.count[cam_name] = 0
    self.prev_time[cam_name] = time.time()
    self.current_stream_dir_raw[cam_name] = self._get_new_stream_dir(cam_name)
    self.filename[cam_name] = None
    self.live_link_lock[cam_name] = threading.Lock()
    alerts_on[cam_name] = True
    self.hls_proc[cam_name], self.proc[cam_name] = self._open_ffmpeg(cam_name)

  def start(self):
    cam_check = time.time()
    cams = database.run_get("links", None)
    for cam_name in cams.keys():
      if type(cams[cam_name]) != str: continue
      print("starting",cam_name,"src:",_redact_url(cams[cam_name]))
      self.init_cam(cam_name=cam_name, src=cams[cam_name])
      if not self.vod[cam_name]:
        threading.Thread(target=self.frame_loop, args=(cam_name,), daemon=True).start()
    global _main_loop_ready
    _main_loop_ready = True
    loop_prof.frame_counts = self.frame_num
    while True:
      if time.time() - cam_check >= 5:
        cam_check = time.time()
        new_cams = database.run_get("links", None)
        for cam_name in new_cams.keys():
          if type(new_cams[cam_name]) != str: continue
          if cam_name not in cams:
            self.init_cam(cam_name=cam_name, src=new_cams[cam_name])
            if not self.vod[cam_name]: threading.Thread(target=self.frame_loop, args=(cam_name,), daemon=True).start()
          else:
            if new_cams[cam_name] != cams[cam_name]:
              cams[cam_name] = new_cams[cam_name]
              self.src[cam_name] = new_cams[cam_name]
              cam_runtime_status.setdefault(cam_name, {})["protocol"] = _detect_protocol_name(new_cams[cam_name])
              cam_runtime_status[cam_name]["url"] = _redact_url(new_cams[cam_name])
              cam_runtime_status[cam_name]["state"] = "reconnecting"
              self.width[cam_name], self.height[cam_name] = _get_stream_resolution(new_cams[cam_name])
              self.hls_proc[cam_name], self.proc[cam_name] = self._open_ffmpeg(cam_name)
        for removed in list(cams.keys()):
          if removed not in new_cams:
            self._kill_procs(removed)
            puma_watchdog.stop(removed)
            cam_runtime_status.pop(removed, None)
        cams = new_cams
        self.active_links = set(cams.keys())
      pending = []
      t0 = time.perf_counter()
      for cam_name in list(cams.keys()):
        if cam_name not in self.vod: continue
        if self.vod[cam_name]: self.process_frame_vod(cam_name=cam_name)
        elif (frame := self.live_frame_for_inference(cam_name)) is not None: pending.append((cam_name, frame))
        else: self.process_frame(cam_name=cam_name)
      t1 = time.perf_counter()
      if pending: # one batched detector call for every live camera with a new frame
        all_dets = model.detect_batch([f for _, f in pending])
        t2 = time.perf_counter()
        for (cam_name, frame), dets in zip(pending, all_dets):
          self.process_frame(cam_name=cam_name, frame=frame, dets=dets)
      else: t2 = t1
      t3 = time.perf_counter()
      process_queue()
      t4 = time.perf_counter()
      if len(object_queue) > 0:
        try:
          img = cv2.imread(str(object_queue[0]))
          if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            clip_latest_img(img)
            process_latest_face(img)
        except Exception as e: print("error in object processing", object_queue[0], e)
        del object_queue[0]
      t5 = time.perf_counter()
      if _PROFILE: loop_prof.add(gather=t1-t0, detect=t2-t1, track_post=t3-t2, queue=t4-t3, objects=t5-t4, frames=len(pending))
      time.sleep(0.005)

  def _get_new_stream_dir(self, cam_name):
      timestamp = "video" if self.vod[cam_name] else datetime.now().strftime("%Y-%m-%d")
      stream_dir_raw = self.output_dir_raw[cam_name] / timestamp
      stream_dir_raw.mkdir(parents=True, exist_ok=True)
      return stream_dir_raw

  def _next_hls_start_number(self, path: Path) -> int:
      max_idx = -1
      try:
          for seg in path.glob("stream_*.m4s"):
              stem = seg.stem.replace("stream_", "")
              if stem.isdigit():
                  idx = int(stem)
                  if idx > max_idx:
                      max_idx = idx
      except Exception:
          pass
      return max_idx + 1

  def _spawn_stdin_hls_encoder(self, path: Path, width: int, height: int, fps: int = 15):
      ffmpeg_path = find_ffmpeg()
      fps = max(1, min(30, int(fps or 15)))
      start_num = self._next_hls_start_number(path)
      init_name = f"init_{start_num:06d}.mp4" if start_num > 0 else "init.mp4"
      hls_flags = "append_list+discont_start+independent_segments" if start_num > 0 else "append_list+independent_segments"
      command = [
          ffmpeg_path,
          "-y",
          "-loglevel", "error",
          "-f", "rawvideo",
          "-pix_fmt", "bgr24",
          "-s", f"{width}x{height}",
          "-r", str(fps),
          "-i", "pipe:0",
          "-c:v", "libx264",
          "-preset", "ultrafast",
          "-tune", "zerolatency",
          "-pix_fmt", "yuv420p",
          "-g", str(fps * 2),
          "-an",
          "-f", "hls",
          "-hls_time", "2",
          "-hls_list_size", "0",
          "-hls_playlist_type", "event",
          "-hls_segment_type", "fmp4",
          "-hls_fmp4_init_filename", init_name,
          "-hls_flags", hls_flags,
          "-hls_segment_filename", str(path / "stream_%06d.m4s"),
          "-start_number", str(start_num),
          str(path / "stream.m3u8"),
      ]
      return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  def _open_ffmpeg(self, cam_name):
    path = self._get_new_stream_dir(cam_name)
    self._kill_procs(cam_name)
    src = self.src.get(cam_name)
    if type(src) != str:
      return None, None

    ffmpeg_path = find_ffmpeg()
    if self.vod[cam_name]:
      self.mode[cam_name] = "vod"
      command = [
        ffmpeg_path,
        "-i", src,
        "-c:v", "copy",
        "-an",
        "-f", "hls",
        "-hls_time", "2",
        "-hls_list_size", "0",
        "-hls_flags", "independent_segments",
        "-hls_segment_type", "fmp4",
        "-hls_fmp4_init_filename", "init.mp4",
        "-hls_segment_filename", str(path / "seg_%06d.m4s"),
        str(path / "stream.m3u8"),
      ]
      return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), None

    # 1. Pumatronix ITSCAM mjpegvideo.cgi / snapshot.cgi
    if is_pumatronix_source(src):
      self.mode[cam_name] = "pumatronix"
      cfg = parse_pumatronix_uri(src)
      fps = int(cfg.get("framerate") or 15)
      if fps <= 0: fps = 15
      client = PumatronixClient(
          ip=str(cfg["ip"]),
          port=int(cfg["port"]),
          user=cfg.get("user"),
          password=cfg.get("password"),
      )
      if cfg.get("watchdog"):
        puma_watchdog.start(cam_name, client)
      else:
        puma_watchdog.stop(cam_name)
      hls_proc = self._spawn_stdin_hls_encoder(path, self.width[cam_name], self.height[cam_name], fps=fps)
      if self.start_time[cam_name] is None: self.start_time[cam_name] = time.time()
      return hls_proc, None

    # 2. ONVIF / ISUP / ISAPI resolution to RTSP or ISAPI polling
    effective_src = src
    if is_onvif_source(src):
      try:
        effective_src = resolve_onvif_to_rtsp(src)
      except Exception as exc:
        err_s = str(exc)
        cam_runtime_status.setdefault(cam_name, {})["last_error"] = err_s
        if "401" in err_s or "notauthorized" in err_s.lower() or "unauthorized" in err_s.lower():
          cam_runtime_status[cam_name]["state"] = "auth_needed"
          self.mode[cam_name] = "auth_wait"
        else:
          self.mode[cam_name] = "wait_retry"
        return None, None
    elif is_isup_or_isapi_source(src):
      resolved = isup_gateway.resolve_stream_source(src)
      if resolved:
        effective_src = resolved
      else:
        self.mode[cam_name] = "isup_wait"
        cam_runtime_status.setdefault(cam_name, {})["state"] = "waiting_isup_7660"
        return None, None

    self.resolved_src[cam_name] = effective_src
    self.mode[cam_name] = "ffmpeg"
    is_rtsp = effective_src.lower().startswith("rtsp")

    start_num = self._next_hls_start_number(path)
    init_name = f"init_{start_num:06d}.mp4" if start_num > 0 else "init.mp4"
    hls_flags = "append_list+discont_start+independent_segments" if start_num > 0 else "append_list+independent_segments"
    hls_cmd = [
        ffmpeg_path,
        "-loglevel", "error",
        *(["-rtsp_transport", "tcp"] if is_rtsp else []),
        "-fflags", "+genpts",
        "-avoid_negative_ts", "make_zero",
        "-i", effective_src,
        "-c:v", "copy",
        "-an",
        "-f", "hls",
        "-hls_time", "2",
        "-hls_list_size", "0",
        "-hls_playlist_type", "event",
        "-hls_segment_type", "fmp4",
        "-hls_fmp4_init_filename", init_name,
        "-hls_flags", hls_flags,
        "-hls_segment_filename", str(path / "stream_%06d.m4s"),
        "-start_number", str(start_num),
        str(path / "stream.m3u8")
    ]
    hls_proc = subprocess.Popen(hls_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if self.start_time[cam_name] is None: self.start_time[cam_name] = time.time()
    
    raw_cmd = [
        ffmpeg_path,
        "-loglevel", "error",
        *(["-rtsp_transport", "tcp"] if is_rtsp else []),
        "-fflags", "+discardcorrupt+nobuffer",
        "-flags", "low_delay",
        "-i", effective_src,
        "-an",
        "-vf", f"scale={self.width[cam_name]}:{self.height[cam_name]}",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "-threads", "1",
        "-"
    ]
    raw_proc = subprocess.Popen(raw_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return hls_proc, raw_proc

  def save_object(self, p, ts=0, cam_name=None):
    p = np.array([p.tlwh[0],p.tlwh[1],(p.tlwh[0]+p.tlwh[2]),(p.tlwh[1]+p.tlwh[3]),p.score,p.class_id,p.track_id])
    timestamp = "video" if self.vod[cam_name] else datetime.now().strftime("%Y-%m-%d")
    filepath = BASE_DIR / "cameras" / f"{cam_name}/objects/{timestamp}"
    filepath.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "cameras" / f"{cam_name}/faces/{timestamp}").mkdir(parents=True, exist_ok=True)
    object_filename = filepath / f"{ts}_{int(p[6])}_{int(p[5])}.jpg"
    x1, y1, x2, y2 = map(int, (p[0], p[1], p[2], p[3]))
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    hw = (x2 - x1) // 2
    hh = (y2 - y1) // 2
    hw *= 2
    hh *= 2
    x1_new = cx - hw
    x2_new = cx + hw
    y1_new = cy - hh
    y2_new = cy + hh
    H, W = self.last_frames[cam_name][-1].shape[:2]
    x1_new = max(0, min(x1_new, W))
    x2_new = max(0, min(x2_new, W))
    y1_new = max(0, min(y1_new, H))
    y2_new = max(0, min(y2_new, H))
    if (y2_new - y1_new) < 80 or (x2_new - x1_new) < 80: return
    full_frame = self.last_frames[cam_name][-1]
    crop = full_frame[y1_new:y2_new, x1_new:x2_new].copy()
    cv2.imwrite(str(object_filename), crop)
    if global_settings.use_clip or global_settings.use_face: object_queue.append(object_filename)
    if getattr(global_settings, "use_ocr", False):
      now_t = time.time()
      min_iv = float(getattr(global_settings, "ocr_interval", 2.0) or 1.0)
      if now_t - self.last_ocr_ts.get(cam_name, 0.0) >= min_iv:
        self.last_ocr_ts[cam_name] = now_t
        try:
          ocr_queue.put_nowait((cam_name, str(object_filename), crop, full_frame.copy(), (x1_new, y1_new, x2_new, y2_new), int(p[5]), ts))
        except _std_queue.Full:
          pass

  def frame_loop(self, cam_name):
    fail_count = 0
    while (BASE_DIR / "cameras" / cam_name).exists() or cam_name in self.src:
      if cam_name not in self.active_links:
        break
      mode = self.mode.get(cam_name, "ffmpeg")
      try:
        # Mode 1: Pumatronix native MJPEG reader + H.264 HLS pipe
        if mode == "pumatronix":
          src = self.src.get(cam_name, "")
          cfg = parse_pumatronix_uri(src)
          client = PumatronixClient(
              ip=str(cfg["ip"]),
              port=int(cfg["port"]),
              user=cfg.get("user"),
              password=cfg.get("password"),
          )
          W, H = self.width[cam_name], self.height[cam_name]
          fps = int(cfg.get("framerate") or 15)
          for frame in client.iter_mjpeg_frames(
              quality=int(cfg["quality"]),
              resolution=str(cfg["resolution"]),
              framerate=fps,
          ):
            if self.src.get(cam_name) != src or cam_name not in self.active_links:
              break
            if frame.shape[1] != W or frame.shape[0] != H:
              frame = cv2.resize(frame, (W, H))
            self.raw_frame[cam_name] = frame
            self.frame_num[cam_name] += 1
            cam_runtime_status.setdefault(cam_name, {})["state"] = "online"
            cam_runtime_status[cam_name]["last_error"] = ""
            hp = self.hls_proc.get(cam_name)
            if hp and hp.stdin and hp.poll() is None:
              try:
                hp.stdin.write(np.ascontiguousarray(frame).data) # no 2.7 MB copy per frame
              except Exception:
                pass
          time.sleep(2.0)
          if cam_name in database.run_get("links", None):
            self.hls_proc[cam_name], self.proc[cam_name] = self._open_ffmpeg(cam_name)
          continue

        # Mode 2: Waiting for credentials (no IP lockout spam), ISUP 7660 device registration, or ONVIF retry
        if mode == "auth_wait":
          time.sleep(2.0)
          continue
        if mode in ("isup_wait", "wait_retry"):
          time.sleep(5.0)
          self.hls_proc[cam_name], self.proc[cam_name] = self._open_ffmpeg(cam_name)
          continue

        # Mode 3: FFmpeg rawvideo pipe (RTSP / ONVIF / ISAPI-RTSP / HTTP)
        proc = self.proc.get(cam_name)
        if proc is None or proc.stdout is None:
          time.sleep(2.0)
          self.hls_proc[cam_name], self.proc[cam_name] = self._open_ffmpeg(cam_name)
          continue

        frame_size = self.width[cam_name] * self.height[cam_name] * 3
        raw_bytes = proc.stdout.read(frame_size)
        if len(raw_bytes) != frame_size:
          err_txt = ""
          if proc.stderr and proc.poll() is not None:
            try:
              err_txt = proc.stderr.read(600).decode("utf-8", errors="ignore")
            except Exception:
              pass
          if "401 Unauthorized" in err_txt or "authorization failed" in err_txt.lower():
            cam_runtime_status.setdefault(cam_name, {})["state"] = "auth_needed"
            cam_runtime_status[cam_name]["last_error"] = "401 Unauthorized (informe usuario/senha)"
            self._kill_procs(cam_name)
            self.mode[cam_name] = "auth_wait"
            continue
          time.sleep(0.5)
          if time.time() - (self.start_time.get(cam_name) or 0) > 10:
            fail_count += 1
            if fail_count > 5:
              if err_txt:
                cam_runtime_status.setdefault(cam_name, {})["last_error"] = err_txt.strip().splitlines()[-1][:140]
              cam_runtime_status.setdefault(cam_name, {})["state"] = "reconnecting"
              self.hls_proc[cam_name], self.proc[cam_name] = self._open_ffmpeg(cam_name)
              fail_count = 0
        else:
          fail_count = 0
          self.raw_frame[cam_name] = np.frombuffer(raw_bytes, np.uint8).reshape((self.height[cam_name], self.width[cam_name], 3))
          self.frame_num[cam_name] += 1
          cam_runtime_status.setdefault(cam_name, {})["state"] = "online"
          cam_runtime_status[cam_name]["last_error"] = ""
          time.sleep(1 / 100)
      except Exception as e:
        err_s = str(e)
        cam_runtime_status.setdefault(cam_name, {})["last_error"] = err_s[:120]
        if "401" in err_s or "unauthorized" in err_s.lower():
          cam_runtime_status[cam_name]["state"] = "auth_needed"
          self._kill_procs(cam_name)
          self.mode[cam_name] = "auth_wait"
        else:
          time.sleep(2.0)

  def process_frame_vod(self, cam_name):
    if cam_name not in self.cap:
      self.cap[cam_name] = cv2.VideoCapture(self.src[cam_name])
      self.src_fps[cam_name] = self.cap[cam_name].get(cv2.CAP_PROP_FPS) or 30

    self.cap[cam_name].grab()  # skip for max fps
    ret, frame = self.cap[cam_name].read()
    self.last_frames[cam_name].append(frame)
    if not ret or cam_name not in self.active_links:
      self.running[cam_name] = False
      if "Processing" not in database.run_get("analysis_prog", cam_name): database.run_put("analysis_prog", cam_name, {"Tracking":100}) # todo stop when done?
    else:
      self.last_preds[cam_name], _ = self.run_inference(frame, cam_name=cam_name)
      database.run_put("analysis_prog", cam_name, {"Tracking":self.cap[cam_name].get(cv2.CAP_PROP_POS_FRAMES)/self.cap[cam_name].get(cv2.CAP_PROP_FRAME_COUNT)*100})


  def live_frame_for_inference(self, cam_name):
    # newest live frame if process_frame would run inference on it now, else None
    if self.raw_frame[cam_name] is None or self.frame_num[cam_name] == self.last_frame_num[cam_name]: return None
    if not any(counter.is_active() for _, counter in self.alert_counters[cam_name].items()): return None
    if global_settings.use_notifs() and not alerts_on[cam_name]: return None
    return self.raw_frame[cam_name].copy()

  def process_frame(self, cam_name, frame=None, dets=None):
    frame_num = self.frame_num[cam_name]
    last_frame_num = self.last_frame_num[cam_name]
    if self.raw_frame[cam_name] is None: return
    if frame_num == last_frame_num: return
    if frame is None: frame = self.raw_frame[cam_name].copy()

    # don't run inference when no active scheds
    if not any(counter.is_active() for _, counter in self.alert_counters[cam_name].items()): self.last_preds[cam_name] = [] # to remove annotation when no alerts active
    else:
      if not global_settings.use_notifs() or alerts_on[cam_name]:
        self.last_frames[cam_name].append(frame) # before run_inference so save_object crops this frame, not the previous one
        preds, frame = self.run_inference(frame, cam_name=cam_name, dets=dets)
        self.last_preds[cam_name] = preds.copy()
        self.last_frame_num[cam_name] = self.frame_num[cam_name]

        curr_time = time.time()
        fps = 1 / max(curr_time - self.prev_time[cam_name], 1e-4)
        self.prev_time[cam_name] = curr_time
        cam_runtime_status.setdefault(cam_name, {})["fps"] = round(fps, 1)
        print(f"\rFPS ({cam_name}): {fps:.2f}   ", end="", flush=True)
      else:
        self.last_frame_num[cam_name] = self.frame_num[cam_name]
        self.last_preds[cam_name] = []

    filtered_preds = self.last_preds[cam_name]

    if self.count[cam_name] > 10:
      if self.last_preview_time[cam_name] is None or time.time() - self.last_preview_time[cam_name] >= 3600: # preview every hour
        self.last_preview_time[cam_name] = time.time()
        self.filename[cam_name] = BASE_DIR / "cameras" / f"{cam_name}/preview.png"
        write_png(self.filename[cam_name], self.raw_frame[cam_name])
      for _,alert in self.alert_counters[cam_name].items():
          if alert.desc is not None: continue
          if not alert.is_active():
            alert.reset_counts()
            continue
          window = alert.window if alert.window else (60 if alert.is_notif else 1)
          if alert.get_counts()[1]:
            if time.time() - alert.last_det >= window and (time.time() - alert.last_det >= window):
              timestamp = "video" if self.vod[cam_name] else datetime.now().strftime("%Y-%m-%d")
              filepath = BASE_DIR / "cameras" / f"{cam_name}/event_images/{timestamp}"
              filepath.mkdir(parents=True, exist_ok=True)
              annotated_frame = draw_predictions(self.last_frames[cam_name][-1].copy(), filtered_preds, color_dict, cam_name=cam_name)
              # todo alerts can be sent with the wrong thumbnail if two happen quickly, use map
              ts = int(self.cap[cam_name].get(cv2.CAP_PROP_POS_FRAMES) / self.src_fps[cam_name]) - 5 if self.vod[cam_name] else int(time.time() - self.start_time[cam_name] - 15)
              self.filename[cam_name] = filepath / f"{ts}_notif.jpg" if alert.is_notif else filepath / f"{ts}.jpg"
              if not self.vod[cam_name]: cv2.imwrite(str(self.filename[cam_name]), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]) # we've 10MB limit for video file, raw png is 3MB!
              if (plain := filepath / f"{ts}.jpg").exists() and (filepath / f"{ts}_notif.jpg").exists():
                plain.unlink() # only one image per event
                self.filename[cam_name] = filepath / f"{ts}_notif.jpg"
              if global_settings.use_notifs() is not None and not self.vod[cam_name] and alert.is_notif:
                title = f"Event Detected ({cam_name})"
                threading.Thread(target=send_notif, kwargs={"session_token": global_settings.userID,"text":title,"body_text": None,"host": global_settings.server_url, "img": self.filename[cam_name]}, daemon=True).start()
                if global_settings.use_qwen: # extra notif if qwen
                  # use frames before last, only one reset needed, must convert to RGB
                  for i in range(len(self.last_frames[cam_name])-1): qwen.generate(image=cv2.cvtColor(self.last_frames[cam_name][i], cv2.COLOR_BGR2RGB), reset=True if i==0 else False)
                  text = qwen.generate(prompt=global_settings.qwen_prompt, image=cv2.cvtColor(cv2.imread(self.filename[cam_name]), cv2.COLOR_BGR2RGB), reset=False) # must reset or run out of context
                  threading.Thread(target=send_notif, kwargs={"session_token": global_settings.userID, "text": f"AI Summary ({cam_name}):", "body_text": text, "host": global_settings.server_url}, daemon=True).start()
                if global_settings.clearcam_user(): threading.Thread(target=export_and_upload, kwargs={"cam_name": cam_name, "thumbnail": self.filename[cam_name], "userID": global_settings.userID, "key": global_settings.key, "start": ts}, daemon=True).start()
              self.last_det[cam_name] = time.time()
              alert.last_det = time.time()
      
      if (time.time() - self.last_live_check[cam_name]) >= 5:
        self.last_live_check[cam_name] = time.time()
        link = database.run_get("links", cam_name)
        if type(link) == list: link = link[0] # todo, flakey?
        if link != self.src[cam_name]:
          self.src[cam_name] = link
          self.hls_proc[cam_name], self.proc[cam_name] = self._open_ffmpeg(cam_name)
        if global_settings.clearcam_user() and not self.vod[cam_name]: threading.Thread(target=self.check_upload_link, args=(cam_name,), daemon=True).start()
      if (time.time() - self.last_counter_update[cam_name]) >= 5: #update counter every 5 secs
        self.last_counter_update[cam_name] = time.time()

        counters = database.run_get("counters", cam_name)
        if counters not in [None, {}]:
          if counters.reset:
            self.counter[cam_name].reset_counts()
            self.counter[cam_name].reset = False
        database.run_put("counters", cam_name, self.counter[cam_name])
        
        alerts = database.run_get("alerts", cam_name)
        for id,a in alerts.items():
          if not a.new: continue
          a.new = False
          database.run_put("alerts", cam_name, a, id=id)
          if a is None:
            del self.alert_counters[cam_name][id]
            continue
          self.alert_counters[cam_name][id] = a
          for c in a.classes: classes.add(str(c))

        self.alert_counters[cam_name] = {i:a for i,a in self.alert_counters[cam_name].items() if i in alerts}
        
        new_settings = database.run_get("settings", cam_name)
        if self.settings[cam_name] is not None and new_settings != self.settings[cam_name] and is_vod(cam_name):
          self.reset_vod(cam_name)
          if "reset" in new_settings: del new_settings["reset"]
        self.settings[cam_name] = new_settings
          
      if global_settings.clearcam_user() and not self.vod[cam_name] and cam_name in self.live_link and (link:=self.live_link[cam_name]) and (time.time() - self.last_live_seg[cam_name]) >= 4:
        self.last_live_seg[cam_name] = time.time()
        threading.Thread(target=self.upload_live_segment, args=(link, cam_name,), daemon=True).start()
    else: self.count[cam_name]+=1

  def upload_live_segment(self, link, cam_name):
    self.last_live_seg[cam_name] = time.time()
    mp4_filename = f"segment.mp4"
    export_clip(self.current_stream_dir_raw[cam_name], Path(mp4_filename), live=True)
    encrypt_file(Path(mp4_filename), Path(f"""{mp4_filename}.aes"""), global_settings.key)
    Path(mp4_filename).unlink()
    upload_to_r2(file_path=Path(f"""{mp4_filename}.aes"""), signed_url=link)

  def check_upload_link(self, cam_name="camera"):
      query_params = urllib.parse.urlencode({
          "name": quote(cam_name),
          "session_token": global_settings.userID
      })
      url = f"https://clearcam.org/get_stream_upload_link?{query_params}"
      
      req = urllib.request.Request(url)
      with urllib.request.urlopen(req) as response:
          if response.status == 200:
              response_data = json.loads(response.read().decode('utf-8'))
              upload_link = response_data.get("upload_link")
              alerts_on_res = response_data.get("alerts_on")
              with self.live_link_lock[cam_name]: self.live_link[cam_name] = upload_link
              alerts_on[cam_name] = (alerts_on_res == 1)
          else:
              if cam_name in self.live_link: self.live_link[cam_name] = None

  def reset_vod(self, cam_name):
    self.cap[cam_name] = cv2.VideoCapture(self.src[cam_name]) # reset video on settings change
    shutil.rmtree(BASE_DIR / "cameras" / cam_name / "objects", ignore_errors=True)
    shutil.rmtree(BASE_DIR / "cameras" / cam_name / "faces", ignore_errors=True)
    shutil.rmtree(BASE_DIR / "cameras" / cam_name / "event_images", ignore_errors=True)

  def run_inference(self, frame, cam_name, dets=None):
    global model
    preds = dets if dets is not None else jit_infer(model, Tensor(frame), yolo_jit_cache).numpy()
    thresh = (self.settings[cam_name].get("threshold") if self.settings[cam_name] else 0.5) or 0.5 #todo clean!
    online_targets = self.tracker[cam_name].update(preds, thresh)
    online_targets = [p for p in online_targets if (classes is None or str(int(p.class_id)) in classes)]
    preds = []
    for x in online_targets:
      if x.tracklet_len < 1: continue # dont alert for 1 frame, too many false positives.  
      # add to objects, regarless of speed
      if x.track_id not in self.pred_occs[cam_name]: self.pred_occs[cam_name][x.track_id] = [time.time()]
      if (len(self.pred_occs[cam_name][x.track_id]) < 20 and (time.time() - self.pred_occs[cam_name][x.track_id][-1]) > 1) or (time.time() - self.pred_occs[cam_name][x.track_id][-1]) > 10:
        self. pred_occs[cam_name][x.track_id].append(time.time())
        ts = round((self.cap[cam_name].get(cv2.CAP_PROP_POS_FRAMES) / self.src_fps[cam_name]) - 5,1) if self.vod[cam_name] else round((time.time() - self.start_time[cam_name] - 5),1)
        self.save_object(x, ts, cam_name=cam_name)

      if x.speed < 2.5: continue #min speed, don't detect still objects, they jitter too. # TODO what's the best min value?
      outside = False
      if hasattr(self, "settings") and self.settings[cam_name] is not None and self.settings[cam_name].get("coords"):
        scaled_coors = np.array(self.settings[cam_name]["coords"])
        scaled_coors[:,] *= [frame.shape[1], frame.shape[0]] # decimal to full
        outside = point_not_in_polygon([[x.tlwh[0], x.tlwh[1]],[(x.tlwh[0]+x.tlwh[2]), x.tlwh[1]],[(x.tlwh[0]), (x.tlwh[1]+x.tlwh[3])],[(x.tlwh[0]+x.tlwh[2]), (x.tlwh[1]+x.tlwh[3])]], scaled_coors)
        outside = outside ^ self.settings[cam_name]["outside"]
      non_zone_alert = False
      if outside: # check if any alerts don't use zone
        for _, alert in self.alert_counters[cam_name].items():
          if not alert.zone:
            non_zone_alert = True
            break
        if not non_zone_alert and outside: continue
      preds.append(np.array([x.tlwh[0],x.tlwh[1],(x.tlwh[0]+x.tlwh[2]),(x.tlwh[1]+x.tlwh[3]),x.score,x.class_id,x.track_id]))
      if (classes is None or str(int(x.class_id)) in classes):
        new = int(x.track_id) not in self.object_set[cam_name]
        new_in_zone = int(x.track_id) not in self.object_set_zone[cam_name] and not outside
        if new:
          self.object_set[cam_name].add(int(x.track_id))
          self.counter[cam_name].add(int(x.class_id))
        if new_in_zone: self.object_set_zone[cam_name].add(int(x.track_id))
        for _, alert in self.alert_counters[cam_name].items():
          if not alert.get_counts()[1] and ((new and not alert.zone) or (new_in_zone and alert.zone)): alert.add(int(x.class_id))
  
    preds = np.array(preds)
    return preds, frame

def is_bright_color(color):
  r, g, b = color
  brightness = (r * 299 + g * 587 + b * 114) / 1000
  return brightness > 127

def draw_predictions(frame, preds, color_dict, cam_name=None):
  for x1, y1, x2, y2, conf, cls, _ in preds:
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    label = f"{class_labels[int(cls)]}:{conf:.2f}"
    color = color_dict[class_labels[int(cls)]]
    frame = draw_rectangle_numpy(frame, (x1, y1), (x2, y2), color, 3)
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    font_color = (0, 0, 0) if is_bright_color(color) else (255, 255, 255)
    frame = draw_rectangle_numpy(frame, (x1, y1 - text_height - 10), (x1 + text_width + 2, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1, cv2.LINE_AA)
  if cam_name and cam_name in last_ocr_by_cam:
    ocr_info = last_ocr_by_cam[cam_name]
    if time.time() - ocr_info.get("ts_epoch", 0) <= 15:
      ocr_label = f"OCR [{ocr_info.get('engine','')}]: {ocr_info.get('text','')} ({int(ocr_info.get('conf',0)*100)}%)"
      (tw, th), _ = cv2.getTextSize(ocr_label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
      frame = draw_rectangle_numpy(frame, (10, 10), (18 + tw, 22 + th), (20, 20, 20), -1)
      cv2.putText(frame, ocr_label, (14, 16 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2, cv2.LINE_AA)
  return frame

def point_not_in_polygon(coords, poly):
    n = len(poly)
    for j in range(len(coords)):
      inside = False
      p1x, p1y = poly[0]
      for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if coords[j][1] > min(p1y, p2y):
          if coords[j][1] <= max(p1y, p2y):
            if coords[j][0] <= max(p1x, p2x):
              if p1y != p2y:
                x_intersect = (coords[j][1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
              else:
                x_intersect = p1x
              if p1x == p2x or coords[j][0] <= x_intersect: inside = not inside
        p1x, p1y = p2x, p2y
      if inside:
        return False
    return True

def run_encode_text(clip, text): return clip.model._encode_text(text, realize=True)

def run_search(clip, image_text, top_k, cam_name, selected_dir): return clip.search(image_text, top_k, cam_name, selected_dir)

def run_clip(clip, im, top_k, cam_name, selected_dir, is_face):
  im = clip.preprocess_face(im) if is_face else clip.preprocess_clip(im)
  if im is not None:
    embedding = clip.adaface(Tensor(im, dtype=dtypes.float32)).numpy() if is_face else jit_infer(clip.model.precompute_embedding, Tensor(im, dtype=dtypes.float32), jit_cache=jit_cache).numpy()
    res = clip.search(None, top_k, cam_name, selected_dir, embedding, is_face)
  else:
    res = []
  return res

class HLSRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args): pass # don't print stuff

    def send_results(self, results, start=0, count=100):
      image_data = []
      for path_str, score in results:
        if score < 0.21: break
        img_path = Path(path_str).resolve()
        ts = event_img_info(img_path.stem)["ts"]
        parts = img_path.parts
        cam_index = parts.index("cameras") + 1
        cam_n = parts[cam_index]
        rel = img_path.relative_to((BASE_DIR / "cameras").resolve()).as_posix()
        image_url = f"/{rel}"
        image_data.append({
          "url": image_url,
          "timestamp": ts,
          "filename": img_path.name,
          "cam_name": cam_n,
          "folder": img_path.parts[-2],
          "score": score,
          "ocr": ocr_by_image.get(str(img_path), ""),
        })
      image_data = image_data[start:start+count]
      response_data = {
        "images": image_data,
        "count": len(image_data),
      }
      self.send_200(response_data)

    def send_200(self, body=None):
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      self.wfile.write(json.dumps(body).encode("utf-8"))

    def get_camera_path(self, cam_name=None):
        if cam_name:
            return BASE_DIR / "cameras" / cam_name / "streams"
        return BASE_DIR / "cameras"
    
    def do_GET(self):
        parsed_path = urlparse(unquote(self.path))
        query = parse_qs(parsed_path.query)
        cam_name = query.get("cam", [None])[0]

        if parsed_path.path in ("/favicon.ico", "/favicon.png"):
          fav_p = Path("assets/favicon.png")
          if fav_p.is_file():
            data_b = fav_p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "public, max-age=86400")
            self.send_header("Content-Length", str(len(data_b)))
            self.end_headers()
            self.wfile.write(data_b)
            return

        if parsed_path.path.startswith("/assets/"):
          rel_asset = parsed_path.path[len("/assets/"):].lstrip("/")
          assets_root = Path("assets").resolve()
          try:
            asset_file = (assets_root / rel_asset).resolve()
            asset_file.relative_to(assets_root)
          except Exception:
            self.send_error(404)
            return
          if not asset_file.is_file():
            self.send_error(404)
            return
          mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml", ".ico": "image/x-icon"}
          data_b = asset_file.read_bytes()
          self.send_response(200)
          self.send_header("Content-Type", mime_map.get(asset_file.suffix.lower(), "application/octet-stream"))
          self.send_header("Cache-Control", "public, max-age=86400")
          self.send_header("Content-Length", str(len(data_b)))
          self.end_headers()
          self.wfile.write(data_b)
          return

        if parsed_path.path == "/discover_cameras":
          subnet = query.get("subnet", [None])[0]
          user = query.get("user", [""])[0]
          password = query.get("password", [""])[0]
          discovered = discover_cameras(subnet_or_range=subnet, user=user, password=password)
          existing = database.run_get("links", None) or {}
          existing_ips = set()
          for _, link_url in existing.items():
            if isinstance(link_url, str):
              m = re.search(r"(\d{1,3}(?:\.\d{1,3}){3})", link_url)
              if m: existing_ips.add(m.group(1))
          for item in discovered:
            item["already_added"] = item["ip"] in existing_ips
          self.send_200({"cameras": discovered, "count": len(discovered)})
          return

        if parsed_path.path == "/camera_status":
          cams_db = database.run_get("links", None) or {}
          out = {}
          for cname, link_url in cams_db.items():
            st = dict(cam_runtime_status.get(cname, {}))
            if not st:
              st = {"protocol": _detect_protocol_name(link_url), "state": "starting", "url": _redact_url(link_url), "fps": 0.0}
            st["last_ocr"] = last_ocr_by_cam.get(cname)
            out[cname] = st
          self.send_200(out)
          return

        if parsed_path.path == "/live_jpg":
          if not cam_name or cam is None or cam.raw_frame.get(cam_name) is None:
            self.send_error(404, "No live frame yet")
            return
          ok, jpg = cv2.imencode(".jpg", cam.raw_frame[cam_name], [cv2.IMWRITE_JPEG_QUALITY, 82])
          if not ok:
            self.send_error(500, "Encode failed")
            return
          data_bytes = jpg.tobytes()
          self.send_response(200)
          self.send_header("Content-Type", "image/jpeg")
          self.send_header("Cache-Control", "no-store")
          self.send_header("Content-Length", str(len(data_bytes)))
          self.end_headers()
          self.wfile.write(data_bytes)
          return

        if parsed_path.path == "/onvif_inspect":
          host = query.get("host", [""])[0]
          port = int(query.get("port", ["80"])[0])
          user = query.get("user", ["admin"])[0]
          password = query.get("password", [""])[0]
          try:
            oclient = ONVIFDirectClient(host=host, port=port, user=user, password=password)
            dev_info = oclient.get_device_info()
            profiles = oclient.get_profiles()
            stream_uri = oclient.get_stream_uri(profile_index=0) if profiles else ""
            self.send_200({"ok": True, "device": dev_info, "profiles": profiles, "stream_uri": stream_uri})
          except Exception as exc:
            self.send_200({"ok": False, "error": str(exc)})
          return

        if parsed_path.path == "/onvif_ptz":
          if not cam_name:
            self.send_error(400, "Missing cam")
            return
          link_url = database.run_get("links", cam_name)
          parsed_u = urlparse(link_url if isinstance(link_url, str) else "")
          host = query.get("host", [parsed_u.hostname or ""])[0]
          port = int(query.get("port", ["80"])[0])
          user = query.get("user", [unquote(parsed_u.username) if parsed_u.username else "admin"])[0]
          password = query.get("password", [unquote(parsed_u.password) if parsed_u.password else ""])[0]
          action = query.get("action", ["stop"])[0]
          try:
            oclient = ONVIFDirectClient(host=host, port=port, user=user, password=password)
            if action == "move":
              pan = float(query.get("pan", ["0"])[0])
              tilt = float(query.get("tilt", ["0"])[0])
              zoom = float(query.get("zoom", ["0"])[0])
              oclient.ptz_continuous_move(pan=pan, tilt=tilt, zoom=zoom)
            else:
              oclient.ptz_stop()
            self.send_200({"ok": True})
          except Exception as exc:
            self.send_200({"ok": False, "error": str(exc)})
          return

        if parsed_path.path == "/pumatronix_api":
          ip = query.get("ip", [""])[0]
          port = int(query.get("port", ["80"])[0])
          user = query.get("user", [None])[0]
          password = query.get("password", [None])[0]
          if cam_name and not ip:
            src_u = database.run_get("links", cam_name)
            if isinstance(src_u, str):
              pcfg = parse_pumatronix_uri(src_u)
              ip = str(pcfg["ip"])
              port = int(pcfg["port"])
              user = user or pcfg.get("user")
              password = password or pcfg.get("password")
          action = query.get("action", ["config"])[0]
          try:
            pclient = PumatronixClient(ip=ip, port=port, user=user, password=password)
            if action == "config":
              param = query.get("param", ["tudo"])[0]
              self.send_200({"ok": True, "ip": ip, "config": pclient.get_config(param), "flash_modes": FLASH_MODES, "resolutions": SUPPORTED_RESOLUTIONS})
            elif action == "set_flash":
              modo = int(query.get("modo", ["5"])[0])
              delay = int(query.get("delay", ["5000"])[0])
              self.send_200({"ok": True, "result": pclient.set_flash(modo, delay)})
            elif action == "set_config":
              k = query.get("key", [""])[0]
              v = query.get("val", [""])[0]
              self.send_200({"ok": True, "result": pclient.set_config({k: v})})
            elif action in ("snapshot", "trigger"):
              qual = int(query.get("quality", ["95"])[0])
              raw_jpg = pclient.trigger_snapshot() if action == "trigger" else pclient.snapshot(quality=qual)
              img_bgr = cv2.imdecode(np.frombuffer(raw_jpg, np.uint8), cv2.IMREAD_COLOR)
              ocr_res = ocr_engine.read_image(
                  img_bgr,
                  engine=getattr(global_settings, "ocr_engine", "rapidocr"),
                  plate_mode=getattr(global_settings, "ocr_plate_mode", False),
              ) if img_bgr is not None else []
              b64_jpg = base64.b64encode(raw_jpg).decode("ascii")
              self.send_200({"ok": True, "image_b64": f"data:image/jpeg;base64,{b64_jpg}", "ocr": ocr_res})
            elif action == "watchdog":
              enable = query.get("enable", ["true"])[0].lower() == "true"
              if enable and cam_name:
                puma_watchdog.start(cam_name, pclient)
                self.send_200({"ok": True, "watchdog": True})
              else:
                if cam_name: puma_watchdog.stop(cam_name)
                self.send_200({"ok": True, "ping": pclient.ping_watchdog(), "watchdog": False})
            elif action == "audit":
              self.send_200({"ok": True, "audit": pclient.get_audit_configs()})
            else:
              self.send_200({"ok": False, "error": f"Unknown action: {action}"})
          except Exception as exc:
            self.send_200({"ok": False, "error": str(exc)})
          return

        if parsed_path.path == "/isup_status":
          self.send_200(isup_gateway.status())
          return

        if parsed_path.path == "/ocr_events":
          q_filter = (query.get("q", [""])[0] or "").strip().upper()
          limit = int(query.get("limit", ["100"])[0])
          all_ocr = database.run_get("ocr_events", cam_name) if cam_name else database.run_get("ocr_events", None)
          flat = []
          if isinstance(all_ocr, dict):
            for k, val in all_ocr.items():
              if isinstance(val, dict):
                if "text" in val:
                  flat.append({kk: vv for kk, vv in val.items() if not isinstance(vv, dict)})
                for _, subval in val.items():
                  if isinstance(subval, dict) and "text" in subval:
                    flat.append(subval)
          if q_filter:
            flat = [x for x in flat if q_filter in str(x.get("text", "")).upper() or q_filter in str(x.get("plate") or "").upper()]
          flat.sort(key=lambda x: x.get("ts_epoch", 0), reverse=True)
          self.send_200({"events": flat[:limit], "engines": ocr_engine.available_engines()})
          return

        if parsed_path.path == "/export":
          date = query.get("date", [None])[0]
          stream_dir = (Path("data") / "cameras" / cam_name / "streams" / date)
          output_path = stream_dir / "_export.mp4"
          m4s_files = sorted(stream_dir.glob("stream_*.m4s"))
          ffmpeg_path = find_ffmpeg()
          command = [
            ffmpeg_path,
            "-y",
            "-loglevel", "error",
            "-i", "pipe:0",
            "-c", "copy",
            "-movflags", "+faststart",
            str(output_path),
          ]
          proc = subprocess.Popen(command, stdin=subprocess.PIPE)
          try:
            with (stream_dir / "init.mp4").open("rb") as f: shutil.copyfileobj(f, proc.stdin)
            for m4s in m4s_files:
              with m4s.open("rb") as f: shutil.copyfileobj(f, proc.stdin)
          finally:
            proc.stdin.close()
          if proc.wait() != 0: output_path.unlink(missing_ok=True)
          self.send_response(200)
          self.send_header("Content-Type", "video/mp4")
          self.send_header("Content-Disposition", f'attachment; filename="{cam_name}_{date}.mp4"',)
          self.send_header("Content-Length", str(output_path.stat().st_size))
          self.end_headers()
          with output_path.open("rb") as f: shutil.copyfileobj(f, self.wfile)
          output_path.unlink()

        if parsed_path.path == "/set_max_storage":
          max_gb = float(query.get("max", [None])[0])
          self.server.max_gb = max_gb
          database.run_put("max_storage", "all", max_gb)
          self.send_200()
          return
        
        if parsed_path.path == "/get_global_settings":
          self.send_200(secret_settings(global_settings).__dict__)
          return
        if parsed_path.path == "/get_max_storage":
          self.send_200(body={"max_gb":self.server.max_gb})
          return

        if parsed_path.path == "/list_cameras":
          cams = database.run_get("links", None)
          progs = database.run_get("analysis_prog", None)
          cam_progress = {cam_name: progs.get(cam_name, None) for cam_name in cams}
          self.send_200(cam_progress)
          return

        if parsed_path.path == "/live_frame":
          target_cam = (query.get("cam", [""])[0] or "").strip()
          jpg_q = max(40, min(95, int(query.get("q", ["82"])[0] or 82)))
          frame_bgr = None
          if cam is not None and target_cam:
            frame_bgr = cam.raw_frame.get(target_cam)
          if frame_bgr is None:
            self.send_error(404, "No live frame available")
            return
          ok_enc, buf_enc = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpg_q])
          if not ok_enc:
            self.send_error(500, "Encode error")
            return
          jpg_bytes = buf_enc.tobytes()
          self.send_response(200)
          self.send_header("Content-Type", "image/jpeg")
          self.send_header("Content-Length", str(len(jpg_bytes)))
          self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
          self.end_headers()
          self.wfile.write(jpg_bytes)
          return

        if parsed_path.path == "/live_mjpeg":
          target_cam = (query.get("cam", [""])[0] or "").strip()
          max_fps = max(1.0, min(20.0, float(query.get("fps", ["8"])[0] or 8.0)))
          jpg_q = max(40, min(92, int(query.get("q", ["78"])[0] or 78)))
          min_dt = 1.0 / max_fps
          if not target_cam:
            self.send_error(400, "Missing cam")
            return
          self.send_response(200)
          self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
          self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
          self.send_header("Pragma", "no-cache")
          self.send_header("Connection", "close")
          self.end_headers()
          last_sent_fnum = -1
          last_send_t = 0.0
          try:
            while True:
              if cam is None or target_cam not in cam.src:
                break
              f_bgr = cam.raw_frame.get(target_cam)
              f_num = cam.frame_num.get(target_cam, -1)
              now_m = time.time()
              if f_bgr is None:
                time.sleep(0.2)
                continue
              # Envia novo frame ou keepalive a cada 1.2s para manter stream fluido
              if f_num == last_sent_fnum and (now_m - last_send_t) < 1.2:
                time.sleep(0.04)
                continue
              elapsed = now_m - last_send_t
              if elapsed < min_dt:
                time.sleep(min_dt - elapsed)
              f_bgr = cam.raw_frame.get(target_cam)
              if f_bgr is None:
                continue
              ok_enc, buf_enc = cv2.imencode(".jpg", f_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpg_q])
              if not ok_enc:
                time.sleep(0.05)
                continue
              payload = buf_enc.tobytes()
              header = (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
              )
              self.wfile.write(header)
              self.wfile.write(payload)
              self.wfile.write(b"\r\n")
              self.wfile.flush()
              last_sent_fnum = f_num
              last_send_t = time.time()
          except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
          return

        if parsed_path.path == "/list_days":          
          base_path = "data/cameras"
          days = set()
          date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
          if os.path.exists(base_path):
            for cam_name in os.listdir(base_path):
              cam_path = os.path.join(base_path, cam_name, "streams")    
              if os.path.exists(cam_path):
                for date_folder in os.listdir(cam_path):
                  date_folder_path = os.path.join(cam_path, date_folder)
                  if os.path.isdir(date_folder_path) and date_pattern.match(date_folder): days.add(date_folder)
          days_list = sorted(list(days), reverse=True, key=lambda x: datetime.strptime(x, "%Y-%m-%d"))
          self.send_200(days_list)
          return

        if parsed_path.path == '/add_camera':
            cam_name = query.get("cam_name", [None])[0]
            src = query.get("src", [None])[0]
            
            if not cam_name or not src:
                self.send_error(400, "Missing cam_name or src")
                return
            
            database.run_put("links", cam_name, src)
            self.send_response(302)
            self.send_header('Location', '/')
            self.end_headers()
            return
        
        if parsed_path.path == "/edit_settings":
            if not cam_name:
                self.send_error(400, "Missing cam or id")
                return
            zone = database.run_get("settings", cam_name)
            if zone is None: zone = {}
            coords_json = query.get("coords", [None])[0]
            if coords_json is not None:
              coords = json.loads(coords_json)
              if isinstance(coords, list):
                if len(coords) >= 3:
                  zone["coords"] = [[float(x), float(y)] for x, y in coords]
                else:
                  if "coords" in zone: del zone["coords"]
            zone["is_notif"] = (str(is_notif).lower() == "true") if (is_notif := query.get("is_notif", [None])[0]) is not None else zone.get("is_notif")
            zone["outside"] = (str(outside).lower() == "true") if (outside := query.get("outside", [None])[0]) is not None else zone.get("outside")
            query.get("threshold", [None])[0] is not None and zone.update({"threshold": float(query.get("threshold", [None])[0])}) #need the val  
            database.run_put("settings", cam_name, zone) # todo, key for each
            if (url := query.get("url")) is not None: database.run_put("links", cam_name, url[0])

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return

        if parsed_path.path == "/edit_alert":
            if not cam_name:
                self.send_error(400, "Missing cam or id")
                return

            raw_alerts = database.run_get("alerts", cam_name)
            alert = None
            alert_id = query.get("id", [None])[0]
            is_on = query.get("is_on", [None])[0]
            zone = query.get("zone", [None])[0]
            is_notif = query.get("is_notif", [None])[0]
            desc = query.get("desc", [None])[0]
            threshold = query.get("threshold", [None])[0]
            if threshold is not None: threshold = float(threshold) / 100
            if alert_id is None: # no id, add alert
                window = query.get("window", [None])[0]
                max_count = query.get("max", [None])[0]
                class_ids = query.get("class_ids", [None])[0]
                sched = json.loads(query.get("sched", ["[[0,86400],[0,86400],[0,86400],[0,86400],[0,86400],[0,86400],[0,86400]]"])[0]) # todo, weekly
                if window: window = int(window)
                max_count = int(max_count)
                classes = [int(c.strip()) for c in class_ids.split(",")]
                alert_id = str(uuid.uuid4())
                alert = RollingClassCounter(
                        window_seconds=window,
                        max=max_count,
                        classes=classes,
                        sched=sched,
                        cam_name=cam_name,
                        desc=desc,
                        threshold=threshold
                    )
                raw_alerts[alert_id] = alert
            else:
              if is_on is not None or is_notif is not None or zone is not None:
                if is_on is not None: raw_alerts[alert_id].is_on = str(is_on).lower() == "true"
                if is_notif is not None: raw_alerts[alert_id].is_notif = str(is_notif).lower() == "true"
                if zone is not None: raw_alerts[alert_id].zone = str(zone).lower() == "true"
                if desc is not None: raw_alerts[alert_id].desc = desc
                if threshold is not None: raw_alerts[alert_id].threshold = threshold
                alert = raw_alerts[alert_id]
                alert.new = True
              else:
                del raw_alerts[alert_id]
            if alert is not None:
              database.run_put("alerts", cam_name, alert, alert_id)
            else:
              database.run_delete("alerts", cam_name, alert_id)
            
            # make vod reset
            settings = database.run_get("settings", cam_name)
            if settings is None: settings = {}
            settings["reset"] = True
            database.run_put("settings", cam_name, settings)

 
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return

        if parsed_path.path == "/get_settings":
            zone = database.run_get("settings",cam_name)
            if zone is not None:
              if cam_name in zone and "settings" in zone[cam_name]: zone = zone[cam_name]["settings"]
            else:
              zone = {}
            
            self.send_200(zone)
            return

        if parsed_path.path == "/get_alerts":
            if not cam_name:
                self.send_error(400, "Missing cam parameter")
                return

            raw_alerts = database.run_get("alerts",cam_name)
            alert_info = []
            for key,alert in raw_alerts.items():
                sched = alert.sched if alert.sched else [[0,86399],True,True,True,True,True,True,True]
                alert_info.append({
                    "window": alert.window,
                    "max": alert.max,
                    "classes": list(alert.classes),
                    "id": str(key),
                    "sched": sched,
                    "is_on": alert.is_on,
                    "is_notif": alert.is_notif,
                    "zone": alert.zone,
                    "desc": alert.desc,
                    "threshold": alert.threshold,
                })
            self.send_200(alert_info)
            return

        if parsed_path.path == '/delete_camera':
            cam_name = query.get("cam_name", [None])[0]
            if not cam_name:
                self.send_error(400, "Missing cam_name parameter")
                return
            
            try:
              shutil.rmtree(BASE_DIR / "cameras" / cam_name, ignore_errors=True)
              if os.path.isfile(database.run_get("links", None)[cam_name]): os.remove(database.run_get("links", None)[cam_name])
              # todo clean
              alerts = database.run_get("alerts", cam_name)
              for id, _ in alerts.items():
                database.run_delete("alerts", cam_name, id=id)
              database.run_delete("links", cam_name)
              database.run_delete("analysis_prog", cam_name)
              database.run_delete("settings", cam_name)
              database.run_delete("counters", cam_name)
            except Exception as e:
              self.send_error(500, f"Error deleting camera: {e}")
              return

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"deleted"}')
            return

        if parsed_path.path == "/get_counts": # todo error fetching counts on first
            if not cam_name:
                self.send_error(400, "Missing cam parameter")
                return
            
            counter = database.run_get("counters", cam_name)
            if counter:
              labeled_counts = {
                class_labels[int(k)]: len(v)
                for k, v in counter.data.items()
                if int(k) < len(class_labels)
              }
              self.send_200(labeled_counts)
              return
            else:
              database.run_put("counters", cam_name, RollingClassCounter(cam_name=cam_name))
              self.send_200([])
      

        if parsed_path.path == "/reset_counts":
          if not cam_name:
            self.send_error(400, "Missing cam parameter")
            return
          counter = database.run_get("counters",cam_name)
          if counter: counter.reset_counts()
          database.run_put("counters", cam_name, counter)
          self.send_response(200)
          self.send_header("Content-Type", "application/json")
          self.end_headers()
          self.wfile.write(b"{}")
          return


        if parsed_path.path in ("/galeria", "/lpr_gallery"):
          with open("plate_gallery.html", "r", encoding="utf-8") as f: html = f.read()
          self.send_response(200)
          self.send_header("Content-Type", "text/html; charset=utf-8")
          self.end_headers()
          self.wfile.write(html.encode("utf-8"))
          return

        if parsed_path.path in ("/veiculos", "/leituras", "/busca_veiculos"):
          with open("vehicle_search.html", "r", encoding="utf-8") as f: html = f.read()
          self.send_response(200)
          self.send_header("Content-Type", "text/html; charset=utf-8")
          self.end_headers()
          self.wfile.write(html.encode("utf-8"))
          return

        if parsed_path.path == "/api/plate_registry":
          reg_all = database.run_get("plate_registry", None) or {}
          reg_items = []
          if isinstance(reg_all, dict):
            for pkey, rval in reg_all.items():
              if isinstance(rval, dict):
                rc = dict(rval)
                rc.setdefault("plate_key", pkey)
                reg_items.append(rc)
          reg_items.sort(key=lambda x: float(x.get("updated_epoch", 0)), reverse=True)
          self.send_200({"ok": True, "items": reg_items})
          return

        if parsed_path.path == "/api/vehicle_search":
          q_filter = (query.get("q", [""])[0] or "").strip().upper()
          plate_filter = normalize_plate_key(query.get("plate", [""])[0] or "")
          color_filter = (query.get("color", [""])[0] or "").strip().lower()
          model_filter = (query.get("model", [""])[0] or "").strip().upper()
          year_filter = (query.get("year", [""])[0] or "").strip()
          uf_filter = (query.get("uf", [""])[0] or "").strip().upper()
          status_filter = (query.get("status", [""])[0] or "").strip().lower()
          cam_filter = (query.get("cam", [""])[0] or "").strip()
          from_filter = (query.get("from", [""])[0] or "").strip()
          to_filter = (query.get("to", [""])[0] or "").strip()
          only_plates = (query.get("only_plates", ["0"])[0] or "0") in ("1", "true", "yes")
          limit = max(1, min(2000, int(query.get("limit", ["300"])[0] or 300)))

          from_epoch = None
          to_epoch = None
          for raw_dt, is_end in ((from_filter, False), (to_filter, True)):
            if not raw_dt:
              continue
            norm_dt = raw_dt.replace("T", " ").strip()
            parsed_ep = None
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
              try:
                dt_v = datetime.strptime(norm_dt, fmt)
                if fmt == "%Y-%m-%d" and is_end:
                  dt_v = dt_v.replace(hour=23, minute=59, second=59)
                elif fmt == "%Y-%m-%d %H:%M" and is_end:
                  dt_v = dt_v.replace(second=59)
                parsed_ep = dt_v.timestamp()
                break
              except ValueError:
                continue
            if parsed_ep is not None:
              if is_end:
                to_epoch = parsed_ep
              else:
                from_epoch = parsed_ep

          raw_all = database.run_get("plate_gallery", None) or {}
          reg_all = database.run_get("plate_registry", None) or {}
          all_items = []
          color_facets = defaultdict(int)
          model_facets = defaultdict(int)
          uf_facets = set()

          if isinstance(raw_all, dict):
            for uid, rec in raw_all.items():
              if not isinstance(rec, dict) or ("scene_url" not in rec and "area_url" not in rec):
                continue
              rc = dict(rec)
              rc.setdefault("uid", uid)
              pkey = normalize_plate_key(rc.get("plate") or rc.get("text"))
              reg_entry = reg_all.get(pkey) if (pkey and isinstance(reg_all, dict)) else None

              ck = str(rc.get("color_key") or "indefinida").lower()
              if ck not in VEHICLE_COLORS:
                ck = "indefinida"
              cmeta = VEHICLE_COLORS[ck]
              rc["color_key"] = ck
              rc["vehicle_color"] = rc.get("vehicle_color") or cmeta["label"]
              rc["color_hex"] = rc.get("color_hex") or cmeta["hex"]
              rc["vehicle_model"] = str(rc.get("vehicle_model") or "PASSEIO / HATCH")
              rc["vehicle_type"] = str(rc.get("vehicle_type") or "Automóvel")
              rc["vehicle_uf"] = str(infer_plate_uf(rc.get("plate")) or (rc.get("vehicle_uf") if rc.get("plate") else "") or "")
              rc["vehicle_year"] = str(rc.get("vehicle_year") or "")
              rc["list_status"] = str(rc.get("list_status") or "sem_restricao")
              rc["notes"] = str(rc.get("notes") or "")

              if isinstance(reg_entry, dict):
                reg_ck = str(reg_entry.get("color_key") or "").lower()
                if reg_ck in VEHICLE_COLORS and reg_ck != "indefinida":
                  rc["color_key"] = reg_ck
                  rc["vehicle_color"] = VEHICLE_COLORS[reg_ck]["label"]
                  rc["color_hex"] = VEHICLE_COLORS[reg_ck]["hex"]
                if reg_entry.get("vehicle_model"):
                  rc["vehicle_model"] = str(reg_entry["vehicle_model"]).upper()
                if reg_entry.get("vehicle_type"):
                  rc["vehicle_type"] = str(reg_entry["vehicle_type"])
                if reg_entry.get("vehicle_year"):
                  rc["vehicle_year"] = str(reg_entry["vehicle_year"])
                if reg_entry.get("vehicle_uf"):
                  rc["vehicle_uf"] = str(reg_entry["vehicle_uf"]).upper()
                if reg_entry.get("list_status"):
                  rc["list_status"] = str(reg_entry["list_status"])
                if reg_entry.get("notes"):
                  rc["notes"] = str(reg_entry["notes"])

              color_facets[rc["color_key"]] += 1
              if rc["vehicle_model"]:
                model_facets[rc["vehicle_model"]] += 1
              if rc["vehicle_uf"]:
                uf_facets.add(rc["vehicle_uf"])
              all_items.append(rc)

          total_all = len(all_items)
          filtered = all_items

          if from_epoch is not None:
            filtered = [x for x in filtered if float(x.get("ts_epoch", 0)) >= from_epoch]
          if to_epoch is not None:
            filtered = [x for x in filtered if float(x.get("ts_epoch", 0)) <= to_epoch]
          if cam_filter:
            filtered = [x for x in filtered if x.get("cam_name") == cam_filter]
          if only_plates:
            filtered = [x for x in filtered if x.get("plate")]
          if plate_filter:
            filtered = [
              x for x in filtered
              if plate_filter in normalize_plate_key(x.get("plate") or x.get("text"))
            ]
          if color_filter:
            filtered = [x for x in filtered if x.get("color_key") == color_filter]
          if model_filter:
            filtered = [
              x for x in filtered
              if model_filter in str(x.get("vehicle_model") or "").upper()
              or model_filter in str(x.get("vehicle_type") or "").upper()
            ]
          if year_filter:
            filtered = [x for x in filtered if year_filter in str(x.get("vehicle_year") or "")]
          if uf_filter:
            filtered = [x for x in filtered if str(x.get("vehicle_uf") or "").upper() == uf_filter]
          if status_filter:
            filtered = [x for x in filtered if str(x.get("list_status") or "sem_restricao").lower() == status_filter]
          if q_filter:
            q_norm = normalize_plate_key(q_filter)
            filtered = [
              x for x in filtered
              if (q_norm and q_norm in normalize_plate_key(x.get("plate") or x.get("text")))
              or q_filter in str(x.get("vehicle_model") or "").upper()
              or q_filter in str(x.get("vehicle_type") or "").upper()
              or q_filter in str(x.get("vehicle_color") or "").upper()
              or q_filter in str(x.get("cam_name") or "").upper()
              or q_filter in str(x.get("notes") or "").upper()
            ]

          filtered.sort(key=lambda x: float(x.get("ts_epoch", 0)), reverse=True)
          unique_plates = len({normalize_plate_key(x.get("plate")) for x in filtered if x.get("plate")})
          sem_rest_cnt = sum(1 for x in filtered if x.get("list_status", "sem_restricao") == "sem_restricao")
          black_cnt = sum(1 for x in filtered if x.get("list_status") == "blacklist")
          white_cnt = sum(1 for x in filtered if x.get("list_status") == "whitelist")

          filt_color_counts = defaultdict(int)
          for x in filtered:
            if x.get("color_key") and x.get("color_key") != "indefinida":
              filt_color_counts[x["color_key"]] += 1
          top_color_label = "—"
          if filt_color_counts:
            top_ck = max(filt_color_counts.items(), key=lambda kv: kv[1])[0]
            top_color_label = VEHICLE_COLORS.get(top_ck, {}).get("label", top_ck)

          all_cams = sorted(list((database.run_get("links", None) or {}).keys()))
          online_cams = sum(1 for v in cam_runtime_status.values() if v.get("state") == "online")
          sorted_models = [k for k, _ in sorted(model_facets.items(), key=lambda kv: kv[1], reverse=True)]

          self.send_200({
            "items": filtered[:limit],
            "stats": {
              "total_all": total_all,
              "filtered_count": len(filtered),
              "unique_plates": unique_plates,
              "sem_restricao": sem_rest_cnt,
              "blacklist": black_cnt,
              "whitelist": white_cnt,
              "top_color": top_color_label,
              "cameras_online": online_cams,
              "cameras_total": len(all_cams),
              "registry_total": len(reg_all) if isinstance(reg_all, dict) else 0,
            },
            "facets": {
              "colors": dict(color_facets),
              "models": sorted_models,
              "ufs": sorted(list(uf_facets)),
              "cameras": all_cams,
            },
            "color_catalog": VEHICLE_COLORS,
          })
          return

        if parsed_path.path == "/api/plate_gallery":
          q_filter = (query.get("q", [""])[0] or "").strip().upper()
          cam_filter = (query.get("cam", [""])[0] or "").strip()
          date_filter = (query.get("date", [""])[0] or "").strip()
          only_plates = (query.get("only_plates", ["0"])[0] or "0") in ("1", "true", "yes")
          limit = int(query.get("limit", ["200"])[0])
          raw_all = database.run_get("plate_gallery", None)
          items = []
          dates_set = set()
          if isinstance(raw_all, dict):
            for uid, rec in raw_all.items():
              if isinstance(rec, dict) and ("scene_url" in rec or "area_url" in rec):
                rec_copy = dict(rec)
                rec_copy.setdefault("uid", uid)
                if rec_copy.get("folder"):
                  dates_set.add(str(rec_copy["folder"]))
                items.append(rec_copy)
          today_str = datetime.now().strftime("%Y-%m-%d")
          total_all = len(items)
          today_count = sum(1 for x in items if x.get("folder") == today_str)
          valid_plates_count = sum(1 for x in items if x.get("plate"))
          avg_conf = round(sum(float(x.get("conf", 0.0)) for x in items) / len(items) * 100, 1) if items else 0.0
          online_cams = sum(1 for v in cam_runtime_status.values() if v.get("state") == "online")

          if cam_filter:
            items = [x for x in items if x.get("cam_name") == cam_filter]
          if date_filter:
            items = [x for x in items if x.get("folder") == date_filter]
          if only_plates:
            items = [x for x in items if x.get("plate")]
          if q_filter:
            items = [
              x for x in items
              if q_filter in str(x.get("plate") or "").upper()
              or q_filter in str(x.get("text") or "").upper()
              or q_filter in str(x.get("cam_name") or "").upper()
            ]
          items.sort(key=lambda x: float(x.get("ts_epoch", 0)), reverse=True)
          all_cams = sorted(list((database.run_get("links", None) or {}).keys()))
          self.send_200({
            "items": items[:limit],
            "stats": {
              "total": total_all,
              "today": today_count,
              "valid_plates": valid_plates_count,
              "avg_conf": avg_conf,
              "cameras_online": online_cams,
              "cameras_total": len(all_cams),
              "active_engine": getattr(global_settings, "ocr_engine", "rapidocr"),
            },
            "cameras": all_cams,
            "dates": sorted(list(dates_set), reverse=True),
            "engines": ocr_engine.available_engines(),
          })
          return

        if parsed_path.path == '/' and "cam" not in query:
          with open("mainview.html", "r", encoding="utf-8") as f: html = f.read()
          self.send_response(200)
          self.send_header('Content-type', 'text/html')
          self.end_headers()
          self.wfile.write(html.encode('utf-8'))
          return
                            
        if parsed_path.path == '/' or parsed_path.path == f'/{cam_name}':
            selected_dir = parse_qs(parsed_path.query).get("folder", [datetime.now().strftime("%Y-%m-%d")])[0]
            start_param = parse_qs(parsed_path.query).get("start", [None])[0]
            try:
                start_time = max(float(start_param),0) if start_param is not None else None
            except ValueError:
                start_time = None

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('cameraview.html', 'r', encoding='utf-8') as f: html = f.read()
            replacements = {
                '{selected_dir}': selected_dir,
                '{class_labels}': json.dumps(class_labels),
                '{start_time}': str(start_time) if start_time is not None else 'null',
                '{cam_name}': cam_name
            }
            for placeholder, value in replacements.items(): html = html.replace(placeholder, value)
            self.wfile.write(html.encode("utf-8"))
            return
        
        requested_path = parsed_path.path.lstrip('/')
        if requested_path.startswith("cameras/"):
            requested_path = requested_path[len("cameras/"):]

        try: # todo
          cam_name = requested_path[:requested_path.index("/")]
          vod = is_vod(cam_name)
          if vod and "preview.png" not in requested_path: requested_path = requested_path.rsplit("/", 2)[0] + "/video/" + requested_path.rsplit("/", 1)[1]
        except Exception:
          pass

        file_path = BASE_DIR / "cameras" / requested_path
        try: file_path = ((BASE_DIR / "cameras").resolve() / requested_path).resolve()
        except (OSError, RuntimeError):
          self.send_error(404)
          return
        try: _ = file_path.relative_to((BASE_DIR / "cameras").resolve())
        except:
          self.send_error(404)
          return

        if not file_path.exists():
            self.send_error(404)
            return

        self.send_response(200)
        if file_path.suffix == '.m3u8':
            self.send_header('Content-Type', 'application/vnd.apple.mpegurl')
            self.send_header('Cache-Control', 'no-cache')
        elif file_path.suffix == '.ts':
            self.send_header('Content-Type', 'video/MP2T')
        elif file_path.suffix == '.m4s':
            self.send_header('Content-Type', 'video/iso.segment')
        elif file_path.suffix == '.png':
            self.send_header('Content-Type', 'image/jpeg')
        elif file_path.suffix.lower() in ('.jpg', '.jpeg'):
            self.send_header('Content-Type', 'image/jpeg')
        self.end_headers()

        with open(file_path, 'rb') as f:
            shutil.copyfileobj(f, self.wfile)

    def do_POST(self):
        parsed_path = urlparse(self.path)

        if parsed_path.path == "/batch_add_cameras":
          content_length = int(self.headers.get("Content-Length", 0))
          body = self.rfile.read(content_length)
          payload = json.loads(body.decode("utf-8"))
          items = payload.get("cameras", [])
          added = []
          for item in items:
            cname = re.sub(r"[^A-Za-z0-9_\-]", "_", str(item.get("cam_name", "")).strip())
            src_u = str(item.get("src", "")).strip()
            if cname and src_u:
              database.run_put("links", cname, src_u)
              added.append(cname)
          self.send_200({"status": "ok", "added": added, "count": len(added)})
          return

        if parsed_path.path == "/isup_config":
          content_length = int(self.headers.get("Content-Length", 0))
          data = json.loads(self.rfile.read(content_length).decode("utf-8"))
          enable = bool(data.get("enable", True))
          port = int(data.get("port", 7660))
          key = str(data.get("key", ""))
          if enable:
            st = isup_gateway.start_listener("0.0.0.0", port=port, isup_key=key)
          else:
            isup_gateway.stop_listener()
            st = isup_gateway.status()
          self.send_200(st)
          return

        if parsed_path.path == "/ocr_test":
          content_length = int(self.headers.get("Content-Length", 0))
          data = json.loads(self.rfile.read(content_length).decode("utf-8"))
          engine = data.get("engine") or getattr(global_settings, "ocr_engine", "rapidocr")
          plate_mode = bool(data.get("plate_mode", getattr(global_settings, "ocr_plate_mode", False)))
          target_cam = data.get("cam")
          img_b64 = data.get("image_b64")
          img_bgr = None
          if img_b64:
            if "," in img_b64: img_b64 = img_b64.split(",", 1)[1]
            raw_b = base64.b64decode(img_b64)
            img_bgr = cv2.imdecode(np.frombuffer(raw_b, np.uint8), cv2.IMREAD_COLOR)
          elif target_cam and cam is not None and cam.raw_frame.get(target_cam) is not None:
            img_bgr = cam.raw_frame[target_cam].copy()
          elif target_cam:
            link_u = database.run_get("links", target_cam)
            if isinstance(link_u, str) and is_pumatronix_source(link_u):
              pcfg = parse_pumatronix_uri(link_u)
              pclient = PumatronixClient(str(pcfg["ip"]), int(pcfg["port"]), pcfg.get("user"), pcfg.get("password"))
              img_bgr = pclient.snapshot_bgr(quality=95)
          if img_bgr is None:
            self.send_200({"ok": False, "error": "Nenhum frame disponivel para OCR nesta camera ainda."})
            return
          t0 = time.time()
          res = ocr_engine.read_image(img_bgr, engine=engine, plate_mode=plate_mode)
          dt_ms = round((time.time() - t0) * 1000, 1)
          if res and target_cam:
            top = res[0]
            saved_rec = save_plate_gallery_capture(
                cam_name=target_cam,
                full_scene_bgr=img_bgr,
                source_bgr=img_bgr,
                top_res=top,
                crop_offset=None,
                cls_id=2,
                video_ts=0.0,
                dedup_window=0.0,
            )
            entry = saved_rec or {
              "cam_name": target_cam,
              "text": top["text"],
              "plate": top.get("plate"),
              "conf": top["conf"],
              "engine": top["engine"],
              "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "ts_epoch": time.time(),
              "url": "",
            }
            last_ocr_by_cam[target_cam] = entry
            database.run_put("ocr_events", target_cam, entry, replace=False)
          ok_jpg, enc_jpg = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
          preview_b64 = f"data:image/jpeg;base64,{base64.b64encode(enc_jpg.tobytes()).decode('ascii')}" if ok_jpg else ""
          self.send_200({"ok": True, "results": res, "elapsed_ms": dt_ms, "preview": preview_b64})
          return

        if parsed_path.path == "/api/plate_capture_now":
          content_length = int(self.headers.get("Content-Length", 0))
          data = json.loads(self.rfile.read(content_length).decode("utf-8")) if content_length > 0 else {}
          target_cam = data.get("cam") or "all"
          engine = data.get("engine") or getattr(global_settings, "ocr_engine", "rapidocr")
          plate_mode = bool(data.get("plate_mode", False))
          all_links = database.run_get("links", None) or {}
          cam_list = [target_cam] if (target_cam != "all" and target_cam in all_links) else [
            c for c in all_links.keys()
            if cam_runtime_status.get(c, {}).get("state") == "online"
          ]
          captured = []
          scanned = 0
          for cname in cam_list:
            img_bgr = None
            if cam is not None and cam.raw_frame.get(cname) is not None:
              img_bgr = cam.raw_frame[cname].copy()
            else:
              link_u = all_links.get(cname)
              if isinstance(link_u, str) and is_pumatronix_source(link_u):
                try:
                  pcfg = parse_pumatronix_uri(link_u)
                  pclient = PumatronixClient(str(pcfg["ip"]), int(pcfg["port"]), pcfg.get("user"), pcfg.get("password"), timeout=2.5)
                  img_bgr = pclient.snapshot_bgr(quality=92)
                except Exception:
                  pass
            if img_bgr is None:
              continue
            scanned += 1
            res = ocr_engine.read_image(img_bgr, engine=engine, plate_mode=plate_mode)
            for item in res[:2]:
              rec = save_plate_gallery_capture(
                  cam_name=cname,
                  full_scene_bgr=img_bgr,
                  source_bgr=img_bgr,
                  top_res=item,
                  crop_offset=None,
                  cls_id=2,
                  video_ts=0.0,
                  dedup_window=0.0,
              )
              if rec:
                captured.append(rec)
          self.send_200({"ok": True, "scanned": scanned, "captured_count": len(captured), "captured": captured})
          return

        if parsed_path.path == "/api/plate_delete":
          content_length = int(self.headers.get("Content-Length", 0))
          data = json.loads(self.rfile.read(content_length).decode("utf-8")) if content_length > 0 else {}
          uid = str(data.get("uid") or "").strip()
          if not uid:
            self.send_200({"ok": False, "error": "Missing uid"})
            return
          rec = database.run_get("plate_gallery", uid)
          if isinstance(rec, dict):
            for k_url in ("scene_url", "area_url", "clean_url"):
              rel = str(rec.get(k_url) or "").lstrip("/")
              if rel:
                fp = (BASE_DIR / "cameras" / rel).resolve()
                if fp.is_file():
                  fp.unlink(missing_ok=True)
          database.run_delete("plate_gallery", uid)
          self.send_200({"ok": True, "deleted": uid})
          return

        if parsed_path.path == "/api/plate_registry":
          content_length = int(self.headers.get("Content-Length", 0))
          data = json.loads(self.rfile.read(content_length).decode("utf-8")) if content_length > 0 else {}
          raw_plate = str(data.get("plate") or "").strip().upper()
          pkey = normalize_plate_key(raw_plate)
          if len(pkey) < 4:
            self.send_200({"ok": False, "error": "Informe uma placa válida (mín. 4 caracteres)."})
            return
          ck = str(data.get("color_key") or "").strip().lower()
          cmeta = VEHICLE_COLORS.get(ck, VEHICLE_COLORS["indefinida"])
          v_model = str(data.get("vehicle_model") or "").strip().upper()
          v_type = str(data.get("vehicle_type") or "").strip()
          v_year = str(data.get("vehicle_year") or "").strip()
          v_uf = str(data.get("vehicle_uf") or infer_plate_uf(pkey) or "").strip().upper()
          l_status = str(data.get("list_status") or "sem_restricao").strip().lower()
          if l_status not in ("sem_restricao", "blacklist", "whitelist"):
            l_status = "sem_restricao"
          notes = str(data.get("notes") or "").strip()

          entry = {
            "plate": raw_plate,
            "plate_key": pkey,
            "color_key": ck if ck in VEHICLE_COLORS else "indefinida",
            "vehicle_color": cmeta["label"],
            "color_hex": cmeta["hex"],
            "vehicle_model": v_model,
            "vehicle_type": v_type,
            "vehicle_year": v_year,
            "vehicle_uf": v_uf,
            "list_status": l_status,
            "notes": notes,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_epoch": time.time(),
          }
          database.run_put("plate_registry", pkey, entry, replace=True)

          # Propaga para capturas existentes dessa placa em plate_gallery
          updated_captures = 0
          gal_all = database.run_get("plate_gallery", None) or {}
          if isinstance(gal_all, dict):
            for guid, grec in gal_all.items():
              if not isinstance(grec, dict):
                continue
              if normalize_plate_key(grec.get("plate") or grec.get("text")) == pkey:
                gcopy = dict(grec)
                if ck in VEHICLE_COLORS and ck != "indefinida":
                  gcopy["color_key"] = ck
                  gcopy["vehicle_color"] = cmeta["label"]
                  gcopy["color_hex"] = cmeta["hex"]
                if v_model:
                  gcopy["vehicle_model"] = v_model
                if v_type:
                  gcopy["vehicle_type"] = v_type
                if v_year:
                  gcopy["vehicle_year"] = v_year
                if v_uf:
                  gcopy["vehicle_uf"] = v_uf
                gcopy["list_status"] = l_status
                gcopy["notes"] = notes
                gcopy["model_source"] = "registry"
                database.run_put("plate_gallery", guid, gcopy, replace=True)
                updated_captures += 1

          self.send_200({"ok": True, "entry": entry, "updated_captures": updated_captures})
          return

        if parsed_path.path == "/api/plate_registry_delete":
          content_length = int(self.headers.get("Content-Length", 0))
          data = json.loads(self.rfile.read(content_length).decode("utf-8")) if content_length > 0 else {}
          pkey = normalize_plate_key(data.get("plate_key") or data.get("plate") or "")
          if not pkey:
            self.send_200({"ok": False, "error": "Missing plate_key"})
            return
          database.run_delete("plate_registry", pkey)
          gal_all = database.run_get("plate_gallery", None) or {}
          if isinstance(gal_all, dict):
            for guid, grec in gal_all.items():
              if isinstance(grec, dict) and normalize_plate_key(grec.get("plate") or grec.get("text")) == pkey:
                gcopy = dict(grec)
                gcopy["list_status"] = "sem_restricao"
                database.run_put("plate_gallery", guid, gcopy, replace=True)
          self.send_200({"ok": True, "deleted": pkey})
          return

        if parsed_path.path == "/api/vehicle_update_record":
          content_length = int(self.headers.get("Content-Length", 0))
          data = json.loads(self.rfile.read(content_length).decode("utf-8")) if content_length > 0 else {}
          uid = str(data.get("uid") or "").strip()
          rec = database.run_get("plate_gallery", uid) if uid else None
          if not isinstance(rec, dict):
            self.send_200({"ok": False, "error": "Registro não encontrado"})
            return
          rc = dict(rec)
          if "plate" in data and str(data["plate"]).strip():
            new_p = str(data["plate"]).strip().upper()
            rc["plate"] = new_p
            rc["text"] = new_p
          ck = str(data.get("color_key") or rc.get("color_key") or "indefinida").strip().lower()
          if ck in VEHICLE_COLORS:
            rc["color_key"] = ck
            rc["vehicle_color"] = VEHICLE_COLORS[ck]["label"]
            rc["color_hex"] = VEHICLE_COLORS[ck]["hex"]
          if "vehicle_model" in data:
            rc["vehicle_model"] = str(data["vehicle_model"] or "").strip().upper()
            rc["model_source"] = "manual"
          if "vehicle_type" in data:
            rc["vehicle_type"] = str(data["vehicle_type"] or "").strip()
          if "vehicle_year" in data:
            rc["vehicle_year"] = str(data["vehicle_year"] or "").strip()
          if "vehicle_uf" in data:
            rc["vehicle_uf"] = str(data["vehicle_uf"] or "").strip().upper()
          if "list_status" in data and data["list_status"] in ("sem_restricao", "blacklist", "whitelist"):
            rc["list_status"] = str(data["list_status"])
          if "notes" in data:
            rc["notes"] = str(data["notes"] or "").strip()

          database.run_put("plate_gallery", uid, rc, replace=True)

          # Opcional: persistir também no cadastro permanente da placa
          if data.get("save_registry", True) and rc.get("plate"):
            pkey = normalize_plate_key(rc.get("plate"))
            if len(pkey) >= 5:
              reg_entry = {
                "plate": rc["plate"],
                "plate_key": pkey,
                "color_key": rc.get("color_key", "indefinida"),
                "vehicle_color": rc.get("vehicle_color", "Não Identificada"),
                "color_hex": rc.get("color_hex", "#64748b"),
                "vehicle_model": rc.get("vehicle_model", ""),
                "vehicle_type": rc.get("vehicle_type", ""),
                "vehicle_year": rc.get("vehicle_year", ""),
                "vehicle_uf": rc.get("vehicle_uf", ""),
                "list_status": rc.get("list_status", "sem_restricao"),
                "notes": rc.get("notes", ""),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "updated_epoch": time.time(),
              }
              database.run_put("plate_registry", pkey, reg_entry, replace=True)

          self.send_200({"ok": True, "record": rc})
          return

        if self.path.startswith("/edit_settings"):
          content_length = int(self.headers.get('Content-Length', 0))
          body = self.rfile.read(content_length)
          data = json.loads(body.decode('utf-8'))
          # keep userid and key if "True"
          if data.get("userID") == True: data["userID"] = global_settings.userID
          if data.get("key") == True: data["key"] = global_settings.key
          add_to_queue(db.run_put, database, "global_settings", "all", GlobalSettings(**data))
          add_to_queue(set_settings, GlobalSettings(**data))
          self.send_200([])
          return

        if self.path.startswith("/analyse-footage"):
          params = parse_qs(parsed_path.query)
          filename = params.get("filename", [None])[0]
          chunk = int(params.get("chunk", [0])[0])
          total = int(params.get("total", [1])[0])
          if not filename:
            self.send_error(400, "Missing filename")
            return
          filename = os.path.basename(filename)
          upload_dir = BASE_DIR / "cameras"
          upload_dir.mkdir(exist_ok=True)
          length = int(self.headers.get("Content-Length", 0))
          if length <= 0:
            self.send_error(411, "Content-Length required")
            return
          final_path = upload_dir / filename
          temp_path = upload_dir / f"{filename}.part"
          with open(temp_path, "ab") as f:
            remaining = length
            while remaining > 0:
              data = self.rfile.read(min(1024 * 1024, remaining))
              if not data: break
              f.write(data)
              remaining -= len(data)
          if chunk == total - 1: temp_path.rename(final_path)
          self.send_200([])

        if parsed_path.path == "/event_thumbs":
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length)

            try:
                data = json.loads(raw_body)
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
                return

            cam_name     = data.get("cam")
            selected_dir = data.get("folder")
            name_contains = data.get("name_contains")
            image_text   = data.get("image_text")
            similar_img  = data.get("similar_img")
            start = data.get("start")
            count = data.get("count")
            is_face = data.get("is_face") or False
            if is_face and not global_settings.use_face: return
            if start is None: start, count = 0, 100
            uploaded_image = data.get("uploaded_image")
            if uploaded_image:
              if ',' in uploaded_image: uploaded_image = uploaded_image.split(',')[1]
              uploaded_image = base64.b64decode(uploaded_image)

            if cam_name:
              camera_dirs = [BASE_DIR / "cameras" / cam_name]
            else:
              camera_dirs = [d for d in (BASE_DIR / "cameras").iterdir() if d.is_dir()]

            if selected_dir:
              selected_dirs = [selected_dir]
            else:
              selected_dirs = list({
                subdir.name 
                for camera_dir in camera_dirs
                if (camera_dir / "streams").is_dir()
                for subdir in (camera_dir / "streams").iterdir() 
                if subdir.is_dir()
              })
            selected_dirs.append("video")

            if image_text and global_settings.use_clip: add_to_queue(object_finder._load_all_embeddings)
            if (uploaded_image or similar_img) and (global_settings.use_clip or global_settings.use_face): add_to_queue(object_finder._load_all_embeddings, is_face)
            
            if uploaded_image and (global_settings.use_clip or is_face):
              results = add_to_queue(run_clip, object_finder, uploaded_image, start+count, cam_name, selected_dir, is_face)
              self.send_results(results, start, count)
              return
            
            if similar_img and (global_settings.use_clip or is_face):
              results = add_to_queue(run_clip, object_finder, similar_img, start+count, cam_name, selected_dir, is_face) # todo one with above
              self.send_results(results, start, count)
              return

            if image_text and global_settings.use_clip:
              results = add_to_queue(run_search, object_finder, image_text, start+count, cam_name, selected_dir)
              self.send_results(results, start, count)
              return

            # Also allow OCR text/plate search when image_text is entered without CLIP
            if image_text and not global_settings.use_clip:
              q_up = image_text.strip().upper()
              ocr_all = database.run_get("ocr_events", cam_name) if cam_name else database.run_get("ocr_events", None)
              matched_imgs = []
              if isinstance(ocr_all, dict):
                entries = []
                for _, v in ocr_all.items():
                  if isinstance(v, dict) and "text" in v: entries.append(v)
                  elif isinstance(v, dict):
                    for _, sv in v.items():
                      if isinstance(sv, dict) and "text" in sv: entries.append(sv)
                for ent in sorted(entries, key=lambda x: x.get("ts_epoch", 0), reverse=True):
                  if q_up in str(ent.get("text", "")).upper() or q_up in str(ent.get("plate") or "").upper():
                    if ent.get("url"):
                      matched_imgs.append({
                        "url": ent["url"],
                        "timestamp": int(float(ent.get("video_ts", 0))),
                        "filename": os.path.basename(ent["url"]),
                        "cam_name": ent.get("cam_name", ""),
                        "folder": ent.get("folder", datetime.now().strftime("%Y-%m-%d")),
                        "ocr": ent.get("text", ""),
                      })
              self.send_200({"images": matched_imgs[start:start+count], "count": len(matched_imgs)})
              return

            image_data = []
            for camera_dir in camera_dirs:
              for selected_dir in selected_dirs:
                event_image_path = camera_dir / "event_images" / selected_dir
                if not event_image_path.exists(): continue
                event_images = sorted(
                  event_image_path.glob("*.jpg"),
                  key=lambda p: int(p.stem.split('_')[0]),
                  reverse=True
                )
                for img in event_images:
                  if name_contains and name_contains not in img.name: continue
                  ts = int(img.stem.split('_')[0])
                  rel_cam = img.relative_to(BASE_DIR / "cameras").as_posix()
                  image_url = f"/{rel_cam}"
                  image_data.append({
                    "url": image_url,
                    "timestamp": ts,
                    "filename": img.name,
                    "cam_name": camera_dir.name,
                    "folder": selected_dir,
                    "ocr": ocr_by_image.get(str(img.resolve()), ""),
                  })

            image_data.sort(key=image_sort_key, reverse=True)
            if start is not None and count is not None: image_data = image_data[start:start+count]

            response_data = {
              "images": image_data,
              "count": len(image_data),
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            return

def image_sort_key(item):
  try: return datetime.strptime(item["folder"], "%Y-%m-%d").timestamp() + item["timestamp"]
  except ValueError: return -1

def schedule_daily_restart(cam, restart_time):
    while True:
      now = datetime.now().time()
      target = time_obj(restart_time[0], restart_time[1])
      if now >= target:
        delta = (24 * 3600) - ((now.hour * 3600 + now.minute * 60 + now.second) - (target.hour * 3600 + target.minute * 60))
      else:
        delta = ((target.hour * 3600 + target.minute * 60) - 
          (now.hour * 3600 + now.minute * 60 + now.second))
      time.sleep(delta)
      cams = database.run_get("links", None)
      for cam_name in cams.keys():
        cam.start_time[cam_name] = None
        cam.hls_proc[cam_name], cam.proc[cam_name] = cam._open_ffmpeg(cam_name)
        cam.current_stream_dir_raw[cam_name] = cam._get_new_stream_dir(cam_name)



def get_lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def get_executable_args(): return ([sys.argv[0]], sys.argv[1:]) if getattr(sys, "frozen", False) else ([sys.executable, sys.argv[0]], sys.argv[1:])

def event_img_info(image): return {"ts": int(float(image.split('_')[0])), "object_id":int(image.split('_')[1]), "class_id":int(image.split('_')[2])}

def upload_to_r2(file_path: Path, signed_url: str, max_retries: int = 0) -> bool:
    try:
      url_parts = urllib.parse.urlparse(signed_url)
      if url_parts.scheme == 'https':
        conn = http.client.HTTPSConnection(url_parts.netloc)
      else:
        conn = http.client.HTTPConnection(url_parts.netloc)
      
      with file_path.open('rb') as f:
        file_size = file_path.stat().st_size
        headers = {'Content-Type': 'application/octet-stream', "Content-Length": str(file_size)}
        conn.request("PUT", url_parts.path + "?" + url_parts.query, body=f, headers=headers)
        response = conn.getresponse()
        if 200 <= response.status < 300: return True
        return False
    except Exception as e:
      print(f"Error uploading to R2: {e}")
      return False

import queue
task_queue = queue.Queue()
def add_to_queue(fn, *args):
  result_queue = queue.Queue(maxsize=1)
  task_queue.put((fn, args, result_queue))
  return result_queue.get()

def add_to_queue_timeout(fn, *args, timeout=2.5):
  if not _main_loop_ready:
    return None
  result_queue = queue.Queue(maxsize=1)
  task_queue.put((fn, args, result_queue))
  try:
    return result_queue.get(timeout=timeout)
  except queue.Empty:
    return None

def process_queue():
  for _ in range(4):
    try:
      fn, args, result_queue = task_queue.get_nowait()
    except queue.Empty:
      return
    try:
      result = fn(*args)
    except Exception as exc:
      print(f"[TaskQueue] warning: {exc}")
      result = None
    try:
      result_queue.put_nowait(result)
    except Exception:
      pass

def run_clip_vehicle_classify(veh_roi_bgr):
  if not _main_loop_ready or not getattr(global_settings, "use_clip", False):
    return None
  return add_to_queue_timeout(vehicle_analyzer.classify_with_clip_mainthread, object_finder, veh_roi_bgr, timeout=2.5)

def process_latest_face(img):
  if global_settings.use_face and str(object_queue[0]).endswith("_0.jpg"):
    face_img = object_finder.img_to_face(img)
    
    if face_img is not None:
      face_out_path = Path(str(object_queue[0]).replace("\\objects\\", "\\faces\\").replace("/objects/", "/faces/"))
      face_out_path.parent.mkdir(parents=True, exist_ok=True)
      cv2.imwrite(str(face_out_path), face_img)
      date = object_queue[0].parent.name
      pkl_path = object_queue[0].parent.parent.parent / "faces" / date /  "embeddings.pkl"
      face_emb = object_finder.adaface(Tensor(face_img).contiguous()).numpy()
      data = pickle.load(open(pkl_path, "rb")) if pkl_path.exists() else {}
      if "embeddings" not in data: data["embeddings"] = {}
      data["embeddings"][str(object_queue[0])] = face_emb
      pkl_path.parent.mkdir(parents=True, exist_ok=True)
      pickle.dump(data, open(pkl_path, "wb"))  

_recent_plate_dedup = {}
_plate_save_lock = threading.Lock()

def save_plate_gallery_capture(cam_name, full_scene_bgr, source_bgr, top_res, crop_offset=None, cls_id=2, video_ts=0.0, dedup_window=5.0, custom_time_str=None, custom_epoch=None):
  if source_bgr is None or source_bgr.size == 0 or not top_res:
    return None
  scene_img = full_scene_bgr if (full_scene_bgr is not None and full_scene_bgr.size > 0) else source_bgr
  plate_code = top_res.get("plate")
  display_text = str(plate_code or top_res.get("text") or "").strip()
  if not display_text:
    return None
  conf_val = float(top_res.get("conf", 0.0))
  now_epoch = float(custom_epoch if custom_epoch is not None else time.time())
  dedup_key = f"{cam_name}::{display_text.upper()}"

  with _plate_save_lock:
    prev = _recent_plate_dedup.get(dedup_key)
    if dedup_window > 0 and prev and (now_epoch - prev[0]) < dedup_window and conf_val <= prev[1]:
      return None
    _recent_plate_dedup[dedup_key] = (now_epoch, conf_val)

  sh, sw = source_bgr.shape[:2]
  raw_box = top_res.get("box") or [0, 0, sw, sh]
  bx1, by1, bx2, by2 = [int(v) for v in raw_box[:4]]
  bx1 = max(0, min(sw - 2, bx1))
  by1 = max(0, min(sh - 2, by1))
  bx2 = max(bx1 + 2, min(sw, bx2))
  by2 = max(by1 + 2, min(sh, by2))

  bw, bh = bx2 - bx1, by2 - by1
  pad_x = max(8, int(bw * 0.14))
  pad_y = max(6, int(bh * 0.25))
  ax1 = max(0, bx1 - pad_x)
  ay1 = max(0, by1 - pad_y)
  ax2 = min(sw, bx2 + pad_x)
  ay2 = min(sh, by2 + pad_y)
  area_crop = source_bgr[ay1:ay2, ax1:ax2].copy()
  if area_crop.size == 0:
    area_crop = source_bgr.copy()
  ah, aw = area_crop.shape[:2]
  if ah > 0 and ah < 110:
    up_scale = min(4.0, 110.0 / float(ah))
    area_crop = cv2.resize(area_crop, (max(1, int(aw * up_scale)), max(1, int(ah * up_scale))), interpolation=cv2.INTER_CUBIC)

  annotated = scene_img.copy()
  fh, fw = annotated.shape[:2]
  if crop_offset is not None and full_scene_bgr is not None:
    vx1, vy1, vx2, vy2 = [int(v) for v in crop_offset]
    cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (235, 190, 40), 2)
    px1 = max(0, min(fw - 2, vx1 + ax1))
    py1 = max(0, min(fh - 2, vy1 + ay1))
    px2 = max(px1 + 2, min(fw, vx1 + ax2))
    py2 = max(py1 + 2, min(fh, vy1 + ay2))
  else:
    px1 = max(0, min(fw - 2, ax1))
    py1 = max(0, min(fh - 2, ay1))
    px2 = max(px1 + 2, min(fw, ax2))
    py2 = max(py1 + 2, min(fh, ay2))

  pkey = normalize_plate_key(plate_code or display_text)
  reg_entry = database.run_get("plate_registry", pkey) if len(pkey) >= 4 else None
  if not isinstance(reg_entry, dict):
    reg_entry = None

  v_attrs = vehicle_analyzer.analyze(
      clean_bgr=scene_img,
      cls_id=cls_id,
      plate_code=plate_code,
      raw_ocr_text=str(top_res.get("raw_text") or display_text),
      plate_box=[px1, py1, px2, py2],
      crop_offset=crop_offset,
      registry_entry=reg_entry,
      clip_runner=run_clip_vehicle_classify,
  )

  is_black = v_attrs.get("list_status") == "blacklist"
  box_col = (40, 40, 235) if is_black else ((80, 230, 0) if plate_code else (0, 195, 255))
  cv2.rectangle(annotated, (px1, py1), (px2, py2), box_col, 3)
  clen = max(8, min(22, (px2 - px1) // 4, (py2 - py1) // 2))
  for cx_p, cy_p, dx_s, dy_s in ((px1, py1, 1, 1), (px2, py1, -1, 1), (px1, py2, 1, -1), (px2, py2, -1, -1)):
    cv2.line(annotated, (cx_p, cy_p), (cx_p + dx_s * clen, cy_p), (255, 255, 255), 3)
    cv2.line(annotated, (cx_p, cy_p), (cx_p, cy_p + dy_s * clen), (255, 255, 255), 3)

  prefix_lbl = "ALERTA BLACKLIST" if is_black else ("PLACA" if plate_code else "OCR")
  tag_txt = f"{prefix_lbl}: {display_text} ({int(round(conf_val * 100))}%) | {v_attrs['vehicle_color']}"
  (tw, th), _ = cv2.getTextSize(tag_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
  ty1 = max(4, py1 - th - 14)
  ty2 = ty1 + th + 10
  tx1 = max(4, min(fw - tw - 16, px1))
  tx2 = min(fw - 4, tx1 + tw + 12)
  cv2.rectangle(annotated, (tx1, ty1), (tx2, ty2), (24, 24, 28), -1)
  cv2.rectangle(annotated, (tx1, ty1), (tx2, ty2), box_col, 2)
  cv2.putText(annotated, tag_txt, (tx1 + 6, ty2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

  dt_obj = datetime.fromtimestamp(now_epoch)
  date_folder = dt_obj.strftime("%Y-%m-%d")
  time_str = custom_time_str or dt_obj.strftime("%Y-%m-%d %H:%M:%S")
  plates_dir = (BASE_DIR / "cameras" / cam_name / "plates" / date_folder).resolve()
  plates_dir.mkdir(parents=True, exist_ok=True)

  safe_txt = re.sub(r"[^A-Za-z0-9]", "", display_text)[:12] or "OCR"
  uid = f"{int(now_epoch * 1000)}_{safe_txt}_{uuid.uuid4().hex[:5]}"
  scene_path = plates_dir / f"{uid}_scene.jpg"
  area_path = plates_dir / f"{uid}_area.jpg"
  clean_path = plates_dir / f"{uid}_clean.jpg"

  cv2.imwrite(str(scene_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
  cv2.imwrite(str(area_path), area_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
  cv2.imwrite(str(clean_path), scene_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

  cams_root = (BASE_DIR / "cameras").resolve()
  scene_url = "/" + scene_path.relative_to(cams_root).as_posix()
  area_url = "/" + area_path.relative_to(cams_root).as_posix()
  clean_url = "/" + clean_path.relative_to(cams_root).as_posix()

  record = {
    "uid": uid,
    "cam_name": cam_name,
    "text": display_text,
    "raw_text": str(top_res.get("raw_text") or display_text),
    "plate": plate_code,
    "conf": round(conf_val, 3),
    "engine": str(top_res.get("engine") or "rapidocr"),
    "class_id": int(cls_id),
    "video_ts": float(video_ts or 0.0),
    "folder": date_folder,
    "time": time_str,
    "ts_epoch": now_epoch,
    "plate_box": [px1, py1, px2, py2],
    "crop_box": [ax1, ay1, ax2, ay2],
    "scene_res": f"{fw}x{fh}",
    "area_res": f"{area_crop.shape[1]}x{area_crop.shape[0]}",
    "scene_url": scene_url,
    "area_url": area_url,
    "clean_url": clean_url,
    "url": scene_url,
    "color_key": v_attrs["color_key"],
    "vehicle_color": v_attrs["vehicle_color"],
    "color_hex": v_attrs["color_hex"],
    "color_conf": v_attrs["color_conf"],
    "vehicle_model": v_attrs["vehicle_model"],
    "vehicle_type": v_attrs["vehicle_type"],
    "vehicle_year": v_attrs["vehicle_year"],
    "vehicle_uf": v_attrs["vehicle_uf"],
    "list_status": v_attrs["list_status"],
    "notes": v_attrs["notes"],
    "model_source": "registry" if reg_entry else ("clip" if (_main_loop_ready and getattr(global_settings, "use_clip", False)) else "onnx"),
  }
  database.run_put("plate_gallery", uid, record, replace=True)
  return record

def backfill_existing_plate_gallery():
  try:
    cams_root = (BASE_DIR / "cameras").resolve()
    existing = database.run_get("plate_gallery", None) or {}
    reg_all = database.run_get("plate_registry", None) or {}

    # Passada 1 imediata (mais recentes primeiro): Cor (HSV/LAB) + Modelo Neural ONNX + Emblema OCR + UF DENATRAN
    if isinstance(existing, dict):
      ordered = sorted(
          [(u, r) for u, r in existing.items() if isinstance(r, dict)],
          key=lambda kv: float(kv[1].get("ts_epoch", 0)),
          reverse=True,
      )
      for uid, rec in ordered:
        needs_upgrade = (
            not rec.get("color_key")
            or rec.get("color_key") == "indefinida"
            or rec.get("model_source") in (None, "", "geometry")
            or (not rec.get("plate") and rec.get("vehicle_uf"))
        )
        if not needs_upgrade:
          continue
        rel_clean = str(rec.get("clean_url") or rec.get("scene_url") or "").lstrip("/")
        if not rel_clean:
          continue
        img_fp = (cams_root / rel_clean).resolve()
        if not img_fp.is_file():
          continue
        clean_bgr = cv2.imread(str(img_fp))
        if clean_bgr is None:
          continue
        pkey = normalize_plate_key(rec.get("plate") or "")
        reg_entry = reg_all.get(pkey) if (pkey and isinstance(reg_all, dict)) else None
        attrs = vehicle_analyzer.analyze(
            clean_bgr=clean_bgr,
            cls_id=int(rec.get("class_id", 2)),
            plate_code=rec.get("plate"),
            raw_ocr_text=rec.get("raw_text") or rec.get("text"),
            plate_box=rec.get("plate_box"),
            crop_offset=None,
            registry_entry=reg_entry,
            clip_runner=None,
        )
        rc = dict(rec)
        rc.update(attrs)
        rc["model_source"] = "registry" if reg_entry else "onnx"
        database.run_put("plate_gallery", uid, rc, replace=True)

    # Capturas novas de objects/ ainda não processadas
    existing = database.run_get("plate_gallery", None) or {}
    known_sources = {str(v.get("backfill_src")) for v in existing.values() if isinstance(v, dict) and v.get("backfill_src")}
    if cams_root.exists():
      for obj_jpg in sorted(cams_root.glob("*/objects/*/*.jpg")):
        src_key = obj_jpg.as_posix()
        if src_key in known_sources:
          continue
        img = cv2.imread(str(obj_jpg))
        if img is None:
          continue
        res = ocr_engine.read_image(img, engine="rapidocr", plate_mode=False)
        if not res:
          continue
        top = res[0]
        cname = obj_jpg.parts[obj_jpg.parts.index("cameras") + 1]
        st_mtime = obj_jpg.stat().st_mtime
        rec = save_plate_gallery_capture(
            cam_name=cname,
            full_scene_bgr=img,
            source_bgr=img,
            top_res=top,
            crop_offset=None,
            cls_id=2,
            video_ts=0.0,
            dedup_window=0.0,
            custom_epoch=st_mtime,
        )
        if rec:
          rec["backfill_src"] = src_key
          database.run_put("plate_gallery", rec["uid"], rec, replace=True)

    # Passada 2 (após loop principal ativo): refina Marca/Modelo via OpenCLIP Zero-Shot
    for _ in range(90):
      if _main_loop_ready:
        break
      time.sleep(0.5)
    if _main_loop_ready and getattr(global_settings, "use_clip", False):
      existing2 = database.run_get("plate_gallery", None) or {}
      reg_all2 = database.run_get("plate_registry", None) or {}
      if isinstance(existing2, dict):
        items_sorted = sorted(
            [(u, r) for u, r in existing2.items() if isinstance(r, dict)],
            key=lambda kv: float(kv[1].get("ts_epoch", 0)),
            reverse=True,
        )
        for uid, rec in items_sorted:
          if rec.get("model_source") in ("clip", "registry", "manual"):
            continue
          pkey = normalize_plate_key(rec.get("plate") or rec.get("text"))
          if pkey and isinstance(reg_all2, dict) and pkey in reg_all2:
            continue
          rel_clean = str(rec.get("clean_url") or rec.get("scene_url") or "").lstrip("/")
          if not rel_clean:
            continue
          img_fp = (cams_root / rel_clean).resolve()
          if not img_fp.is_file():
            continue
          clean_bgr = cv2.imread(str(img_fp))
          if clean_bgr is None:
            continue
          attrs = vehicle_analyzer.analyze(
              clean_bgr=clean_bgr,
              cls_id=int(rec.get("class_id", 2)),
              plate_code=rec.get("plate") or rec.get("text"),
              plate_box=rec.get("plate_box"),
              crop_offset=None,
              registry_entry=None,
              clip_runner=run_clip_vehicle_classify,
          )
          rc = dict(rec)
          rc["vehicle_model"] = attrs["vehicle_model"]
          rc["vehicle_type"] = attrs["vehicle_type"]
          if rc.get("color_key") in (None, "", "indefinida"):
            rc["color_key"] = attrs["color_key"]
            rc["vehicle_color"] = attrs["vehicle_color"]
            rc["color_hex"] = attrs["color_hex"]
          rc["model_source"] = "clip"
          database.run_put("plate_gallery", uid, rc, replace=True)
          time.sleep(0.03)
  except Exception as exc:
    print(f"[Plate Backfill] warning: {exc}")

def ocr_worker_loop():
  while True:
    try:
      item = ocr_queue.get(timeout=1.0)
    except _std_queue.Empty:
      continue
    try:
      if len(item) == 7:
        cam_n, obj_path_str, crop_bgr, full_bgr, crop_off, cls_id, video_ts = item
      else:
        cam_n, obj_path_str, crop_bgr, cls_id, video_ts = item
        full_bgr, crop_off = crop_bgr, None
      if not getattr(global_settings, "use_ocr", True):
        continue
      target_bgr = crop_bgr
      effective_full = full_bgr
      effective_off = crop_off
      # Optional high-res Pumatronix snapshot on vehicle detection (car=2, moto=3, bus=5, truck=7)
      if getattr(global_settings, "ocr_pumatronix_hi_res", False) and cls_id in (2, 3, 5, 7):
        link_u = database.run_get("links", cam_n)
        if isinstance(link_u, str) and is_pumatronix_source(link_u):
          try:
            pcfg = parse_pumatronix_uri(link_u)
            pclient = PumatronixClient(str(pcfg["ip"]), int(pcfg["port"]), pcfg.get("user"), pcfg.get("password"), timeout=2.5)
            snap_bgr = pclient.snapshot_bgr(quality=95)
            if snap_bgr is not None:
              target_bgr = snap_bgr
              effective_full = snap_bgr
              effective_off = None
          except Exception:
            pass
      results = ocr_engine.read_image(
          target_bgr,
          engine=getattr(global_settings, "ocr_engine", "rapidocr"),
          plate_mode=getattr(global_settings, "ocr_plate_mode", False),
      )
      if results:
        top = results[0]
        obj_p = Path(obj_path_str).resolve()
        saved_rec = save_plate_gallery_capture(
            cam_name=cam_n,
            full_scene_bgr=effective_full,
            source_bgr=target_bgr,
            top_res=top,
            crop_offset=effective_off,
            cls_id=cls_id,
            video_ts=video_ts,
            dedup_window=5.0,
        )
        rel_url = saved_rec["scene_url"] if saved_rec else ""
        if not rel_url:
          try:
            rel_url = "/" + obj_p.relative_to((BASE_DIR / "cameras").resolve()).as_posix()
          except Exception:
            pass
        entry = saved_rec or {
          "cam_name": cam_n,
          "text": top["text"],
          "plate": top.get("plate"),
          "conf": top["conf"],
          "engine": top["engine"],
          "class_id": cls_id,
          "video_ts": video_ts,
          "folder": obj_p.parent.name,
          "url": rel_url,
          "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          "ts_epoch": time.time(),
        }
        last_ocr_by_cam[cam_n] = entry
        ocr_by_image[str(obj_p)] = top["text"]
        database.run_put("ocr_events", cam_n, entry, replace=False)
    except Exception as exc:
      print(f"[OCR Worker] warning: {exc}")

def set_settings(x): # todo, save to db, do logic in GlobalSettings class, sanitize inputs so yolo doesn't crash
  global global_settings
  global model
  global yolo_res
  global yolo_jit_cache
  global qwen
  global qwen_prompt
  if x.use_clip:
    object_finder.init_clip()
  else:
    object_finder.turn_off_clip()

  if x.use_face:
    object_finder.init_face()
  else:
    object_finder.turn_off_face()

  if x.model_size != global_settings.model_size or x.model_res != global_settings.model_res:
    yolo_jit_cache = {}
    model = YOLOv9(x.model_size, x.model_res)

  if x.key == None: x.userID = None

  if getattr(x, "isup_listen", True):
    isup_gateway.start_listener("0.0.0.0", int(getattr(x, "isup_port", 7660) or 7660), str(getattr(x, "isup_key", "") or ""))
  else:
    isup_gateway.stop_listener()

  # todo change size mid run doesn't work
  if (x.use_qwen and qwen is None) or (x.use_qwen and global_settings.qwen_size != x.qwen_size):
    try:
      qwen = None
      qwen = Qwen3VL(size=f"{x.qwen_size}B", res=(544, 960)) # h, w. they need to be multiples of 32
      qwen_prompt = "What has been detected on my CCTV camera? Write in one short sentence"
      print("prewarming Qwen")
      qwen.prewarm()
      print("DONE")
    except Exception as exc:
      print(f"[Qwen3VL] Falha ao carregar GGUF ({exc}); mantendo servidor ativo sem Qwen.")
      qwen = None
      x.use_qwen = False
      database.run_put("global_settings", "all", x)
  global_settings = x

def clip_latest_img(img):
  if global_settings.use_clip:
    img = object_finder.preprocess(img)
    try:
      data = pickle.load(open(object_queue[0].parent / 'embeddings.pkl', 'rb')) if os.path.exists(object_queue[0].parent / 'embeddings.pkl') else {}
    except Exception: data = {}
    if "embeddings" not in data: data["embeddings"] = {}
    emb = jit_infer(object_finder.model.precompute_embedding, Tensor(img).unsqueeze(0), jit_cache).numpy()
    data["embeddings"][str(object_queue[0])] = emb
    with open(object_queue[0].parent / 'embeddings.pkl', "wb") as f: pickle.dump(data, f)
  
    if global_settings.use_notifs():
      cam_name = object_queue[0].parts[object_queue[0].parts.index("cameras")+1:object_queue[0].parts.index("objects")][0]
      alerts = database.run_get("alerts", cam_name) # todo, get cam_name from file path!
      for k, v in alerts.items():
        if time.time() - v.last_det < 60 or not v.is_active(): continue
        if v.desc is None: continue
        if not hasattr(v, "desc_emb") or v.desc_emb is None:
          v.desc_emb = run_encode_text(object_finder, v.desc)
          database.run_put("alerts", cam_name, v, id=k)

        similarity = (v.desc_emb @ emb.T).item()
        print("sim =",similarity,v.desc,object_queue[0])
        if similarity > v.threshold:
          send_notif(session_token=global_settings.userID, text=f"Event Detected ({cam_name}: {v.desc})", body_text=None, host=global_settings.server_url)
          alerts[k].last_det = time.time()
          database.run_put("alerts", cam_name, alerts[k], k)
          if global_settings.clearcam_user():
            seen_time = event_img_info(str(object_queue[0]).split("/")[-1].split(".jpg")[0])["ts"]
            threading.Thread(target=export_and_upload, kwargs={"cam_name": cam_name, "thumbnail": object_queue[0], "userID": global_settings.userID, "key": global_settings.key, "start": seen_time, "length": 20}, daemon=True).start()
          break

cams = dict()
active_subprocesses = []
import socket
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = (_sys.platform != "win32")
    def handle_error(self, request, client_address):
      exc = _sys.exc_info()[1]
      if isinstance(exc, (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
        return
      super().handle_error(request, client_address)
    def __init__(self, server_address, RequestHandlerClass):
      ThreadingMixIn.__init__(self)
      HTTPServer.__init__(self, server_address, RequestHandlerClass)
      self.cleanup_stop_event = threading.Event()
      self.cleanup_thread = None
      max_gb = database.run_get("max_storage", None)
      if max_gb == {}:
        database.run_put("max_storage", "all", 256)
        max_gb = database.run_get("max_storage", None)
      self.max_gb = max_gb["all"]
      self.object_finder_stop_event = threading.Event()
      self.object_finder_thread = None
      self._setup_cleanup_and_clip_thread()

    def _setup_cleanup_and_clip_thread(self):
      if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
        self.cleanup_stop_event.clear()
        self.cleanup_thread = threading.Thread(target=self._cleanup_task, daemon=True, name="StorageCleanup")
        self.cleanup_thread.start()

    def _cleanup_task(self):
      while not self.cleanup_stop_event.is_set():
          try:
              self._check_and_cleanup_storage()
          except Exception as e:
              print(f"Cleanup error: {e}")
          self.cleanup_stop_event.wait(timeout=600)

    def _check_and_cleanup_storage(self):
      total_size = sum(f.stat().st_size for f in (BASE_DIR / "cameras").glob('**/*') if f.is_file())
      size_gb = total_size / (1000 ** 3)
      free_gb = shutil.disk_usage(BASE_DIR / "cameras").free / (1000 ** 3)
      if size_gb > self.max_gb or free_gb < 5: self._cleanup_oldest_files() # todo unhardcode

    def _cleanup_oldest_files(self):
      camera_dirs = []
      for cam_dir in (BASE_DIR / "cameras").iterdir():
        if cam_dir.is_dir():
          dir_size = sum(f.stat().st_size for f in cam_dir.glob('**/*') if f.is_file())
          camera_dirs.append((cam_dir, dir_size))
      
      if not camera_dirs: return
          
      largest_cam_raw = max(camera_dirs, key=lambda x: x[1])[0]
      for largest_cam in [largest_cam_raw]:
        streams_dir = largest_cam / "streams"
        if not streams_dir.exists():
          shutil.rmtree(largest_cam)
          continue
            
        recordings = []
        for rec_dir in streams_dir.iterdir():
          if rec_dir.is_dir(): recordings.append((rec_dir, rec_dir.stat().st_ctime))
        if not recordings:
          shutil.rmtree(largest_cam)
          print(f"Deleted camera folder with empty streams: {largest_cam}")
          continue
        recordings.sort(key=lambda x: x[1])
        oldest_recording = recordings[0][0]
        shutil.rmtree(oldest_recording)
        event_images_dir = largest_cam.with_name(largest_cam.name) / Path("event_images") / Path(oldest_recording.name)
        object_images_dir = largest_cam.with_name(largest_cam.name) / Path("objects") / Path(oldest_recording.name)
        face_images_dir = largest_cam.with_name(largest_cam.name) / Path("faces") / Path(oldest_recording.name)
        plate_images_dir = largest_cam.with_name(largest_cam.name) / Path("plates") / Path(oldest_recording.name)
        #dets_dir = largest_cam.with_name(largest_cam.name) / Path("dets") / Path(oldest_recording.name)
        if event_images_dir.exists(): shutil.rmtree(event_images_dir)
        if object_images_dir.exists(): shutil.rmtree(object_images_dir)
        if face_images_dir.exists(): shutil.rmtree(face_images_dir)
        if plate_images_dir.exists(): shutil.rmtree(plate_images_dir)
        #if dets_dir.exists(): shutil.rmtree(dets_dir)
        print(f"Deleted oldest recording: {oldest_recording}")

class GlobalSettings:
  def __init__(self, use_clip=False, use_face=False ,model_size="t", model_res=960, userID=None, key=None, use_qwen=False, qwen_size=2,
               qwen_prompt="What has been detected on my CCTV camera? Write in one short sentence", server_url="https://clearcam.org",
               use_ocr=True, ocr_engine="rapidocr", ocr_plate_mode=False, ocr_interval=2.0, ocr_pumatronix_hi_res=False,
               isup_listen=True, isup_port=7660, isup_key=""):
    self.use_clip = use_clip
    self.use_face = use_face
    self.model_size = model_size
    self.model_res = model_res
    self.userID = userID
    self.key= key
    self.use_qwen = use_qwen
    self.qwen_size = qwen_size
    self.qwen_prompt = qwen_prompt
    self.server_url = server_url
    self.use_ocr = use_ocr
    self.ocr_engine = ocr_engine
    self.ocr_plate_mode = ocr_plate_mode
    self.ocr_interval = float(ocr_interval or 2.0)
    self.ocr_pumatronix_hi_res = ocr_pumatronix_hi_res
    self.isup_listen = isup_listen
    self.isup_port = int(isup_port or 7660)
    self.isup_key = isup_key or ""

  # add any new properties here
  def __setstate__(self, state):
    self.__dict__.update(state)
    if not hasattr(self, "qwen_prompt"): self.qwen_prompt = "What has been detected on my CCTV camera? Write in one short sentence"
    if not hasattr(self, "server_url"): self.server_url = "https://clearcam.org"
    if not hasattr(self, "use_ocr"): self.use_ocr = True
    if not hasattr(self, "ocr_engine"): self.ocr_engine = "rapidocr"
    if not hasattr(self, "ocr_plate_mode"): self.ocr_plate_mode = False
    if not hasattr(self, "ocr_interval"): self.ocr_interval = 2.0
    if not hasattr(self, "ocr_pumatronix_hi_res"): self.ocr_pumatronix_hi_res = False
    if not hasattr(self, "isup_listen"): self.isup_listen = True
    if not hasattr(self, "isup_port"): self.isup_port = 7660
    if not hasattr(self, "isup_key"): self.isup_key = ""

  def use_notifs(self): return self.userID or self.server_url != "https://clearcam.org"
  def clearcam_user(self): return self.userID is not None

def secret_settings(settings):
    return GlobalSettings(
        use_clip=settings.use_clip,
        use_face=settings.use_face,
        model_size=settings.model_size,
        model_res=settings.model_res,
        userID=settings.userID is not None,
        key=settings.key is not None,
        use_qwen=settings.use_qwen,
        qwen_size=settings.qwen_size,
        qwen_prompt=settings.qwen_prompt,
        server_url=settings.server_url,
        use_ocr=getattr(settings, "use_ocr", True),
        ocr_engine=getattr(settings, "ocr_engine", "rapidocr"),
        ocr_plate_mode=getattr(settings, "ocr_plate_mode", False),
        ocr_interval=getattr(settings, "ocr_interval", 2.0),
        ocr_pumatronix_hi_res=getattr(settings, "ocr_pumatronix_hi_res", False),
        isup_listen=getattr(settings, "isup_listen", True),
        isup_port=getattr(settings, "isup_port", 7660),
        isup_key=getattr(settings, "isup_key", ""),
    )

if __name__ == "__main__":
  jit_cache = {}
  yolo_jit_cache = {}
  alerts_on = {}
  qwen = None
  multiprocessing.set_start_method("spawn", force=True)
  # tinygrad releases the GIL on every OpenCL call via ctypes; with ~60 capture/OCR/HTTP threads the default 5 ms
  # switch interval made the inference thread wait up to 5 ms to get it back after each kernel launch
  _sys.setswitchinterval(float(_os.environ.get("CLEARCAM_GIL_SWITCH", "0.0005")))
  database = db()
  cams = database.run_get("links", None)
  classes = {"0","1","2","3","5","7"} # person, bike, car, motorbike, bus, truck

  
  from models.objects import ObjectFinder

  object_queue = []
  cam_name = next((arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--cam_name=")), "my_camera")
  
  class_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names').read_text().split("\n")
  color_dict = {label: tuple((((i+1) * 50) % 256, ((i+1) * 100) % 256, ((i+1) * 150) % 256)) for i, label in enumerate(class_labels)}
  cam = None

  global_settings = database.run_get("global_settings", "all")
  if global_settings == {}: # todo, use None?
    global_settings = GlobalSettings()
    database.run_put("global_settings", "all", global_settings)

  model = YOLOv9(global_settings.model_size, res=int(global_settings.model_res))
  object_finder = ObjectFinder()
  cam = VideoCapture()

  if global_settings.use_clip: object_finder.init_clip()
  if global_settings.use_face: object_finder.init_face()
  if getattr(global_settings, "isup_listen", True):
    isup_gateway.start_listener("0.0.0.0", int(getattr(global_settings, "isup_port", 7660)), str(getattr(global_settings, "isup_key", "")))
  threading.Thread(target=ocr_worker_loop, daemon=True, name="OCRWorker").start()
  threading.Thread(target=backfill_existing_plate_gallery, daemon=True, name="PlateBackfill").start()

  if global_settings.use_qwen:
    try:
      qwen = Qwen3VL(size=f"{global_settings.qwen_size}B", res=(544, 960)) # h, w. they need to be multiples of 32
      print("prewarming Qwen")
      qwen.prewarm()
      print("DONE")
    except Exception as exc:
      print(f"[Qwen3VL] GGUF cache incompleto ou erro de carga ({exc}); desativando use_qwen para iniciar Brascontrol Smart Cam.", flush=True)
      qwen = None
      global_settings.use_qwen = False
      database.run_put("global_settings", "all", global_settings)


  req_port = next((int(arg.split("=", 1)[1]) for arg in sys.argv[1:] if arg.startswith("--port=")), int(_os.environ.get("CLEARCAM_PORT", "8088")))
  server = None
  for candidate_port in (req_port, 8088, 8090, 8095):
    try:
      server = ThreadedHTTPServer(('0.0.0.0', candidate_port), RequestHandlerClass=HLSRequestHandler)
      threading.Thread(target=server.serve_forever, daemon=True).start()
      print(f"Brascontrol Smart Cam v{APP_VERSION} — Serving at http://{get_lan_ip()}:{candidate_port} (local: http://localhost:{candidate_port})", flush=True)
      break
    except OSError as e:
      if getattr(e, "winerror", None) in (10048, 10013) or e.errno == socket.errno.EADDRINUSE:
        print(f"Port {candidate_port} in use, trying next...", flush=True)
      else:
        raise
    
  restart_time = (0, 0)
  threading.Thread(
    target=schedule_daily_restart,
    args=(cam, restart_time),
    daemon=True
  ).start()
  cam.start()

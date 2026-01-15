# qr_camera.py
"""
QRCamera (IMX500) - Capture @ 2028x1520, Display @ half resolution
---------------------------------------------------------------
✅ Capture + detection always run at 2028×1520 (so CALIB_CONSTANT stays valid)
✅ Preview window (imshow) is downscaled to half to reduce load
✅ Full-length real-time video recording (writes EVERY captured frame @ 30fps)
✅ Detection logs stored as JSONL every time QR is detected/updated
✅ Decoded payload stored + "send once" helper

Requires: rpicam-vid, OpenCV, numpy
"""

import os
import time
import json
import math
import threading
import subprocess
import logging
from datetime import datetime
from queue import Queue, Full, Empty
from collections import deque
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np

log = logging.getLogger("QR_CAM")


class QRCamera:
    # Fixed capture resolution (calibration depends on this)
    CAP_W = 2028
    CAP_H = 1520
    CAP_FPS = 30.0

    # Display resolution (half)
    DISP_W = CAP_W // 2
    DISP_H = CAP_H // 2

    def __init__(
        self,
        session_dir: str,
        detection_interval: int = 2,     # run QR decode every Nth frame (2 => ~15Hz)
        median_window: int = 3,          # smoothing over last N detections
        max_jump_px: int = 100,          # reject sudden center jumps (pixels, at 2028x1520)
        show_preview: bool = True,
        record_video: bool = True,
    ):
        self.session_dir = session_dir
        self.show_preview = bool(show_preview)
        self.record_video = bool(record_video)

        self.detection_interval = max(1, int(detection_interval))
        self.median_window = max(1, int(median_window))
        self.max_jump_px = int(max_jump_px)

        self.center = (self.CAP_W // 2, self.CAP_H // 2)

        # Artifacts
        self.videos_dir = os.path.join(session_dir, "videos")
        self.dets_dir = os.path.join(session_dir, "detections")
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.dets_dir, exist_ok=True)

        self.video_path = os.path.join(self.videos_dir, "qr_fullrun.mp4")
        self.frames_jsonl = os.path.join(self.dets_dir, "qr_frames.jsonl")

        # Threads & state
        self._cap_active = False
        self._det_active = False
        self._proc: Optional[subprocess.Popen] = None

        self._frame_q: "Queue[np.ndarray]" = Queue(maxsize=120)
        self._history: "deque[Optional[Dict[str, Any]]]" = deque(maxlen=10)

        self._lock = threading.Lock()
        self._latest: Optional[Dict[str, Any]] = None
        self._latest_payload: Optional[str] = None
        self._last_payload_sent: Optional[str] = None

        self._last_good_center: Optional[Tuple[int, int]] = None

        self._qr = cv2.QRCodeDetector()
        self._writer: Optional[cv2.VideoWriter] = None

    # ---------------------- public API ----------------------
    def initialize(self) -> bool:
        # Check rpicam-vid existence
        try:
            r = subprocess.run(["rpicam-vid", "--help"], capture_output=True, text=True, timeout=5)
            if r.returncode != 0:
                raise RuntimeError("rpicam-vid returned non-zero")
        except Exception as e:
            log.error(f"Camera init check failed: {e}")
            return False

        if self.record_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.video_path, fourcc, float(self.CAP_FPS), (self.CAP_W, self.CAP_H))
            if not self._writer.isOpened():
                log.warning("VideoWriter failed to open; continuing without recording")
                self._writer = None

        return True

    def start(self) -> bool:
        cmd = [
            "rpicam-vid",
            "--width", str(self.CAP_W),
            "--height", str(self.CAP_H),
            "--framerate", str(int(self.CAP_FPS)),
            "--codec", "mjpeg",
            "--timeout", "0",
            "--output", "-",
            "--autofocus-mode", "continuous",
            "--nopreview",
        ]

        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
            time.sleep(1.0)
            if self._proc.poll() is not None:
                log.error("rpicam-vid terminated immediately")
                return False
        except Exception as e:
            log.error(f"Failed to start rpicam-vid: {e}")
            return False

        self._cap_active = True
        self._det_active = True

        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._detect_loop, daemon=True).start()

        log.info("🎥 QRCamera started (capture 2028×1520, display half)")
        return True

    def stop(self):
        self._cap_active = False
        self._det_active = False
        time.sleep(0.5)

        if self.show_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass

        if self._writer:
            try:
                self._writer.release()
                log.info(f"🎞️ Video saved: {self.video_path}")
            except Exception:
                pass

        log.info("🛑 QRCamera stopped")

    def get_latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def get_payload_if_new(self) -> Optional[str]:
        with self._lock:
            p = self._latest_payload
        if p and p != self._last_payload_sent:
            self._last_payload_sent = p
            return p
        return None

    def reset_tracking(self):
        with self._lock:
            self._latest = None
            self._latest_payload = None
        self._history.clear()
        self._last_payload_sent = None
        self._last_good_center = None

    # ---------------------- internals ----------------------
    def _append_jsonl(self, rec: Dict[str, Any]):
        try:
            with open(self.frames_jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:
            log.warning(f"JSONL write failed: {e}")

    def _capture_loop(self):
        buf = b""
        while self._cap_active and self._proc and self._proc.poll() is None:
            try:
                chunk = self._proc.stdout.read(4096)
                if not chunk:
                    break
                buf += chunk

                s = buf.find(b"\xff\xd8")
                e = buf.find(b"\xff\xd9")
                if s != -1 and e != -1 and s < e:
                    jpeg = buf[s:e + 2]
                    buf = buf[e + 2:]

                    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    # Record FULL frames for real-time full-length video
                    if self._writer is not None:
                        self._writer.write(frame)

                    # Keep capture queue for detection thread
                    try:
                        self._frame_q.put(frame, timeout=0.01)
                    except Full:
                        # If detector is slow, drop frames (video is still full-length from capture)
                        pass

            except Exception as e:
                log.error(f"Capture loop error: {e}")
                break

    def _detect_loop(self):
        fcount = 0
        while self._det_active:
            try:
                frame = self._frame_q.get(timeout=0.2)
            except Empty:
                continue

            fcount += 1
            if (fcount % self.detection_interval) != 0:
                # still show preview (downscaled) even if skipping detection
                if self.show_preview:
                    self._show_preview(frame)
                continue

            det = self._run_qr(frame)

            if self.show_preview:
                self._show_preview(frame, det)

    def _show_preview(self, frame: np.ndarray, det: Optional[Dict[str, Any]] = None):
        # Downscale ONLY for display
        disp = cv2.resize(frame, (self.DISP_W, self.DISP_H), interpolation=cv2.INTER_AREA)

        # Optional overlay on disp (scaled coordinates)
        if det and det.get("bbox"):
            x1, y1, x2, y2 = det["bbox"]
            sx = self.DISP_W / self.CAP_W
            sy = self.DISP_H / self.CAP_H
            x1d, y1d, x2d, y2d = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
            cv2.rectangle(disp, (x1d, y1d), (x2d, y2d), (0, 255, 0), 2)

            # center cross
            cx, cy = self.DISP_W // 2, self.DISP_H // 2
            cv2.drawMarker(disp, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)

            # QR center
            qx, qy = det["center_x"], det["center_y"]
            qxd, qyd = int(qx * sx), int(qy * sy)
            cv2.circle(disp, (qxd, qyd), 5, (0, 255, 255), -1)

            txt = det.get("payload", "")
            if txt:
                cv2.putText(disp, txt[:32], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        try:
            cv2.imshow("QR Preview (Half Res)", disp)
            cv2.waitKey(1)
        except Exception:
            pass

    def _run_qr(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        cx, cy = self.center
        data, points, _ = self._qr.detectAndDecode(frame)

        if points is None or len(points) == 0:
            with self._lock:
                self._latest = None
            self._history.append(None)
            return None

        pts = points[0].astype(int)  # 4x2
        x1 = int(np.min(pts[:, 0])); y1 = int(np.min(pts[:, 1]))
        x2 = int(np.max(pts[:, 0])); y2 = int(np.max(pts[:, 1]))
        qr_cx = int((x1 + x2) / 2)
        qr_cy = int((y1 + y2) / 2)

        # jump rejection
        if self._last_good_center is not None:
            j = math.hypot(qr_cx - self._last_good_center[0], qr_cy - self._last_good_center[1])
            if j > float(self.max_jump_px):
                # reject this detection (keep last)
                det_last = self.get_latest()
                return det_last

        offx = qr_cx - cx
        offy = qr_cy - cy

        det = {
            "label": "qr",
            "confidence": 1.0 if data else 0.7,
            "bbox": (x1, y1, x2, y2),
            "center_x": qr_cx,
            "center_y": qr_cy,
            "offset_x": offx,
            "offset_y": offy,
            "distance_px": float(math.hypot(offx, offy)),
            "frame_center": (cx, cy),
            "payload": data.strip() if isinstance(data, str) else "",
            "t": time.time(),
        }

        # smoothing (median over last N)
        self._history.append(det)
        recent = [d for d in list(self._history)[-self.median_window:] if d]
        if recent:
            mx = int(np.median([d["center_x"] for d in recent]))
            my = int(np.median([d["center_y"] for d in recent]))
            det["center_x"] = mx
            det["center_y"] = my
            det["offset_x"] = mx - cx
            det["offset_y"] = my - cy
            det["distance_px"] = float(math.hypot(det["offset_x"], det["offset_y"]))

        self._last_good_center = (det["center_x"], det["center_y"])

        # Update shared latest + payload
        with self._lock:
            self._latest = det
            if det["payload"]:
                self._latest_payload = det["payload"]

        # Log detection snapshot
        self._append_jsonl({
            "t": det["t"],
            "bbox": det["bbox"],
            "center": [det["center_x"], det["center_y"]],
            "offset": [det["offset_x"], det["offset_y"]],
            "distance_px": det["distance_px"],
            "payload": det["payload"],
        })

        return det

import os
import time
import json
import math
import logging
from datetime import datetime
from collections import deque
from typing import Optional, Tuple, Dict, Any

import cv2
from picamera2 import Picamera2
from pymavlink import mavutil


# ==========================================================
# CONFIG (change these)
# ==========================================================
SERIAL_PORT = "/dev/ttyAMA0"   # or /dev/serial0
BAUD = 57600

CAPTURE_SIZE = (1280, 720)     # bench test size (your working one)
PREVIEW = True                 # display window
SEND_ONCE_PER_PAYLOAD = True   # avoid spamming MP

# QR stability parameters
STABLE_FRAMES_REQUIRED = 8     # consecutive frames with QR geometry
DETECTION_INTERVAL = 1         # detect every frame (bench). set 2 if CPU high
MEDIAN_WINDOW = 3              # median over last N centers
MAX_JUMP_PX = 80               # reject sudden jumps in center

# Logging / artifacts
SAVE_VIDEO = True
FPS = 30.0

# ==========================================================
# Logging setup
# ==========================================================
def make_session_dir() -> str:
    d = os.path.join("sessions_qr_bench", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(d, "videos"), exist_ok=True)
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    os.makedirs(os.path.join(d, "detections"), exist_ok=True)
    return d

def setup_logging(session_dir: str):
    log = logging.getLogger("QR_BENCH")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(os.path.join(session_dir, "logs", "qr_bench.log"))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    log.addHandler(fh)
    log.addHandler(sh)
    return log


# ==========================================================
# MAVLink: send to Mission Planner "Messages" tab
# ==========================================================
class MPLink:
    def __init__(self, port: str, baud: int, log: logging.Logger):
        self.port = port
        self.baud = baud
        self.log = log
        self.master = None

    def connect(self) -> bool:
        try:
            self.master = mavutil.mavlink_connection(self.port, baud=self.baud)
            # Wait heartbeat so we know MP/Pixhawk link is alive
            self.log.info("📡 Waiting for heartbeat...")
            self.master.wait_heartbeat(timeout=10)
            self.log.info("✅ MAVLink heartbeat OK")
            return True
        except Exception as e:
            self.log.error(f"❌ MAVLink connect failed: {e}")
            return False

    def statustext(self, text: str, severity: int = 6):
        """
        severity: 6=INFO, 4=WARNING, 3=ERROR
        MAVLink STATUSTEXT is max ~50 chars.
        """
        if not self.master:
            return
        try:
            msg = self.master.mav.statustext_encode(severity, text.encode()[:50])
            self.master.mav.send(msg)
        except Exception as e:
            self.log.warning(f"STATUSTEXT send failed: {e}")


# ==========================================================
# QR Detector with smoothing + stability
# ==========================================================
class QRBenchDetector:
    def __init__(self, log: logging.Logger, frame_size: Tuple[int, int]):
        self.log = log
        self.qr = cv2.QRCodeDetector()
        self.W, self.H = frame_size
        self.cx, self.cy = self.W // 2, self.H // 2

        self.history = deque(maxlen=10)
        self.last_good_center: Optional[Tuple[int, int]] = None
        self.stable_frames = 0
        self.last_payload_sent: Optional[str] = None

    def _append_jsonl(self, path: str, rec: Dict[str, Any]):
        try:
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def detect(self, frame_bgr, jsonl_path: str) -> Optional[Dict[str, Any]]:
        """
        Returns a dict:
          {center_x, center_y, offset_x, offset_y, bbox, payload, stable, stable_frames}
        Or None if no QR geometry.
        """
        data, points, _ = self.qr.detectAndDecode(frame_bgr)

        if points is None or len(points) == 0:
            self.stable_frames = 0
            return None

        pts = points.astype(int).reshape(-1, 2)
        x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
        x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
        qx, qy = (x1 + x2) // 2, (y1 + y2) // 2

        # Jump rejection
        if self.last_good_center is not None:
            jump = math.hypot(qx - self.last_good_center[0], qy - self.last_good_center[1])
            if jump > MAX_JUMP_PX:
                # reject this detection as noise
                self.log.debug(f"Jump rejected: {jump:.1f}px")
                return None

        det = {
            "t": time.time(),
            "bbox": (x1, y1, x2, y2),
            "center_x": qx,
            "center_y": qy,
            "offset_x": qx - self.cx,
            "offset_y": qy - self.cy,
            "payload": data.strip() if isinstance(data, str) else "",
        }

        # Median smoothing over last MEDIAN_WINDOW detections
        self.history.append(det)
        recent = [d for d in list(self.history)[-MEDIAN_WINDOW:] if d]
        if recent:
            mx = int(sorted([d["center_x"] for d in recent])[len(recent)//2])
            my = int(sorted([d["center_y"] for d in recent])[len(recent)//2])
            det["center_x"] = mx
            det["center_y"] = my
            det["offset_x"] = mx - self.cx
            det["offset_y"] = my - self.cy

        self.last_good_center = (det["center_x"], det["center_y"])

        # stability count (geometry present)
        self.stable_frames += 1
        det["stable_frames"] = self.stable_frames
        det["stable"] = (self.stable_frames >= STABLE_FRAMES_REQUIRED)

        # log detection snapshot
        self._append_jsonl(jsonl_path, det)
        return det

    def should_send_payload(self, payload: str) -> bool:
        if not payload:
            return False
        if not SEND_ONCE_PER_PAYLOAD:
            return True
        if payload == self.last_payload_sent:
            return False
        self.last_payload_sent = payload
        return True


# ==========================================================
# Main bench module
# ==========================================================
def main():
    session_dir = make_session_dir()
    log = setup_logging(session_dir)

    jsonl_path = os.path.join(session_dir, "detections", "qr_frames.jsonl")
    video_path = os.path.join(session_dir, "videos", "qr_bench.mp4")

    # MAVLink link
    mp = MPLink(SERIAL_PORT, BAUD, log)
    if not mp.connect():
        log.error("Exiting: MAVLink not connected")
        return

    mp.statustext("QR bench module started", severity=6)

    # PiCamera2 setup (your working config)
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": CAPTURE_SIZE, "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)

    # Video writer (optional)
    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, FPS, CAPTURE_SIZE)
        if not writer.isOpened():
            log.warning("Video writer failed, continuing without video")
            writer = None
        else:
            log.info(f"🎞️ Recording video: {video_path}")

    detector = QRBenchDetector(log, CAPTURE_SIZE)

    log.info("✅ Running QR detection (bench). Press 'q' to quit.")
    last_send_t = 0.0

    try:
        frame_count = 0
        while True:
            frame = picam2.capture_array()  # RGB
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # record every frame (real-time playback)
            if writer:
                writer.write(frame_bgr)

            frame_count += 1
            if (frame_count % DETECTION_INTERVAL) != 0:
                if PREVIEW:
                    cv2.imshow("QR Bench", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            det = detector.detect(frame_bgr, jsonl_path)

            if det:
                # Draw box
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw centers
                cv2.circle(frame_bgr, (det["center_x"], det["center_y"]), 6, (0, 255, 255), -1)
                cv2.drawMarker(frame_bgr, (CAPTURE_SIZE[0]//2, CAPTURE_SIZE[1]//2),
                               (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

                payload = det.get("payload", "")
                if payload:
                    cv2.putText(frame_bgr, payload[:40], (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Send to Mission Planner when stable + decoded
                if det.get("stable") and payload and detector.should_send_payload(payload):
                    # throttle a little to avoid spam
                    if time.time() - last_send_t > 0.8:
                        msg = f"QR: {payload}"
                        mp.statustext(msg, severity=6)
                        log.info(f"📨 Sent to MP: {payload}")
                        last_send_t = time.time()

            if PREVIEW:
                cv2.imshow("QR Bench", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            picam2.stop()
        except Exception:
            pass
        if writer:
            try:
                writer.release()
                log.info(f"🎞️ Saved: {video_path}")
            except Exception:
                pass
        mp.statustext("QR bench module stopped", severity=6)
        log.info(f"Session saved at: {session_dir}")


if __name__ == "__main__":
    main()

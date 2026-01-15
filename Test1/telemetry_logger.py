# telemetry_logger.py
"""
TelemetryLogger - logs drone state periodically to CSV + JSONL.

✅ Logs every run into the session folder.
✅ Safe attribute access (won't crash if something isn't available)
✅ Captures: time, mode, armed, lat, lon, alt, groundspeed, heading, battery, gps fix/sats

Usage:
    tlog = TelemetryLogger(vehicle, session_dir)
    tlog.start(period=0.5)
    ...
    tlog.stop()
"""

import os
import csv
import json
import time
import threading
from typing import Optional, Dict, Any


class TelemetryLogger:
    def __init__(self, vehicle, session_dir: str):
        self.vehicle = vehicle
        self.session_dir = session_dir
        self.telemetry_dir = os.path.join(session_dir, "telemetry")
        os.makedirs(self.telemetry_dir, exist_ok=True)

        self.csv_path = os.path.join(self.telemetry_dir, "telemetry.csv")
        self.jsonl_path = os.path.join(self.telemetry_dir, "telemetry.jsonl")

        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._period = 0.5

        self._csv_file = None
        self._csv_writer = None
        self._csv_header_written = False

    def start(self, period: float = 0.5):
        self._period = float(period)
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2)

        if self._csv_file:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass
        self._csv_file = None
        self._csv_writer = None
        self._csv_header_written = False

    def _safe_get(self) -> Dict[str, Any]:
        v = self.vehicle

        def g(obj, attr, default=None):
            try:
                return getattr(obj, attr)
            except Exception:
                return default

        # mode / armed
        mode = g(g(v, "mode", None), "name", "Unknown")
        armed = bool(g(v, "armed", False))

        # location
        lat = lon = alt = None
        try:
            fr = g(g(v, "location", None), "global_relative_frame", None)
            lat = g(fr, "lat", None)
            lon = g(fr, "lon", None)
            alt = g(fr, "alt", None)
        except Exception:
            pass

        groundspeed = g(v, "groundspeed", None)
        heading = g(v, "heading", None)

        # battery
        batt_v = None
        try:
            b = g(v, "battery", None)
            batt_v = g(b, "voltage", None)
        except Exception:
            pass

        # gps
        gps_fix = gps_sats = None
        try:
            gps = g(v, "gps_0", None)
            gps_fix = g(gps, "fix_type", None)
            gps_sats = g(gps, "satellites_visible", None)
        except Exception:
            pass

        # system status
        sys_state = g(g(v, "system_status", None), "state", "Unknown")

        return {
            "t": time.time(),
            "mode": mode,
            "armed": armed,
            "lat": lat, "lon": lon, "alt": alt,
            "groundspeed": groundspeed,
            "heading": heading,
            "battery_v": batt_v,
            "gps_fix": gps_fix,
            "gps_sats": gps_sats,
            "system": sys_state,
        }

    def _loop(self):
        # open CSV lazily
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)

        while not self._stop:
            rec = self._safe_get()

            # CSV
            if not self._csv_header_written:
                self._csv_writer.writerow(list(rec.keys()))
                self._csv_header_written = True
            self._csv_writer.writerow(list(rec.values()))
            self._csv_file.flush()

            # JSONL
            try:
                with open(self.jsonl_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:
                pass

            time.sleep(self._period)

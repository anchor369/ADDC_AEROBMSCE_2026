# main.py
"""
QR Drop Mission (simple)
------------------------
✅ Takeoff
✅ Go to approximate coordinates
✅ Detect QR @ search height
✅ Send decoded message to Mission Planner (STATUSTEXT) + local messages log
✅ Start centering + parallel descent (FollowerController)
✅ Hold center at drop altitude
✅ Drop via servo
✅ Climb back to search height
✅ RTL
✅ Save logs/video/telemetry each run

Only necessary changes vs last generated main:
- Uses new QRCamera (2028x1520 capture, half display)
- Adds TelemetryLogger
- Adds per-run session folder + local messages logging
- Adds climb-back-to-search-height before RTL
"""

import os
import time
import json
import logging
from datetime import datetime

from drone_base_controller import DroneBaseController
from follower_controller import FollowerController
from qr_camera import QRCamera
from telemetry_logger import TelemetryLogger

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("MAIN")

# ---------------- Tunables ----------------
TAKEOFF_ALT_M     = 8.0
SEARCH_ALT_M      = 8.0
DROP_ALT_M        = 5.0

QR_STABLE_FRAMES  = 10
QR_GATE_TIMEOUT_S = 20.0

CENTER_HOLD_S     = 6.0
CENTER_TOL_M      = 0.12

DESCENT_RATE_FAST = 0.22
DESCENT_RATE_SLOW = 0.14

SERVO_CHANNEL     = 3
SERVO_DROP_PWM    = 1000
SERVO_RESET_PWM   = 1900
SERVO_DROP_TIME_S = 1.7

CALIB_CONSTANT    = 1401  # calibrated at 2028×1520 (keep unchanged)

# Approx target GPS from teammate
TARGET_LAT = 12.8808000
TARGET_LON = 77.5603000

# Climb back
CLIMB_VZ_UP        = -0.45  # BODY_NED: negative z is up
CLIMB_TIMEOUT_S    = 35.0

def make_session_dir() -> str:
    session_dir = os.path.join("sessions_qr", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(os.path.join(session_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "messages"), exist_ok=True)
    return session_dir

def append_message(session_dir: str, rec: dict):
    path = os.path.join(session_dir, "messages", "statustext.jsonl")
    try:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass

def mp_text(drone: DroneBaseController, session_dir: str, text: str, severity: int = 6):
    drone.mav_text(text, severity=severity)
    append_message(session_dir, {"t": time.time(), "severity": severity, "text": text})

def wait_for_qr_stable(cam: QRCamera, timeout_s: float, stable_frames: int) -> bool:
    t0 = time.time()
    stable = 0
    while time.time() - t0 < timeout_s:
        det = cam.get_latest()
        if det and det.get("confidence", 0.0) >= 0.7:
            stable += 1
            if stable >= stable_frames:
                return True
        else:
            stable = 0
        time.sleep(0.05)
    return False

def climb_back_to_search_alt(drone: DroneBaseController, session_dir: str) -> None:
    mp_text(drone, session_dir, f"Climb to {SEARCH_ALT_M:.1f}m")
    t0 = time.time()
    while True:
        if drone.pilot_override:
            mp_text(drone, session_dir, "Pilot override: stop climb", severity=4)
            break

        alt = drone.vehicle.location.global_relative_frame.alt
        if alt >= (SEARCH_ALT_M - 0.2):
            break

        if (time.time() - t0) > CLIMB_TIMEOUT_S:
            mp_text(drone, session_dir, "Climb timeout", severity=4)
            break

        drone.send_body_velocity(0.0, 0.0, CLIMB_VZ_UP)
        time.sleep(0.3)

    # stop
    drone.send_body_velocity(0.0, 0.0, 0.0)
    time.sleep(0.5)

def main():
    session_dir = make_session_dir()
    log.info(f"Session: {session_dir}")

    drone = DroneBaseController()
    if not drone.connect_drone("/dev/ttyAMA0", 57600):
        log.error("Drone connection failed")
        return
    drone.start_mode_monitoring()

    # Telemetry logging
    tlog = TelemetryLogger(drone.vehicle, session_dir=session_dir)
    tlog.start(period=0.5)

    # Takeoff
    mp_text(drone, session_dir, "Takeoff")
    if not drone.arm_and_takeoff_safe(TAKEOFF_ALT_M):
        mp_text(drone, session_dir, "Takeoff failed", severity=3)
        tlog.stop()
        drone.cleanup()
        return
    mp_text(drone, session_dir, f"Takeoff OK {TAKEOFF_ALT_M:.1f}m")

    # Go to approximate target at search altitude
    mp_text(drone, session_dir, "Goto target")
    drone.goto_location(TARGET_LAT, TARGET_LON, SEARCH_ALT_M, groundspeed=2.5)
    time.sleep(6.0)  # settle (replace with distance check if you want)

    # Start camera (capture fixed 2028×1520, display half)
    cam = QRCamera(
        session_dir=session_dir,
        detection_interval=2,
        median_window=3,
        max_jump_px=100,
        show_preview=True,
        record_video=True,
    )
    assert cam.initialize(), "Camera init failed"
    assert cam.start(), "Camera start failed"

    # Follower (centering + parallel descent)
    follower = FollowerController(drone.vehicle, cam, calib_constant=CALIB_CONSTANT)

    # If you want to tune offsets live, do here:
    # follower.offset_forward_m = -0.20
    # follower.offset_right_m =  0.10
    # follower.offset_x_px = 0
    # follower.offset_y_px = 0

    try:
        # Phase: detect QR
        mp_text(drone, session_dir, "Looking for QR")
        cam.reset_tracking()

        found = wait_for_qr_stable(cam, timeout_s=QR_GATE_TIMEOUT_S, stable_frames=QR_STABLE_FRAMES)
        if not found:
            mp_text(drone, session_dir, "QR NOT FOUND -> RTL", severity=4)
            cam.stop()
            drone.safe_rtl()
            tlog.stop()
            drone.cleanup()
            return

        # Send decoded payload to MP (once)
        payload = cam.get_payload_if_new()
        if payload:
            mp_text(drone, session_dir, f"QR: {payload}")
        else:
            mp_text(drone, session_dir, "QR detected (decode empty)", severity=4)

        # Phase: center + descend to DROP_ALT_M
        mp_text(drone, session_dir, "Center+Descend")
        while True:
            if drone.pilot_override:
                mp_text(drone, session_dir, "Pilot override: stopping", severity=4)
                break

            alt = drone.vehicle.location.global_relative_frame.alt
            if alt <= (DROP_ALT_M + 0.3):
                break

            rate = DESCENT_RATE_FAST if alt > (DROP_ALT_M + 1.0) else DESCENT_RATE_SLOW
            s = follower.step(
                allow_descent=True,
                descent_rate_fast=DESCENT_RATE_FAST,
                descent_rate_slow=DESCENT_RATE_SLOW,
                hold_alt_below_m=DROP_ALT_M,
            )

            # Optional: if you want additional tolerance check
            # if s.get("err_norm", 999.0) < 0.25: descent is allowed by follower gating anyway

            time.sleep(0.05)

        # Phase: hold & center at drop altitude
        mp_text(drone, session_dir, "Hold@DropAlt")
        t0 = time.time()
        while time.time() - t0 < CENTER_HOLD_S:
            if drone.pilot_override:
                break
            s = follower.step(allow_descent=False, hold_alt_below_m=DROP_ALT_M)
            time.sleep(0.05)

        # Drop
        mp_text(drone, session_dir, "DROP")
        drone.set_servo(SERVO_CHANNEL, SERVO_DROP_PWM)
        time.sleep(SERVO_DROP_TIME_S)
        drone.set_servo(SERVO_CHANNEL, SERVO_RESET_PWM)
        mp_text(drone, session_dir, "Drop complete")

        # Climb back to search altitude
        climb_back_to_search_alt(drone, session_dir)

        # RTL
        if not drone.pilot_override:
            mp_text(drone, session_dir, "RTL")
            drone.safe_rtl()
        else:
            mp_text(drone, session_dir, "Pilot override: not forcing RTL", severity=4)

    except KeyboardInterrupt:
        mp_text(drone, session_dir, "KeyboardInterrupt -> LAND", severity=4)
        drone.safe_land()
    except Exception as e:
        mp_text(drone, session_dir, f"Exception: {type(e).__name__}", severity=3)
        try:
            drone.safe_land()
        except Exception:
            pass
    finally:
        try:
            cam.stop()
        except Exception:
            pass
        tlog.stop()
        drone.cleanup()
        log.info("✅ Mission complete and cleaned up")

if __name__ == "__main__":
    main()

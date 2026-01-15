# follower_controller.py
"""
FollowerController (QR / Helipad centering + smooth descend)
-----------------------------------------------------------
Designed as a drop-in upgrade of your previous Aerothon follower_controller.py.

Goals (as you asked):
- Start centering immediately when detection is present
- Descend in parallel with centering (but safely gated so it doesn't dive while off-center)
- Steady: no sudden velocity jumps (slew limiting + caps + taper + smoothing-friendly)
- Not too slow: altitude-based velocity caps (fast high, gentle low)
- Less overshoot: tapering near center + deadband + integrator clamp
- Break at center: deadband + locked_frames
- Offset knobs: camera/payload offset in meters (+/-), easy to tune
- Step-wise smooth behavior: high/mid/low altitude “gears”
- Safety checks: vision-lost hover, wrong-way detection, progress watchdog freeze

Expected camera output (from your QR camera module):
cam.get_latest() returns dict with:
  {
    "offset_x": px (QR center - frame_center_x), + means QR to right
    "offset_y": px (QR center - frame_center_y), + means QR lower (down)
    "center_x","center_y","bbox","confidence","label",...
  }

This controller outputs BODY_NED velocities (m/s):
  vx: forward (+), vy: right (+), vz: down (+)

"""

import time, math, logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

from drone_mavlink import set_body_velocity  # uses MAV_FRAME_BODY_NED


# ==========================================================
# ✨ TUNABLES (keep ALL changeables here)
# ==========================================================
class FOLLOW:
    # ---- Calibration ----
    CALIB_CONSTANT = 1401.0  # px/m * altitude_m  (same concept as your old code)

    # ---- Core control gains ----
    KP = 0.65
    KI = 0.07

    # ---- Centering behavior ----
    DEADBAND_M = 0.12        # inside this → vx=vy=0 (break at center)
    LOCK_BAND_M = 0.22       # “good enough band” for stable descent gating
    LOCK_FRAMES = 10         # consecutive frames inside deadband to consider “locked”
    DESCENT_LOCK_FRAMES = 12 # consecutive frames inside lock_band for allowing descent

    # ---- Altitude-based speed caps (fast at high alt, gentle near ground) ----
    VCAP_HIGH = 0.90         # alt > HIGH_ALT_M
    VCAP_MID  = 0.55         # MID_ALT_M < alt <= HIGH_ALT_M
    VCAP_LOW  = 0.35         # alt <= MID_ALT_M
    HIGH_ALT_M = 8.0
    MID_ALT_M  = 5.5

    # ---- Smoothness ----
    SLEW_LIMIT = 0.08        # max delta per step (m/s)

    # ---- Anti-overshoot / stability helpers ----
    TAPER_RADIUS_M = 0.40    # taper scale: gain reduces when err is small (<= this)
    INTEGRATOR_CLAMP = 0.30  # max abs integrator

    # ---- Vision loss ----
    LOSS_HOVER_S = 2.0       # after this warn, but still hover
    LOSS_ABORT_S = 4.0       # after this you should consider RTL/land in main logic

    # ---- Wrong-way detector (prevents runaway when axis mapping is wrong) ----
    WRONG_WAY_FRAMES = 6
    WRONG_WAY_GAIN_SCALE = 0.35  # if wrong-way triggers, reduce KP and freeze briefly
    WRONG_WAY_FREEZE_S = 0.6

    # ---- Progress watchdog (if not improving, freeze briefly) ----
    PROGRESS_INTERVAL_S = 1.6
    PROGRESS_MIN_IMPROVEMENT = 0.12  # needs >=12% improvement per interval
    PROGRESS_FREEZE_S = 0.7

    # ---- Descent policy ----
    # Step-wise descent rates: your main.py can override too.
    DESCENT_RATE_FAST = 0.22
    DESCENT_RATE_SLOW = 0.14
    DESCENT_SWITCH_ALT_M = 6.2

    # If far from center, either stop descent or do a tiny “creep” descent
    DESCENT_ALLOW_ERR_M = 0.30
    DESCENT_CREEP_RATE = 0.08  # small descent when error is larger (optional)
    ENABLE_CREEP_DESCENT = True

    # ---- Offset knobs (VERY IMPORTANT) ----
    # These offsets are applied in METERS in BODY frame.
    # +offset_forward_m means "target is forward of camera center" so drone moves forward to align.
    # If your payload release point is *behind* camera target, you may set negative offset_forward_m, etc.
    OFFSET_FORWARD_M = 0.0   # (+forward, -back)
    OFFSET_RIGHT_M   = 0.0   # (+right, -left)

    # Optional: pixel bias (rarely needed, but useful if image center is not true center)
    OFFSET_X_PX = 0.0        # (+ means treat target more to right)
    OFFSET_Y_PX = 0.0        # (+ means treat target more down)

    # ---- Minimum altitude safety ----
    MIN_ALT_FOR_CONTROL_M = 1.2  # below this, stop velocity control (handover to LAND)


# ==========================================================
# Internal state
# ==========================================================
ctrl = logging.getLogger("FOLLOW")


@dataclass
class LockState:
    locked_frames: int = 0
    descent_frames: int = 0

    last_ppm: Optional[float] = None
    ix: float = 0.0
    iy: float = 0.0

    last_vx: float = 0.0
    last_vy: float = 0.0
    last_time: Optional[float] = None

    vision_lost_since: Optional[float] = None

    # Wrong-way detection
    last_err_x: Optional[float] = None
    last_err_y: Optional[float] = None
    wrong_dir_count: int = 0
    freeze_until: Optional[float] = None

    # Progress watchdog
    progress_start: Optional[float] = None
    progress_ref_err: Optional[float] = None
    progress_freeze_until: Optional[float] = None


# ==========================================================
# FollowerController
# ==========================================================
class FollowerController:
    def __init__(self, vehicle, camera, calib_constant: float = FOLLOW.CALIB_CONSTANT):
        self.v = vehicle
        self.cam = camera
        self.calib_k = float(calib_constant) if calib_constant else None
        self.state = LockState()

        # Expose offsets as instance variables so you can tune them live
        self.offset_forward_m = FOLLOW.OFFSET_FORWARD_M
        self.offset_right_m = FOLLOW.OFFSET_RIGHT_M
        self.offset_x_px = FOLLOW.OFFSET_X_PX
        self.offset_y_px = FOLLOW.OFFSET_Y_PX

        # Expose core tunables (so you can tune from main if needed)
        self.kp = FOLLOW.KP
        self.ki = FOLLOW.KI
        self.deadband_m = FOLLOW.DEADBAND_M
        self.lock_band_m = FOLLOW.LOCK_BAND_M
        self.lock_frames = FOLLOW.LOCK_FRAMES
        self.descent_lock_frames = FOLLOW.DESCENT_LOCK_FRAMES

    # ---------------- Utilities ----------------
    def _alt(self) -> float:
        return float(getattr(getattr(getattr(self.v, "location", None), "global_relative_frame", None), "alt", 0.0) or 0.0)

    def _ppm(self, alt: float) -> float:
        """Pixels per meter: ppm = K / altitude. Includes sanity rejection vs jumps."""
        alt = max(0.25, float(alt))
        ppm = self.calib_k / alt

        # Reject huge ppm jumps (prevents sudden velocity spikes if alt glitches)
        if self.state.last_ppm is not None:
            jump = abs(ppm - self.state.last_ppm)
            if jump > 0.25 * self.state.last_ppm:  # 25% jump reject
                return self.state.last_ppm

        self.state.last_ppm = ppm
        return ppm

    def _vel_cap(self, alt: float) -> float:
        if alt > FOLLOW.HIGH_ALT_M:
            return FOLLOW.VCAP_HIGH
        if alt > FOLLOW.MID_ALT_M:
            return FOLLOW.VCAP_MID
        return FOLLOW.VCAP_LOW

    def _slew(self, new_v: float, last_v: float) -> float:
        dv = new_v - last_v
        if dv > FOLLOW.SLEW_LIMIT:
            return last_v + FOLLOW.SLEW_LIMIT
        if dv < -FOLLOW.SLEW_LIMIT:
            return last_v - FOLLOW.SLEW_LIMIT
        return new_v

    def _send(self, vx: float, vy: float, vz: float):
        """Send BODY_NED velocity command."""
        set_body_velocity(self.v, vx, vy, vz)

    # ==========================================================
    # Main step
    # ==========================================================
    def step(
        self,
        allow_descent: bool = True,
        descent_rate_fast: float = FOLLOW.DESCENT_RATE_FAST,
        descent_rate_slow: float = FOLLOW.DESCENT_RATE_SLOW,
        hold_alt_below_m: Optional[float] = None,  # if set, never descend below this
    ) -> Dict[str, Any]:
        """
        Run one control iteration:
        - centers in XY always when vision present
        - descends only when stable enough (and not below hold_alt_below_m)
        - returns a summary dict you can log.

        Important:
        - For your mission: call step() at ~20 Hz (sleep 0.05)
        """
        now = time.time()
        det = self.cam.get_latest()
        alt = self._alt()

        # Safety: below minimum, stop controlling (handover to LAND in main)
        if alt > 0 and alt < FOLLOW.MIN_ALT_FOR_CONTROL_M:
            self._send(0, 0, 0)
            return {
                "reason": "alt_too_low_handover",
                "alt": alt,
                "vx": 0.0, "vy": 0.0, "vz": 0.0,
                "err_norm": 999.0,
            }

        ppm = self._ppm(alt) if (det and self.calib_k) else None
        vision_ok = bool(det and ppm)

        # ---------------- Vision lost handling ----------------
        if not vision_ok:
            if self.state.vision_lost_since is None:
                self.state.vision_lost_since = now
                ctrl.info("VISION LOST → hover")
            lost_dur = now - self.state.vision_lost_since
            if lost_dur >= FOLLOW.LOSS_HOVER_S:
                ctrl.warning("VISION LOST (long) → hover, consider safety action soon")
            self._send(0, 0, 0)
            return {
                "reason": "vision_lost",
                "alt": alt,
                "vx": 0.0, "vy": 0.0, "vz": 0.0,
                "err_norm": 999.0,
                "vision_lost_s": lost_dur,
            }

        # vision available again
        self.state.vision_lost_since = None

        # ---------------- Apply pixel bias offsets ----------------
        # offset_x_px/+ means treat target slightly right → increases error_y_body (vy).
        # offset_y_px/+ means treat target slightly down  → affects error_x_body (vx) after mapping.
        offx_px = float(det["offset_x"]) + float(self.offset_x_px)
        offy_px = float(det["offset_y"]) + float(self.offset_y_px)

        # ---------------- Error computation (pixels -> meters) ----------------
        # Image axes:
        #   +x right, +y down
        # Convert to meters and flip Y so +ey means "up" in image sense.
        ex_m = offx_px / ppm
        ey_m = (offy_px / ppm) * -1.0

        # Map to BODY frame (same mapping as your old controller):
        # err_x_body drives vx (forward/back), err_y_body drives vy (right/left)
        err_x_body = ey_m
        err_y_body = ex_m

        # ---------------- Apply camera/payload offset compensation in METERS ----------------
        # These are the knobs you will tune on ground.
        err_x_body += float(self.offset_forward_m)
        err_y_body += float(self.offset_right_m)

        err_norm = math.hypot(err_x_body, err_y_body)

        # ---------------- Freeze windows (wrong-way / watchdog) ----------------
        if self.state.freeze_until and now < self.state.freeze_until:
            self._send(0, 0, 0)
            return {
                "reason": "freeze_wrong_way",
                "alt": alt, "vx": 0.0, "vy": 0.0, "vz": 0.0,
                "err_x": err_x_body, "err_y": err_y_body, "err_norm": err_norm,
            }

        if self.state.progress_freeze_until and now < self.state.progress_freeze_until:
            self._send(0, 0, 0)
            return {
                "reason": "freeze_no_progress",
                "alt": alt, "vx": 0.0, "vy": 0.0, "vz": 0.0,
                "err_x": err_x_body, "err_y": err_y_body, "err_norm": err_norm,
            }

        # ---------------- Wrong-way detector ----------------
        # If error magnitude keeps increasing on same sign, your axis mapping might be wrong or wind too high.
        Kp_eff = self.kp
        wrong_axis = False

        if self.state.last_err_x is not None:
            if abs(err_x_body) > abs(self.state.last_err_x) and (math.copysign(1, err_x_body) == math.copysign(1, self.state.last_err_x)):
                self.state.wrong_dir_count += 1
                wrong_axis = True

        if self.state.last_err_y is not None:
            if abs(err_y_body) > abs(self.state.last_err_y) and (math.copysign(1, err_y_body) == math.copysign(1, self.state.last_err_y)):
                self.state.wrong_dir_count += 1
                wrong_axis = True

        if wrong_axis and self.state.wrong_dir_count >= FOLLOW.WRONG_WAY_FRAMES:
            ctrl.warning("WRONG DIRECTION SUSPECTED → freeze + reduce gains")
            self.state.wrong_dir_count = 0
            self.state.freeze_until = now + FOLLOW.WRONG_WAY_FREEZE_S
            self._send(0, 0, 0)
            return {
                "reason": "wrong_way_triggered",
                "alt": alt, "vx": 0.0, "vy": 0.0, "vz": 0.0,
                "err_x": err_x_body, "err_y": err_y_body, "err_norm": err_norm,
            }
        else:
            # reset if not trending wrong
            if not wrong_axis:
                self.state.wrong_dir_count = 0

        self.state.last_err_x, self.state.last_err_y = err_x_body, err_y_body

        # ---------------- Deadband / lock ----------------
        if err_norm < self.deadband_m:
            self.state.locked_frames += 1
            vx = vy = 0.0
        else:
            self.state.locked_frames = 0

            # PI control
            dt = 0.05 if self.state.last_time is None else max(0.02, now - self.state.last_time)
            self.state.last_time = now

            self.state.ix += err_x_body * dt
            self.state.iy += err_y_body * dt
            self.state.ix = max(-FOLLOW.INTEGRATOR_CLAMP, min(FOLLOW.INTEGRATOR_CLAMP, self.state.ix))
            self.state.iy = max(-FOLLOW.INTEGRATOR_CLAMP, min(FOLLOW.INTEGRATOR_CLAMP, self.state.iy))

            # taper reduces overshoot near center
            taper = min(1.0, err_norm / FOLLOW.TAPER_RADIUS_M)

            vx = (Kp_eff * err_x_body * taper) + (self.ki * self.state.ix)
            vy = (Kp_eff * err_y_body * taper) + (self.ki * self.state.iy)

        # ---------------- Caps + slew ----------------
        vcap = self._vel_cap(alt)
        vx = max(-vcap, min(vcap, vx))
        vy = max(-vcap, min(vcap, vy))

        vx = self._slew(vx, self.state.last_vx)
        vy = self._slew(vy, self.state.last_vy)
        self.state.last_vx, self.state.last_vy = vx, vy

        # ---------------- Progress watchdog ----------------
        # If we don't improve within PROGRESS_INTERVAL_S, freeze briefly.
        if self.state.progress_start is None:
            self.state.progress_start = now
            self.state.progress_ref_err = err_norm
        else:
            if (now - self.state.progress_start) > FOLLOW.PROGRESS_INTERVAL_S:
                ref = self.state.progress_ref_err if self.state.progress_ref_err is not None else err_norm
                need = (1.0 - FOLLOW.PROGRESS_MIN_IMPROVEMENT) * ref
                if err_norm > need:
                    ctrl.warning("NO PROGRESS → freezing velocities briefly")
                    self.state.progress_freeze_until = now + FOLLOW.PROGRESS_FREEZE_S
                    self._send(0, 0, 0)
                    self.state.progress_start = now
                    self.state.progress_ref_err = err_norm
                    return {
                        "reason": "no_progress_freeze",
                        "alt": alt, "vx": 0.0, "vy": 0.0, "vz": 0.0,
                        "err_x": err_x_body, "err_y": err_y_body, "err_norm": err_norm,
                    }

                self.state.progress_start = now
                self.state.progress_ref_err = err_norm

        # ---------------- Descent gating (parallel descend + centering) ----------------
        # Track stability in a broader band for descent permission
        if err_norm < self.lock_band_m:
            self.state.descent_frames += 1
        else:
            self.state.descent_frames = 0

        locked = (self.state.locked_frames >= self.lock_frames)
        stable_for_descent = (self.state.descent_frames >= self.descent_lock_frames)

        # Choose descent rate “gear”
        if alt > FOLLOW.DESCENT_SWITCH_ALT_M:
            desired_vz = float(descent_rate_fast)
        else:
            desired_vz = float(descent_rate_slow)

        # Hard stop below hold altitude if requested
        if hold_alt_below_m is not None and alt <= float(hold_alt_below_m):
            desired_vz = 0.0
            allow_descent = False

        # Also stop descent if far from center (unless creep enabled)
        if allow_descent:
            if err_norm <= FOLLOW.DESCENT_ALLOW_ERR_M and stable_for_descent:
                vz = desired_vz
            else:
                if FOLLOW.ENABLE_CREEP_DESCENT and err_norm <= (FOLLOW.DESCENT_ALLOW_ERR_M * 1.6):
                    vz = min(desired_vz, FOLLOW.DESCENT_CREEP_RATE)
                else:
                    vz = 0.0
        else:
            vz = 0.0

        # send command
        self._send(vx, vy, vz)

        return {
            "reason": "ok",
            "alt": alt,
            "ppm": ppm,
            "err_x": err_x_body,
            "err_y": err_y_body,
            "err_norm": err_norm,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "locked": locked,
            "stable_for_descent": stable_for_descent,
            "offset_forward_m": self.offset_forward_m,
            "offset_right_m": self.offset_right_m,
            "offset_x_px": self.offset_x_px,
            "offset_y_px": self.offset_y_px,
        }

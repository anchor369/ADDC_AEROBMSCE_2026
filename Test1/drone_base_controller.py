# drone_base_controller.py
if sys.version_info >= (3, 10):
    import collections.abc
    import collections
    collections.MutableMapping = collections.abc.MutableMapping
import time
import threading
import logging
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil

log = logging.getLogger("DRONE")

class DroneBaseController:
    def __init__(self):
        self.vehicle = None
        self.pilot_override = False
        self.autonomous_active = False
        self.expected_mode = None
        self.stop_monitoring = False
        self.monitor_thread = None

    def connect_drone(self, connection_string="/dev/ttyAMA0", baud_rate=57600):
        log.info(f"📡 Connecting: {connection_string} @ {baud_rate}")
        try:
            self.vehicle = connect(connection_string, baud=baud_rate, wait_ready=False)
            time.sleep(2)
            log.info("✅ Connected")
            try:
                log.info(f"Mode: {self.vehicle.mode.name}")
            except:
                pass
            return True
        except Exception as e:
            log.exception(f"❌ Connect failed: {e}")
            return False

    # ---------------- Mode monitoring (pilot override) ----------------
    def start_mode_monitoring(self):
        self.stop_monitoring = False
        self.monitor_thread = threading.Thread(target=self._mode_monitor_loop, daemon=True)
        self.monitor_thread.start()
        log.info("👁️ Mode monitoring started")

    def _mode_monitor_loop(self):
        while not self.stop_monitoring and self.vehicle:
            try:
                current_mode = self.vehicle.mode.name
                if self.autonomous_active and self.expected_mode and current_mode != self.expected_mode:
                    log.warning(f"🚨 PILOT OVERRIDE! expected={self.expected_mode}, actual={current_mode}")
                    self.pilot_override = True
                    self.autonomous_active = False
                time.sleep(0.5)
            except Exception:
                time.sleep(1)

    def safe_mode_change(self, target_mode: str, timeout_s: float = 5.0):
        if self.pilot_override:
            log.warning("🤚 Pilot override active: refusing mode change")
            return False
        try:
            self.vehicle.mode = VehicleMode(target_mode)
            self.expected_mode = target_mode
            t0 = time.time()
            while time.time() - t0 < timeout_s:
                if self.pilot_override:
                    return False
                if self.vehicle.mode.name == target_mode:
                    return True
                time.sleep(0.1)
            log.warning(f"⚠️ Mode change timeout → {self.vehicle.mode.name}")
            return False
        except Exception as e:
            log.exception(f"Mode change failed: {e}")
            return False

    # ---------------- Mission primitives ----------------
    def arm_and_takeoff_safe(self, target_alt_m: float):
        if self.pilot_override:
            return False

        self.autonomous_active = True
        if not self.safe_mode_change("GUIDED"):
            return False

        log.info("🔍 Waiting for armable...")
        while not self.vehicle.is_armable:
            if self.pilot_override:
                return False
            time.sleep(1)

        log.info("⚡ Arming motors")
        self.vehicle.armed = True
        t0 = time.time()
        while not self.vehicle.armed and time.time() - t0 < 10:
            if self.pilot_override:
                return False
            time.sleep(0.5)
        if not self.vehicle.armed:
            log.error("❌ Arm failed")
            return False

        log.info(f"🛫 Taking off to {target_alt_m:.1f}m")
        self.vehicle.simple_takeoff(target_alt_m)

        while True:
            if self.pilot_override:
                return False
            alt = self.vehicle.location.global_relative_frame.alt
            log.info(f"Alt: {alt:.2f}m")
            if alt >= target_alt_m * 0.95:
                break
            time.sleep(1)

        return True

    def goto_location(self, lat: float, lon: float, alt_m: float, groundspeed: float = 2.0):
        if self.pilot_override:
            return False
        try:
            self.vehicle.groundspeed = groundspeed
            tgt = LocationGlobalRelative(lat, lon, alt_m)
            self.vehicle.simple_goto(tgt)
            return True
        except Exception as e:
            log.exception(f"goto failed: {e}")
            return False

    def safe_land(self):
        return self.safe_mode_change("LAND")

    def safe_rtl(self):
        return self.safe_mode_change("RTL")

    # ---------------- MAVLink helpers ----------------
    def mav_text(self, text: str, severity: int = 6):
        """Shows in Mission Planner Messages tab (STATUSTEXT)."""
        try:
            msg = self.vehicle.message_factory.statustext_encode(severity, text.encode()[:50])
            self.vehicle.send_mavlink(msg)
            self.vehicle.flush()
        except Exception:
            pass

    def set_servo(self, channel: int, pwm: int):
        """MAV_CMD_DO_SET_SERVO"""
        try:
            msg = self.vehicle.message_factory.command_long_encode(
                0, 0,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,
                channel, pwm,
                0, 0, 0, 0, 0
            )
            self.vehicle.send_mavlink(msg)
            self.vehicle.flush()
            log.info(f"SERVO ch={channel} pwm={pwm}")
        except Exception as e:
            log.exception(f"set_servo failed: {e}")

    def send_body_velocity(self, vx: float, vy: float, vz: float):
        """BODY_NED velocity. X fwd, Y right, Z down."""
        try:
            type_mask = 0b0000111111000111
            msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
                0, 0, 0,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                type_mask,
                0, 0, 0,
                vx, vy, vz,
                0, 0, 0,
                0, 0
            )
            self.vehicle.send_mavlink(msg)
            self.vehicle.flush()
        except Exception:
            pass

    def cleanup(self):
        self.stop_monitoring = True
        try:
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2)
        except:
            pass
        try:
            if self.vehicle:
                self.vehicle.close()
        except:
            pass
        log.info("🔒 Cleanup done")

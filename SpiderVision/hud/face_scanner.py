"""
SpiderVision — Face Scanner HUD
Renders scanning beam, tracking box, face landmarks,
depth circle, target lock, and confidence indicators.
Uses OpenCV-drawn elements composited onto camera texture.
"""
import math
import time
import cv2
import numpy as np
from config.settings import (
    WINDOW_W, WINDOW_H, CAMERA_W, CAMERA_H,
    C_BLUE, C_RED, C_WHITE
)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class FaceScannerHUD:
    """
    Renders all face-tracking HUD elements onto a BGR numpy frame.
    Results are then uploaded to the camera texture for display.
    """

    def __init__(self):
        self._scan_y     = 0.0   # 0..1 beam position
        self._scan_dir   = 1.0
        self._scan_pct   = 0.0   # 0..100
        self._locked     = False
        self._lock_time  = 0.0
        self._lock_flash = False
        self._tts_said_hello      = False
        self._tts_said_spider     = False
        self._last_identity = None

    # ── Draw ──────────────────────────────────────────────────────────────────

    def render(self, bgr: np.ndarray, face_data, dt: float, audio=None) -> np.ndarray:
        """
        Draw face HUD on bgr frame. Returns modified frame.
        """
        out = bgr.copy()
        h, w = out.shape[:2]
        t = time.time()

        identity = face_data.identity_status
        session_unlocked = getattr(face_data, "session_unlocked", False)
        # A confirmed stranger or enrollment change starts a new welcome cycle.
        # Absence and transient verification do not end an unlocked session.
        if identity in ("DANGER", "DANGER_OTHER", "NOT_ENROLLED", "ENROLLING"):
            self._tts_said_hello = False
            self._scan_pct = 0.0
            self._locked = False
        if identity in ("DANGER", "DANGER_OTHER") and self._last_identity != identity and audio:
            audio.play("warning")
            if identity == "DANGER_OTHER":
                audio.speak("Danger, Mister. Unknown person detected ahead.")
        self._last_identity = identity
        if not face_data.detected:
            if not session_unlocked:
                self._scan_pct = max(0.0, self._scan_pct - dt * 30)
                self._locked = False
            if identity == "DANGER_OTHER":
                cv2.putText(out, "DANGER - UNKNOWN PERSON AHEAD", (24, h - 48),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            elif session_unlocked:
                cv2.putText(out, "SESSION ACTIVE - FACE OUT OF VIEW", (24, h - 48),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 210, 120), 2, cv2.LINE_AA)
            else:
                cv2.putText(out, "NO FACE | E ENROLL | D DELETE", (24, h - 48),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 180, 255), 2, cv2.LINE_AA)
            return out

        # Scale box coords to frame pixels
        bx  = int(face_data.box_x * w)
        by  = int(face_data.box_y * h)
        bw  = int(face_data.box_w * w)
        bh  = int(face_data.box_h * h)
        cx  = bx + bw // 2
        cy  = by + bh // 2

        # ── Scanning beam ────────────────────────────────────────────────────
        self._scan_y += self._scan_dir * dt * 2.0
        if self._scan_y > 1.0:
            self._scan_y = 1.0
            self._scan_dir = -1.0
        if self._scan_y < 0.0:
            self._scan_y = 0.0
            self._scan_dir = 1.0

        beam_y = by + int(self._scan_y * bh)
        cv2.line(out, (bx, beam_y), (bx + bw, beam_y), (255, 200, 0), 1, cv2.LINE_AA)
        # Glow trail
        for off in range(1, 6):
            alpha = 0.5 - off * 0.09
            if alpha < 0: break
            by_trail = beam_y - int(off * self._scan_dir * 2)
            color = (int(255 * alpha), int(200 * alpha), 0)
            if 0 <= by_trail < h:
                cv2.line(out, (bx, by_trail), (bx + bw, by_trail), color, 1, cv2.LINE_AA)

        # ── Tracking box (animated corners) ──────────────────────────────────
        cl = 20   # corner length
        thick = 2
        col_box = (0, 0, 255) if identity in ("DANGER", "DANGER_OTHER") else (0, 191, 255)
        # Animated pulse
        pulse_off = int(3 * math.sin(t * 5.0))
        x1, y1, x2, y2 = bx - 4, by - 4, bx + bw + 4, by + bh + 4
        # 4 corner L-brackets
        for px, py, sx, sy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                (x2, y2, -1, -1), (x1, y2, 1, -1)]:
            cv2.line(out, (px, py), (px + sx * cl, py), col_box, thick, cv2.LINE_AA)
            cv2.line(out, (px, py), (px, py + sy * cl), col_box, thick, cv2.LINE_AA)

        # Box outline (faint)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 80, 160), 1, cv2.LINE_AA)

        # ── Face Landmarks ────────────────────────────────────────────────────
        if face_data.landmarks:
            from tracking.face_tracker import FaceTracker
            for idx in FaceTracker.KEY_INDICES:
                if idx < len(face_data.landmarks):
                    lx = int(face_data.landmarks[idx][0] * w)
                    ly = int(face_data.landmarks[idx][1] * h)
                    cv2.circle(out, (lx, ly), 1, (0, 255, 200), -1, cv2.LINE_AA)

        # ── Depth circle ──────────────────────────────────────────────────────
        depth_r = int(15 + face_data.depth_estimate * 40)
        cv2.ellipse(out, (cx, cy), (depth_r, depth_r),
                    0, 0, 360, (100, 200, 255), 1, cv2.LINE_AA)
        cv2.ellipse(out, (cx, cy), (depth_r + 4, depth_r + 4),
                    0, 0, int(face_data.confidence * 360),
                    (0, 200, 255), 1, cv2.LINE_AA)

        # ── Scanning progress ─────────────────────────────────────────────────
        if identity == "AUTHORIZED":
            self._scan_pct = min(100.0, self._scan_pct + dt * 35)
        elif not session_unlocked:
            self._scan_pct = 0.0
            self._locked = False

        # ── Target lock ───────────────────────────────────────────────────────
        if self._scan_pct >= 100.0 and not self._locked:
            self._locked    = True
            self._lock_time = t
            if audio:
                audio.play("lock")
            if not self._tts_said_hello:
                if audio:
                    audio.speak("Welcome, Mister. Face scan complete. Spider Vision online.")
                self._tts_said_hello = True

        if self._locked:
            flash = (math.sin((t - self._lock_time) * 8.0) > 0)
            lock_col = (0, 255, 60) if flash else (0, 180, 40)
            cv2.line(out, (cx - 20, cy), (cx + 20, cy), lock_col, 1, cv2.LINE_AA)
            cv2.line(out, (cx, cy - 20), (cx, cy + 20), lock_col, 1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 8, lock_col, 1, cv2.LINE_AA)


        # ── HUD text ──────────────────────────────────────────────────────────
        font = cv2.FONT_HERSHEY_DUPLEX
        # Scanning / Locked
        status_txt = "SESSION ACTIVE" if session_unlocked and self._locked else {
            "DANGER": "DANGER - UNKNOWN FACE",
            "DANGER_OTHER": "DANGER - UNKNOWN PERSON AHEAD",
            "NOT_ENROLLED": "PRESS E TO ENROLL FACE",
            "ENROLLING": f"ENROLLING {face_data.enroll_progress}",
            "SCANNING": "VERIFYING FACE",
        }.get(identity, "SCAN COMPLETE" if self._locked else f"SCANNING {int(self._scan_pct)}%")
        status_col = (0, 0, 255) if identity in ("DANGER", "DANGER_OTHER") else ((0, 255, 80) if self._locked else (0, 200, 255))
        cv2.putText(out, status_txt, (bx, by - 24), font, 0.45, status_col, 1, cv2.LINE_AA)
        cv2.putText(out, "WEB: SPIDER HAND OR PINCH  /  S: PHOTO",
                    (24, h - 20), font, 0.43, (0, 190, 255), 1, cv2.LINE_AA)
        # Confidence bar
        bar_w = bw
        bar_h = 4
        bar_x, bar_y = bx, by + bh + 12
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (30, 60, 80), -1)
        fill = int(bar_w * face_data.confidence)
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
                      (0, 200, 255), -1)
        conf_txt = f"CONF {face_data.confidence:.2f}"
        cv2.putText(out, conf_txt, (bar_x, bar_y + 18), font, 0.38,
                    (0, 160, 200), 1, cv2.LINE_AA)

        return out

    @property
    def is_locked(self) -> bool:
        return self._locked

    @property
    def scan_pct(self) -> float:
        return self._scan_pct

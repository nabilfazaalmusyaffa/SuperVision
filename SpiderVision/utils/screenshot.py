"""
SpiderVision — Screenshot Utility
Captures the rendered frame and saves with timestamp.
"""
import os
import time
import cv2
import numpy as np
from utils.logger import log
from config.settings import SCREENSHOTS_DIR, SCREENSHOT_PREFIX


class ScreenshotManager:
    """Handles countdown + save of screenshots."""

    def __init__(self, audio=None):
        self._audio          = audio
        self._countdown      = 0.0   # seconds remaining
        self._active         = False
        self._capture_done   = False
        self._total          = 3.0

    def start(self):
        if not self._active:
            self._active       = True
            self._countdown    = self._total
            self._capture_done = False
            if self._audio:
                self._audio.play("scanning")

    def update(self, dt: float) -> bool:
        """Returns True on the frame the capture fires."""
        if not self._active:
            return False
        self._countdown -= dt
        if self._countdown <= 0 and not self._capture_done:
            self._capture_done = True
            self._active       = False
            return True
        return False

    def render_countdown(self, bgr: np.ndarray, t: float) -> np.ndarray:
        """Draw countdown overlay."""
        if not self._active or self._countdown <= 0:
            return bgr
        h, w = bgr.shape[:2]
        n    = int(math.ceil(self._countdown))
        # White flash ring
        frac = 1.0 - (self._countdown % 1.0)
        r    = int(60 + frac * 80)
        alpha_ring = 1.0 - frac
        overlay = bgr.copy()
        cv2.circle(overlay, (w // 2, h // 2), r, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha_ring * 0.6, bgr, 1 - alpha_ring * 0.6, 0, bgr)

        # Countdown number
        font  = cv2.FONT_HERSHEY_DUPLEX
        text  = str(n)
        scale = 4.0
        thickness = 6
        sz, _ = cv2.getTextSize(text, font, scale, thickness)
        tx = (w - sz[0]) // 2
        ty = (h + sz[1]) // 2
        cv2.putText(bgr, text, (tx + 2, ty + 2), font, scale, (0, 80, 200), thickness, cv2.LINE_AA)
        cv2.putText(bgr, text, (tx, ty), font, scale, (0, 200, 255), thickness, cv2.LINE_AA)
        return bgr

    @staticmethod
    def capture_screen(ctx, width, height):
        """Read final composited OpenGL framebuffer, not raw camera video."""
        raw = ctx.screen.read(components=3, alignment=1)
        rgb = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        return cv2.cvtColor(np.flipud(rgb).copy(), cv2.COLOR_RGB2BGR)

    def save(self, bgr: np.ndarray, audio=None) -> str:
        """Save frame to screenshots folder."""
        ts   = time.strftime("%Y%m%d_%H%M%S")
        name = f"{SCREENSHOT_PREFIX}_{ts}.png"
        path = os.path.join(SCREENSHOTS_DIR, name)
        if not cv2.imwrite(path, bgr):
            raise OSError(f"Could not save screenshot: {path}")
        log.info(f"Screenshot saved: {path}")
        if audio:
            audio.play("capture")
        return path

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def countdown(self) -> float:
        return self._countdown


import math   # needed by render_countdown

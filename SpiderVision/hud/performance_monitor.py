"""
SpiderVision — Performance Monitor HUD
Displays FPS, CPU, GPU, RAM, camera latency, threads, tracking confidence.
Rendered on the camera texture (OpenCV text).
"""
import time
import psutil
import threading
import numpy as np
import cv2
from config.settings import PERF_UPDATE_INTERVAL, PERF_X, PERF_Y
from utils.logger import log

try:
    import GPUtil
    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False
    log.warning("GPUtil not available — GPU monitoring disabled")


class PerformanceMonitor:
    """
    Polls system metrics in a background thread.
    Renders HUD text onto BGR frame in top-left corner.
    """

    def __init__(self):
        self._cpu_pct   = 0.0
        self._gpu_pct   = 0.0
        self._ram_mb    = 0.0
        self._thread_n  = 0
        self._last_poll = 0.0
        self._lock      = threading.Lock()
        self._alive     = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="PerfMonitor"
        )
        self._poll_thread.start()

        # Tracking confidence (set externally)
        self.face_confidence  = 0.0
        self.hand_confidence  = 0.0
        self.camera_latency   = 0.0

    def _poll_loop(self):
        while self._alive:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.Process().memory_info().rss / 1024 / 1024
            gpu = 0.0
            if _HAS_GPU:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0].load * 100
                except Exception:
                    pass
            nthreads = threading.active_count()
            with self._lock:
                self._cpu_pct   = cpu
                self._gpu_pct   = gpu
                self._ram_mb    = ram
                self._thread_n  = nthreads
            time.sleep(PERF_UPDATE_INTERVAL)

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self, bgr: np.ndarray, fps: float, camera_latency: float) -> np.ndarray:
        with self._lock:
            cpu = self._cpu_pct
            gpu = self._gpu_pct
            ram = self._ram_mb
            nth = self._thread_n

        lines = [
            (f"FPS  {fps:5.1f}", self._color_fps(fps)),
            (f"CPU  {cpu:5.1f}%", self._color_pct(cpu)),
            (f"GPU  {gpu:5.1f}%", self._color_pct(gpu)),
            (f"RAM  {ram:5.0f}MB", (160, 220, 255)),
            (f"CAM  {camera_latency:4.1f}ms", (160, 220, 255)),
            (f"THR  {nth:3d}", (160, 220, 255)),
            (f"FCE  {self.face_confidence:.2f}", self._color_conf(self.face_confidence)),
            (f"HND  {self.hand_confidence:.2f}", self._color_conf(self.hand_confidence)),
        ]

        font  = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.38
        pad   = 18
        bx    = PERF_X
        by    = PERF_Y + 14

        # Background panel
        panel_w = 120
        panel_h = len(lines) * pad + 6
        overlay = bgr.copy()
        cv2.rectangle(overlay, (bx - 4, by - 14), (bx + panel_w, by + panel_h),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, bgr, 0.5, 0, bgr)

        # Panel border
        cv2.rectangle(bgr, (bx - 4, by - 14), (bx + panel_w, by + panel_h),
                      (0, 120, 200), 1, cv2.LINE_AA)

        # Title
        cv2.putText(bgr, "SPIDER VISION", (bx, by - 2), font, 0.32,
                    (0, 200, 255), 1, cv2.LINE_AA)

        for i, (txt, col) in enumerate(lines):
            y = by + 6 + i * pad
            # Glow (shadow) pass
            cv2.putText(bgr, txt, (bx + 1, y + 1), font, scale,
                        (0, 40, 80), 1, cv2.LINE_AA)
            # Main text
            cv2.putText(bgr, txt, (bx, y), font, scale, col, 1, cv2.LINE_AA)

        return bgr

    @staticmethod
    def _color_fps(fps: float) -> tuple:
        if fps >= 55: return (0, 255, 120)
        if fps >= 30: return (0, 220, 255)
        return (0, 80, 255)

    @staticmethod
    def _color_pct(pct: float) -> tuple:
        if pct < 50: return (0, 255, 120)
        if pct < 75: return (0, 200, 255)
        return (0, 80, 255)

    @staticmethod
    def _color_conf(c: float) -> tuple:
        if c > 0.8: return (0, 255, 120)
        if c > 0.5: return (0, 200, 255)
        return (0, 80, 255)

    def stop(self):
        self._alive = False

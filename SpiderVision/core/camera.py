"""
SpiderVision — Threaded Camera Stream
Double-buffered OpenCV capture running in a background thread.
"""
import cv2
import time
import threading
import numpy as np
from utils.logger import log
from config.settings import CAMERA_SOURCE, CAMERA_W, CAMERA_H, CAMERA_FPS


class CameraStream:
    """
    Captures frames from camera in a dedicated thread.
    Main thread calls read() to get the latest frame without blocking.
    """

    def __init__(self, source=None):
        src = source if source is not None else CAMERA_SOURCE
        self._cap = cv2.VideoCapture(src)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
        self._cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        self._frame: np.ndarray | None = None
        self._lock  = threading.Lock()
        self._alive = True
        self._ready = threading.Event()
        self.latency_ms: float = 0.0
        self.connected: bool   = False

        if not self._cap.isOpened():
            log.error(f"Camera source {src!r} could not be opened")
        else:
            self.connected = True
            log.info(f"Camera opened: {src} @ {CAMERA_W}×{CAMERA_H}")

        self._thread = threading.Thread(target=self._loop, daemon=True, name="CameraCapture")
        self._thread.start()

    # ── Capture Loop ──────────────────────────────────────────────────────────

    def _loop(self):
        while self._alive:
            if not self._cap.isOpened():
                time.sleep(0.1)
                continue
            t0 = time.perf_counter()
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            # Mirror horizontally (webcam convention)
            frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame  = frame
                self.latency_ms = (time.perf_counter() - t0) * 1000.0
            self._ready.set()
            self.connected = True

    # ── Public API ────────────────────────────────────────────────────────────

    def read(self) -> np.ndarray | None:
        """Return a copy of the most recent frame (BGR, uint8), or None."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def read_rgb(self) -> np.ndarray | None:
        """Return frame in RGB format (for MediaPipe)."""
        bgr = self.read()
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def wait_for_first_frame(self, timeout: float = 5.0) -> bool:
        return self._ready.wait(timeout)

    @property
    def frame_size(self) -> tuple[int, int]:
        return (CAMERA_W, CAMERA_H)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def stop(self):
        self._alive = False
        self._thread.join(timeout=2.0)
        self._cap.release()
        log.info("Camera stream stopped")

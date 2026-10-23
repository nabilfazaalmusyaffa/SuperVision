"""Thread-safe visual session latch; not authentication or liveness detection."""
import threading


class FaceSession:
    """Keep controls unlocked across temporary owner absence."""

    def __init__(self):
        self._lock = threading.Lock()
        self._unlocked = False

    @property
    def unlocked(self) -> bool:
        with self._lock:
            return self._unlocked

    def observe(self, raw_status: str) -> bool:
        """Consume a raw identity status and return the session's unlock state."""
        with self._lock:
            if raw_status == "AUTHORIZED":
                self._unlocked = True
            elif raw_status in ("DANGER", "DANGER_OTHER", "NOT_ENROLLED", "ENROLLING"):
                self._unlocked = False
            return self._unlocked

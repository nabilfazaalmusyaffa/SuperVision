"""Opt-in, local-only face *approximation* using OpenCV LBPH.

Not an authentication system: photographs and similar-looking people may pass.
Never use its result to grant access or make safety decisions.
"""
from pathlib import Path
import os
import threading
import time

import cv2


class FaceIdentity:
    def __init__(self, path, samples_required=20, sample_interval=0.15, threshold=50.0):
        self.path = Path(path)
        self.samples_required = samples_required
        self.sample_interval = sample_interval
        self.threshold = threshold
        self._lock = threading.RLock()
        self._model = cv2.face.LBPHFaceRecognizer_create()
        self.enrolled = False
        self._samples = []
        self._last_sample = 0.0
        self._match_frames = 0
        self._unknown_frames = 0
        self.status = 'NOT_ENROLLED'
        if self.path.exists():
            try:
                self._model.read(str(self.path))
                self.enrolled = True
                self.status = 'NO_FACE'
                self._calibrate_threshold()
            except (cv2.error, OSError):
                self.status = 'NOT_ENROLLED'

    @property
    def sample_count(self):
        with self._lock:
            return len(self._samples)

    def _calibrate_threshold(self):
        """Set a tolerant visual-demo threshold from enrolled face variation.

        OpenCV's LBPH distance is not a probability. A fixed value of 50 was
        below even neighboring frames from the owner's enrollment (~76-79).
        Capped at 90 to avoid accepting arbitrary high-distance frames.
        """
        histograms = self._model.getHistograms()
        if len(histograms) < 2:
            return
        nearest = [min(cv2.compareHist(histograms[i], histograms[j],
                                       cv2.HISTCMP_CHISQR_ALT)
                       for j in range(len(histograms)) if j != i)
                   for i in range(len(histograms))]
        self.threshold = min(90.0, max(self.threshold, max(nearest) + 8.0))

    def start_enrollment(self):
        with self._lock:
            self._samples = []
            self._last_sample = 0.0
            self.status = 'ENROLLING'
            self._match_frames = self._unknown_frames = 0

    def delete_enrollment(self):
        with self._lock:
            self._samples = []
            self.enrolled = False
            self.status = 'NOT_ENROLLED'
            self._match_frames = self._unknown_frames = 0
            self._model = cv2.face.LBPHFaceRecognizer_create()
            self.path.unlink(missing_ok=True)

    @staticmethod
    def _crop(frame, box):
        if box is None:
            return None
        x, y, w, h = (int(v) for v in box)
        height, width = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 - x < 45 or y2 - y < 45:
            return None
        roi = frame[y:y2, x:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        return cv2.equalizeHist(cv2.resize(gray, (100, 100)))

    def process(self, frame, box):
        with self._lock:
            crop = self._crop(frame, box)
            if crop is None:
                self._match_frames = self._unknown_frames = 0
                self.status = 'ENROLLING' if self.status == 'ENROLLING' else 'NO_FACE'
                return self.status

            if self.status == 'ENROLLING':
                now = time.monotonic()
                if now - self._last_sample >= self.sample_interval:
                    self._samples.append(crop)
                    self._last_sample = now
                if len(self._samples) >= self.samples_required:
                    import numpy as np
                    self._model.train(self._samples, np.zeros(len(self._samples), dtype=np.int32))
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    fd = os.open(self.path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
                    os.close(fd)
                    os.chmod(self.path, 0o600)
                    self._model.write(str(self.path))
                    self._calibrate_threshold()
                    self._samples = []
                    self.enrolled = True
                    self.status = 'SCANNING'
                return self.status

            if not self.enrolled:
                self.status = 'NOT_ENROLLED'
                return self.status

            _, distance = self._model.predict(crop)
            if distance <= self.threshold:
                self._match_frames += 1
                self._unknown_frames = 0
                self.status = 'AUTHORIZED' if self._match_frames >= 3 else 'SCANNING'
            else:
                self._unknown_frames += 1
                self._match_frames = 0
                self.status = 'DANGER' if self._unknown_frames >= 3 else 'SCANNING'
            return self.status

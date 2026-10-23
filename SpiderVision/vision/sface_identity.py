"""Opt-in visual face matching using OpenCV Zoo YuNet + SFace embeddings.

This is a demo interaction, not secure authentication or liveness detection.
Only local face embeddings are saved; camera frames are never saved here.
"""
import os
from pathlib import Path
import tempfile
import threading
import time

import cv2
import numpy as np

from config.settings import BASE_DIR

MODELS_DIR = Path(BASE_DIR) / "assets" / "models"
DETECTOR_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"


class SFaceIdentity:
    """Detect, enroll, and compare faces with cosine similarity."""

    def __init__(self, path, extractor=None, samples_required=8,
                 sample_interval=0.15, threshold=0.363):
        self.path = Path(path)
        self.samples_required = samples_required
        self.sample_interval = sample_interval
        self.threshold = threshold  # OpenCV Zoo's benchmark threshold, not a probability
        self._lock = threading.RLock()
        self._samples = []
        self._templates = None
        self._last_sample = 0.0
        self._match_frames = 0
        self._unknown_frames = 0
        self._mixed_frames = 0
        self.status = "NOT_ENROLLED"
        self.last_score = None
        if extractor is None:
            if not (DETECTOR_MODEL.is_file() and RECOGNIZER_MODEL.is_file()):
                raise FileNotFoundError("YuNet/SFace model files missing in assets/models")
            self._detector = cv2.FaceDetectorYN_create(str(DETECTOR_MODEL), "", (320, 320), score_threshold=0.8)
            self._recognizer = cv2.FaceRecognizerSF_create(str(RECOGNIZER_MODEL), "")
            self._extractor = self._extract_camera_feature
        else:
            self._extractor = extractor
        if self.path.exists():
            try:
                with np.load(self.path, allow_pickle=False) as data:
                    features = data["features"]
                if (features.ndim != 2 or features.shape[0] < 1 or
                        features.shape[0] > 32 or features.shape[1] < 3 or
                        not np.all(np.isfinite(features))):
                    raise ValueError("Invalid face enrollment data")
                self._templates = self._normalize_rows(features)
                self.status = "NO_FACE"
            except (OSError, ValueError, KeyError, TypeError):
                self._templates = None

    @staticmethod
    def _normalize_rows(features):
        features = np.asarray(features, dtype=np.float32)
        norms = np.linalg.norm(features, axis=-1, keepdims=True)
        return features / np.maximum(norms, 1e-8)

    def _extract_camera_feature(self, frame):
        if frame is None or frame.ndim != 3:
            return None
        height, width = frame.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        # Evaluate every YuNet detection; the best face alone can hide a stranger.
        features = []
        for face in faces:
            aligned = self._recognizer.alignCrop(frame, face)
            features.append(self._recognizer.feature(aligned).reshape(-1).copy())
        return np.stack(features)

    @property
    def enrolled(self):
        with self._lock:
            return self._templates is not None

    @property
    def sample_count(self):
        with self._lock:
            return len(self._samples)

    def start_enrollment(self):
        with self._lock:
            self._samples = []
            self._last_sample = 0.0
            self._match_frames = self._unknown_frames = self._mixed_frames = 0
            self.status = "ENROLLING"

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".face-", suffix=".npz", dir=self.path.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez_compressed(f, features=self._templates)
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, self.path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        # Previous LBPH enrollment is incompatible with SFace and no longer needed.
        self.path.with_suffix(".yml").unlink(missing_ok=True)

    def delete_enrollment(self):
        with self._lock:
            self._templates = None
            self._samples = []
            self._match_frames = self._unknown_frames = self._mixed_frames = 0
            self.status = "NOT_ENROLLED"
            self.path.unlink(missing_ok=True)
            self.path.with_suffix(".yml").unlink(missing_ok=True)

    def process(self, frame, box=None):
        """Return NO_FACE, ENROLLING, NOT_ENROLLED, SCANNING, AUTHORIZED,
        DANGER (stranger only), or DANGER_OTHER (owner and stranger).

        `box` is accepted for compatibility with the main tracking loop; YuNet
        supplies its own five landmark points for correct SFace alignment.
        """
        with self._lock:
            feature = self._extractor(frame)
            if feature is None:
                self._match_frames = self._unknown_frames = self._mixed_frames = 0
                self.last_score = None
                self.status = "ENROLLING" if self.status == "ENROLLING" else "NO_FACE"
                return self.status
            features = np.asarray(feature, dtype=np.float32)
            if features.ndim == 1:
                features = features[np.newaxis, :]
            if features.ndim != 2 or not len(features):
                self._match_frames = self._unknown_frames = self._mixed_frames = 0
                self.last_score = None
                self.status = "ENROLLING" if self.status == "ENROLLING" else "NO_FACE"
                return self.status
            norms = np.linalg.norm(features, axis=1)
            valid = np.isfinite(norms) & (norms >= 1e-8)
            if not np.any(valid):
                self._match_frames = self._unknown_frames = self._mixed_frames = 0
                self.last_score = None
                self.status = "ENROLLING" if self.status == "ENROLLING" else "NO_FACE"
                return self.status
            features = features[valid] / norms[valid, np.newaxis]
            if self.status == "ENROLLING":
                if len(features) != 1 or not np.all(valid):
                    return self.status
                now = time.monotonic()
                if now - self._last_sample >= self.sample_interval:
                    self._samples.append(features[0].copy())
                    self._last_sample = now
                if len(self._samples) >= self.samples_required:
                    self._templates = np.stack(self._samples)
                    self._save()
                    self._samples = []
                    self.status = "SCANNING"
                return self.status
            if self._templates is None:
                self.status = "NOT_ENROLLED"
                return self.status
            if features.shape[1] != self._templates.shape[1]:
                self.status = "NOT_ENROLLED"
                return self.status
            scores = np.max(self._templates @ features.T, axis=0)
            self.last_score = float(np.max(scores))
            owner_present = np.any(scores >= self.threshold)
            stranger_present = np.any(scores < self.threshold) or not np.all(valid)
            if owner_present and stranger_present:
                self._mixed_frames += 1
                self._match_frames = self._unknown_frames = 0
                self.status = "DANGER_OTHER" if self._mixed_frames >= 3 else "SCANNING"
            elif owner_present:
                self._match_frames += 1
                self._unknown_frames = self._mixed_frames = 0
                self.status = "AUTHORIZED" if self._match_frames >= 3 else "SCANNING"
            else:
                self._unknown_frames += 1
                self._match_frames = self._mixed_frames = 0
                self.status = "DANGER" if self._unknown_frames >= 3 else "SCANNING"
            return self.status

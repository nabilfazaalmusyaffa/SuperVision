"""
SpiderVision — Face Tracker
MediaPipe Face Detection + Face Mesh with 3D landmark smoothing.
"""
import cv2
import time
import threading
import numpy as np
import glm
import mediapipe as mp

from utils.logger import log
from utils.math_utils import ema
from config.settings import (
    WINDOW_W, WINDOW_H, CAMERA_W, CAMERA_H,
    FACE_TRACK_SMOOTH, FACE_CONFIDENCE_MIN
)


class FaceData:
    """Holds processed face detection + mesh data."""
    __slots__ = [
        "detected", "box_x", "box_y", "box_w", "box_h",
        "landmarks", "confidence", "depth_estimate",
        "center_x", "center_y", "scan_progress", "identity_status", "enroll_progress",
        "session_unlocked"
    ]

    def __init__(self):
        self.detected       = False
        self.box_x          = 0.0   # normalised 0..1
        self.box_y          = 0.0
        self.box_w          = 0.0
        self.box_h          = 0.0
        self.landmarks      = []    # list of (x, y, z) normalised
        self.confidence     = 0.0
        self.depth_estimate = 1.0   # 1.0 = far, 0.0 = very close
        self.center_x       = 0.5
        self.center_y       = 0.5
        self.scan_progress  = 0.0   # 0..1 scanning animation
        self.identity_status = "NOT_ENROLLED"
        self.enroll_progress = ""
        self.session_unlocked = False


class FaceTracker:
    """
    Tracks faces using MediaPipe FaceMesh (468 landmarks + iris).
    Provides smoothed FaceData for HUD consumption.
    """

    def __init__(self):
        self._data      = FaceData()
        self._lock      = threading.Lock()
        self._smooth_lm = None   # smoothed landmarks
        self._scan_t    = 0.0   # scan timer
        self._init_mediapipe()
        log.info("FaceTracker ready")

    def _init_mediapipe(self):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,   # includes iris
            min_detection_confidence=FACE_CONFIDENCE_MIN,
            min_tracking_confidence=0.5,
        )

        # Simplified face detection for bounding box
        self._mp_face_det = mp.solutions.face_detection
        self._face_det = self._mp_face_det.FaceDetection(
            model_selection=0,  # webcam distance, not long-range camera
            min_detection_confidence=FACE_CONFIDENCE_MIN
        )

    # ── Processing ────────────────────────────────────────────────────────────

    def process(self, rgb_frame: np.ndarray, dt: float) -> FaceData:
        data = FaceData()

        # Face detection for bounding box + confidence
        det_result = self._face_det.process(rgb_frame)
        if det_result.detections:
            det  = det_result.detections[0]
            conf = det.score[0] if det.score else 0.0
            bb   = det.location_data.relative_bounding_box
            data.box_x      = max(0.0, bb.xmin)
            data.box_y      = max(0.0, bb.ymin)
            data.box_w      = min(1.0, bb.width)
            data.box_h      = min(1.0, bb.height)
            data.confidence = conf
            data.center_x   = data.box_x + data.box_w * 0.5
            data.center_y   = data.box_y + data.box_h * 0.5
            # Depth estimate from face width (larger = closer)
            data.depth_estimate = max(0.05, 1.0 - data.box_w * 2.0)
            data.detected   = True

        # FaceMesh for landmarks
        mesh_result = self._face_mesh.process(rgb_frame)
        if mesh_result.multi_face_landmarks:
            raw_lm = [
                (lm.x, lm.y, lm.z)
                for lm in mesh_result.multi_face_landmarks[0].landmark
            ]
            # EMA smooth landmarks
            if self._smooth_lm is not None and len(self._smooth_lm) == len(raw_lm):
                smoothed = [
                    (
                        ema(self._smooth_lm[i][0], raw_lm[i][0], FACE_TRACK_SMOOTH),
                        ema(self._smooth_lm[i][1], raw_lm[i][1], FACE_TRACK_SMOOTH),
                        ema(self._smooth_lm[i][2], raw_lm[i][2], FACE_TRACK_SMOOTH),
                    )
                    for i in range(len(raw_lm))
                ]
            else:
                smoothed = raw_lm
            self._smooth_lm = smoothed
            data.landmarks  = smoothed
            data.detected   = True
            if data.box_w == 0.0:
                xs = [p[0] for p in smoothed]
                ys = [p[1] for p in smoothed]
                data.box_x = max(0.0, min(xs))
                data.box_y = max(0.0, min(ys))
                data.box_w = min(1.0 - data.box_x, max(xs) - data.box_x)
                data.box_h = min(1.0 - data.box_y, max(ys) - data.box_y)
                data.confidence = FACE_CONFIDENCE_MIN

        # Scan progress animation
        if data.detected:
            self._scan_t = (self._scan_t + dt * 1.8) % 1.0
        else:
            self._scan_t = max(0.0, self._scan_t - dt * 2.0)
        data.scan_progress = self._scan_t

        with self._lock:
            self._data = data
        return data

    def get_data(self) -> FaceData:
        with self._lock:
            return self._data

    # ── Key landmark indices ───────────────────────────────────────────────────

    # Subset for rendering (key facial feature points)
    KEY_INDICES = [
        # Face oval
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
        # Eyes
        33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
        # Nose
        1, 2, 98, 327,
        # Mouth
        61, 291, 0, 17, 37, 267,
    ]

    def close(self):
        self._face_mesh.close()
        self._face_det.close()
        log.info("FaceTracker closed")

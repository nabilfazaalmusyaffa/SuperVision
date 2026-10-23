"""
SpiderVision — Spider Mask Detector
Heuristic detector for spider-themed face coverage.
Uses color analysis + coverage ratio in the face bounding box.
No external ML model required.
"""
import cv2
import numpy as np
from utils.logger import log
from config.settings import MASK_RED_RATIO, MASK_BLACK_RATIO


class MaskDetector:
    """
    Detects whether the user is wearing a spider-themed mask
    (predominantly red/dark with circular coverage of the face).
    Uses HSV color analysis on the face region.
    """

    def __init__(self):
        self._confidence   = 0.0
        self._smooth_conf  = 0.0
        self._detected     = False
        log.info("MaskDetector ready (heuristic mode)")

    def detect(self, bgr: np.ndarray, face_data) -> tuple[bool, float]:
        """
        Returns (is_detected, confidence) 0..1.
        Analyzes face bounding box region for red/black dominance.
        """
        if not face_data.detected:
            self._smooth_conf *= 0.9
            self._detected = self._smooth_conf > 0.5
            return self._detected, self._smooth_conf

        h, w = bgr.shape[:2]
        bx  = int(face_data.box_x * w)
        by  = int(face_data.box_y * h)
        bw  = int(face_data.box_w * w)
        bh  = int(face_data.box_h * h)

        # Clamp to image
        bx  = max(0, bx)
        by  = max(0, by)
        x2  = min(w, bx + bw)
        y2  = min(h, by + bh)
        if x2 - bx < 10 or y2 - by < 10:
            return False, 0.0

        roi = bgr[by:y2, bx:x2]
        if roi.size == 0:
            return False, 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total_px = roi.shape[0] * roi.shape[1]

        # Red detection (two hue ranges in HSV)
        red_lo1 = np.array([0,   100, 50])
        red_hi1 = np.array([10,  255, 255])
        red_lo2 = np.array([165, 100, 50])
        red_hi2 = np.array([180, 255, 255])
        mask_r1  = cv2.inRange(hsv, red_lo1, red_hi1)
        mask_r2  = cv2.inRange(hsv, red_lo2, red_hi2)
        red_px   = cv2.countNonZero(cv2.bitwise_or(mask_r1, mask_r2))

        # Dark/black detection
        black_lo = np.array([0, 0,   0])
        black_hi = np.array([180, 50, 60])
        mask_blk = cv2.inRange(hsv, black_lo, black_hi)
        blk_px   = cv2.countNonZero(mask_blk)

        red_ratio = red_px / total_px
        blk_ratio = blk_px / total_px

        # Skin color detection (to know what's NOT mask)
        skin_lo = np.array([0,  20, 70])
        skin_hi = np.array([20, 150, 255])
        mask_sk  = cv2.inRange(hsv, skin_lo, skin_hi)
        skin_px  = cv2.countNonZero(mask_sk)
        skin_ratio = skin_px / total_px

        # Score: high red + black + low skin = mask
        raw_conf = min(1.0, max(0.0,
            red_ratio / max(MASK_RED_RATIO, 0.001) * 0.5
            + blk_ratio / max(MASK_BLACK_RATIO, 0.001) * 0.3
            - skin_ratio * 1.0
        ))

        # Smooth
        self._smooth_conf += 0.15 * (raw_conf - self._smooth_conf)
        self._detected     = self._smooth_conf > 0.5

        return self._detected, self._smooth_conf

    @property
    def is_active(self) -> bool:
        return self._detected

    @property
    def confidence(self) -> float:
        return self._smooth_conf

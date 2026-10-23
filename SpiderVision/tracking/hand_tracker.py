"""
SpiderVision — Hand Tracker
MediaPipe Hands inference with smoothing and gesture detection.
Runs in a background thread; main thread reads latest state.
"""
import cv2
import time
import threading
import numpy as np
import glm
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from utils.logger import log
from utils.math_utils import ema
from config.settings import (
    WINDOW_W, WINDOW_H,
    HAND_SMOOTH, HAND_MIN_CONFIDENCE, HAND_MIN_PRESENCE, MAX_HANDS,
    CAMERA_W, CAMERA_H
)
from tracking.gesture import (
    HandState, Gesture, detect_gesture, palm_center
)


class HandTracker:
    """
    Real-time hand tracking via MediaPipe Hands (VIDEO mode).
    Provides up to 2 HandState objects with smoothed positions.
    """

    def __init__(self):
        self._hands: list[HandState] = []
        self._lock = threading.Lock()
        self._ts   = 0           # monotonic timestamp for MediaPipe VIDEO mode
        self._init_mediapipe()
        log.info("HandTracker ready")

    def _init_mediapipe(self):
        """Use MediaPipe Hands solution (legacy API — broad compatibility)."""
        self._mp_hands = mp.solutions.hands
        self._hands_sol = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=HAND_MIN_CONFIDENCE,
            min_tracking_confidence=HAND_MIN_PRESENCE,
        )
        self._mp_draw = mp.solutions.drawing_utils
        # Smoothed landmark history per hand slot
        self._smooth: dict[str, list] = {}

    @staticmethod
    def screen_x(normalized_x, side):
        # CameraStream already mirrors the frame; handedness is not a transform.
        return normalized_x * WINDOW_W

    # ── Process ───────────────────────────────────────────────────────────────

    def process(self, rgb_frame: np.ndarray) -> list[HandState]:
        """
        Process an RGB frame and return list of HandState (0, 1, or 2).
        Call this from the tracking thread.
        """
        result = self._hands_sol.process(rgb_frame)
        states: list[HandState] = []

        if result.multi_hand_landmarks and result.multi_handedness:
            for idx, (hand_lm, handedness) in enumerate(
                zip(result.multi_hand_landmarks, result.multi_handedness)
            ):
                side = handedness.classification[0].label   # "Left" or "Right"
                conf = handedness.classification[0].score

                # Raw landmarks → list of (x, y, z) normalised 0..1
                raw_lm = [(lm.x, lm.y, lm.z) for lm in hand_lm.landmark]

                # Smooth landmarks using EMA
                if side in self._smooth:
                    smoothed = []
                    for j, (rx, ry, rz) in enumerate(raw_lm):
                        px, py, pz = self._smooth[side][j]
                        smoothed.append((
                            ema(px, rx, HAND_SMOOTH),
                            ema(py, ry, HAND_SMOOTH),
                            ema(pz, rz, HAND_SMOOTH),
                        ))
                else:
                    smoothed = raw_lm

                self._smooth[side] = smoothed

                # Classify gesture
                gest = detect_gesture(smoothed)

                # Palm position in window pixel coords
                pcx, pcy = palm_center(smoothed)
                palm_px   = glm.vec2(
                    self.screen_x(pcx, side),
                    pcy * WINDOW_H
                )
                palm_3d = glm.vec3(smoothed[0][0], smoothed[0][1], smoothed[0][2])

                state = HandState(
                    handedness = side,
                    gesture    = gest,
                    landmarks  = smoothed,
                    aim_landmarks = raw_lm,
                    palm_pos   = palm_px,
                    palm_pos_3d= palm_3d,
                    confidence = conf,
                )
                states.append(state)

            # Reset smoothing for missing hands
            self._smooth = {s.handedness: self._smooth[s.handedness] for s in states}
        else:
            self._smooth = {}

        with self._lock:
            self._hands = states
        return states

    # ── Public read ───────────────────────────────────────────────────────────

    def get_hands(self) -> list[HandState]:
        with self._lock:
            return list(self._hands)

    def get_by_side(self, side: str) -> HandState | None:
        """Returns first hand matching 'Left' or 'Right', or None."""
        with self._lock:
            for h in self._hands:
                if h.handedness == side:
                    return h
        return None

    # ── Drawing helper ────────────────────────────────────────────────────────

    def draw_skeleton(self, bgr: np.ndarray, hands: list[HandState]) -> np.ndarray:
        """
        Draw hand skeleton on a BGR numpy frame (for debug / camera texture).
        Returns modified copy.
        """
        out = bgr.copy()
        h, w = out.shape[:2]
        for hand in hands:
            lm = hand.landmarks
            # Draw connections
            connections = self._mp_hands.HAND_CONNECTIONS
            for a, b in connections:
                ax, ay = int(lm[a][0] * w), int(lm[a][1] * h)
                bx, by = int(lm[b][0] * w), int(lm[b][1] * h)
                cv2.line(out, (ax, ay), (bx, by), (0, 180, 255), 1, cv2.LINE_AA)
            # Draw joints
            for i, (x, y, _) in enumerate(lm):
                px, py = int(x * w), int(y * h)
                color = (255, 60, 60) if i in (4, 8, 12, 16, 20) else (0, 200, 255)
                cv2.circle(out, (px, py), 4 if i in (4, 8) else 3, color, -1, cv2.LINE_AA)
        return out

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        self._hands_sol.close()
        log.info("HandTracker closed")

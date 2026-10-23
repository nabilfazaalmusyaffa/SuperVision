"""
SpiderVision — Tracking: Gesture Recognition
Defines all gesture types, detection logic, and state management.
"""
import math
import time
import glm
from dataclasses import dataclass, field
from config.settings import (
    FINGER_RATIO, THUMB_RATIO, GESTURE_HOLD_SEC, WINDOW_W, WINDOW_H
)


# ─── Gesture Enum ─────────────────────────────────────────────────────────────

class Gesture:
    NONE         = "NONE"
    FIST         = "FIST"         # All closed → close menu
    PALM_OPEN    = "PALM_OPEN"    # All open  → open menu
    POINT        = "POINT"        # Index only
    WEB_SHOT     = "WEB_SHOT"     # Index + Pinky + Thumb (Spider-Man)
    L_SHAPE      = "L_SHAPE"      # Thumb + Index (photo frame)
    PEACE        = "PEACE"        # Index + Middle
    THUMBS_UP    = "THUMBS_UP"    # Thumb only
    PINCH        = "PINCH"        # Thumb + Index close


@dataclass
class HandState:
    handedness: str      = "Right"   # "Left" or "Right"
    gesture: str         = Gesture.NONE
    landmarks: list      = field(default_factory=list)   # list of (x,y,z) tuples
    aim_landmarks: list  = field(default_factory=list)   # current tips; skeleton remains smoothed
    palm_pos: glm.vec2   = field(default_factory=lambda: glm.vec2(0,0))
    palm_pos_3d: glm.vec3= field(default_factory=lambda: glm.vec3(0,0,0))
    confidence: float    = 0.0
    # Gesture hold timer
    _hold_start: float   = field(default=0.0, repr=False)
    _held_gesture: str   = field(default=Gesture.NONE, repr=False)

    def update_gesture(self, new_gest: str) -> bool:
        """
        Returns True when the gesture has been held long enough to trigger.
        """
        now = time.time()
        if new_gest != self._held_gesture:
            self._held_gesture = new_gest
            self._hold_start   = now
        if new_gest != Gesture.NONE:
            return (now - self._hold_start) >= GESTURE_HOLD_SEC
        return False


class GestureLatch:
    """Trigger once per held gesture, resetting on release/disappearance."""
    def __init__(self, hold_seconds=GESTURE_HOLD_SEC):
        self.hold_seconds = hold_seconds
        self._states = {}

    def update(self, side, gesture, now):
        old, since, fired = self._states.get(side, (Gesture.NONE, now, False))
        if old != gesture:
            old, since, fired = gesture, now, False
        if gesture != Gesture.NONE and not fired and now - since >= self.hold_seconds:
            fired = True
            self._states[side] = (gesture, since, fired)
            return True
        self._states[side] = (gesture, since, fired)
        return False

    def forget_missing(self, present):
        for side in tuple(self._states):
            if side not in present:
                del self._states[side]


def web_aim(landmarks):
    """Aim along index finger MCP→tip in already-mirrored camera coordinates."""
    if len(landmarks) < 9:
        return glm.vec2(0.0, -1.0)
    dx = landmarks[8][0] - landmarks[5][0]
    dy = landmarks[8][1] - landmarks[5][1]
    size = math.hypot(dx, dy)
    return glm.vec2(dx / size, dy / size) if size > 0.025 else glm.vec2(0.0, -1.0)


def web_origin(landmarks):
    """Spider web emerges from the index fingertip, not the palm center."""
    if len(landmarks) < 9:
        return glm.vec2(WINDOW_W * 0.5, WINDOW_H * 0.5)
    return glm.vec2(landmarks[8][0] * WINDOW_W, landmarks[8][1] * WINDOW_H)

# ─── Finger Open Detection ────────────────────────────────────────────────────

def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def fingers_open(landmarks: list) -> list[bool]:
    """
    Returns [thumb, index, middle, ring, pinky] booleans.
    landmarks: list of 21 (x,y,z) normalized tuples from MediaPipe.
    """
    if len(landmarks) < 21:
        return [False] * 5

    wrist  = landmarks[0]
    ref    = landmarks[17]   # pinky MCP (wrist width reference)

    # Thumb: compare tip(4) distance from wrist vs joint(2)
    thumb = _dist(landmarks[4], ref) > _dist(landmarks[2], ref) * THUMB_RATIO

    # An extended finger can point toward the wrist when aiming downward.
    # Measure extension from its own MCP and reject joints folded back.
    joints = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
    others = []
    for mcp, pip, tip in joints:
        base = _dist(landmarks[pip], landmarks[mcp])
        forward = ((landmarks[pip][0] - landmarks[mcp][0]) *
                   (landmarks[tip][0] - landmarks[pip][0]) +
                   (landmarks[pip][1] - landmarks[mcp][1]) *
                   (landmarks[tip][1] - landmarks[pip][1]))
        if base < 0.001:  # incomplete landmarks in test/recovery paths
            others.append(_dist(landmarks[tip], wrist) >
                          _dist(landmarks[pip], wrist) * FINGER_RATIO)
        else:
            others.append(_dist(landmarks[tip], landmarks[mcp]) > base * FINGER_RATIO
                          and forward >= 0)

    return [thumb] + others


def is_retraction_pose(landmarks: list) -> bool:
    """Mostly folded fingers; allow an extended thumb or unreliable pinky.

    The index must fold, so a pointing hand or web-shot pose cannot pull a web.
    Check current (unsmoothed) landmarks for timely response.
    """
    if len(landmarks) < 21:
        return False
    _, index, middle, ring, pinky = fingers_open(landmarks)
    return not index and sum(not finger for finger in (middle, ring, pinky)) >= 2


# ─── Gesture Classifiers ──────────────────────────────────────────────────────

def classify_gesture(fingers: list[bool]) -> str:
    t, i, m, r, p = fingers   # thumb, index, middle, ring, pinky

    # Spider-Man web shot: thumb + index + pinky OUT, middle + ring IN
    if t and i and not m and not r and p:
        return Gesture.WEB_SHOT

    if not any(fingers):
        return Gesture.FIST

    if all(fingers):
        return Gesture.PALM_OPEN

    if t and i and not m and not r and not p:
        return Gesture.L_SHAPE

    if not t and i and m and not r and not p:
        return Gesture.PEACE

    if not t and i and not m and not r and not p:
        return Gesture.POINT

    if t and not i and not m and not r and not p:
        return Gesture.THUMBS_UP

    return Gesture.NONE


# ─── Pinch Detection ──────────────────────────────────────────────────────────

def is_pinching(landmarks: list, threshold: float = 0.05) -> bool:
    """Detect thumb-index pinch (normalised distance)."""
    if len(landmarks) < 9:
        return False
    return _dist(landmarks[4], landmarks[8]) < threshold


def detect_gesture(landmarks):
    """Classify pose and thumb-index pinch, excluding a closed fist."""
    fingers = fingers_open(landmarks)
    pose = classify_gesture(fingers)
    if (pose != Gesture.FIST and is_pinching(landmarks)
            and _dist(landmarks[8], landmarks[0]) > 0.06):
        return Gesture.PINCH
    return pose


# ─── Landmark helpers ─────────────────────────────────────────────────────────

def palm_center(landmarks: list) -> tuple[float, float]:
    """Average of wrist + 5 MCP joints → stable palm center."""
    idxs = [0, 1, 5, 9, 13, 17]
    xs = [landmarks[i][0] for i in idxs]
    ys = [landmarks[i][1] for i in idxs]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def index_tip(landmarks: list) -> tuple[float, float]:
    return landmarks[8][0], landmarks[8][1]


def middle_tip(landmarks: list) -> tuple[float, float]:
    return landmarks[12][0], landmarks[12][1]

"""
SpiderVision — Monocular Depth Estimator
Heuristic depth estimation from face size and hand Z-coordinates.
"""
import math
from config.settings import WINDOW_W, WINDOW_H


class DepthEstimator:
    """
    Provides approximate depth (Z) estimates for tracked objects.
    Uses face bounding box size (larger = closer) and MediaPipe 3D landmarks.
    """

    # Typical face width at 1m (~30cm face at 1m → ~24% of 1080p height)
    FACE_REFERENCE_SIZE = 0.24   # fraction of frame height

    def face_depth(self, face_data) -> float:
        """
        Returns estimated depth in arbitrary units.
        0.0 = very close (face fills frame), 3.0 = far.
        """
        if not face_data.detected:
            return 3.0
        size_ratio = face_data.box_h   # height fraction
        if size_ratio < 0.01:
            return 3.0
        depth = self.FACE_REFERENCE_SIZE / max(size_ratio, 0.01)
        return min(5.0, depth)

    def hand_depth(self, landmarks: list) -> float:
        """
        Returns Z depth from MediaPipe landmark Z coordinates.
        MediaPipe Z is relative to wrist, negative = toward camera.
        """
        if not landmarks:
            return 0.0
        wrist_z = landmarks[0][2]
        # MediaPipe Z is in normalized units ~[-0.3, 0.3]
        return max(0.0, 1.0 + wrist_z * 5.0)

    def parallax_offset(self, face_data, layer_z: float) -> tuple[float, float]:
        """
        Compute HUD parallax offset for a given depth layer.
        Returns (dx, dy) pixel offset from center.
        """
        if not face_data.detected:
            return (0.0, 0.0)
        # Face angle from center
        face_cx = face_data.center_x - 0.5   # -0.5 .. 0.5
        face_cy = face_data.center_y - 0.5
        parallax = 0.04 * layer_z
        dx = face_cx * WINDOW_W * parallax
        dy = face_cy * WINDOW_H * parallax
        return (dx, dy)

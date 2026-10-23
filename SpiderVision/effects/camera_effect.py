"""
SpiderVision — Camera Effect (Screen Shake, DOF, Flash)
"""
import math
import random
import time
import glm
from config.settings import SHAKE_DECAY, SHAKE_WEB_INTENSITY, SHAKE_CAPTURE_INTENSITY


class CameraEffect:
    """
    Manages camera-level effects:
    - Screen shake (dampened spring)
    - White flash (for screenshot)
    - Exposure pulse
    """

    def __init__(self):
        self._shake_offset  = glm.vec2(0.0, 0.0)
        self._shake_vel     = glm.vec2(0.0, 0.0)
        self._shake_target  = glm.vec2(0.0, 0.0)
        self._flash_alpha   = 0.0
        self._flash_duration= 0.0
        self._flash_start   = 0.0
        self._exposure_mul  = 1.0

    # ── Shake ─────────────────────────────────────────────────────────────────

    def shake(self, intensity: float = SHAKE_WEB_INTENSITY):
        """Add a random shake impulse."""
        angle = random.uniform(0, math.pi * 2)
        self._shake_vel = glm.vec2(
            math.cos(angle) * intensity * 80.0,
            math.sin(angle) * intensity * 80.0
        )

    def shake_web(self):
        self.shake(SHAKE_WEB_INTENSITY)

    def shake_capture(self):
        self.shake(SHAKE_CAPTURE_INTENSITY)

    # ── Flash ─────────────────────────────────────────────────────────────────

    def flash_white(self, duration: float = 0.35):
        self._flash_alpha    = 1.0
        self._flash_duration = duration
        self._flash_start    = time.time()

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, dt: float):
        # Spring-damper shake
        spring_k  = 80.0
        damping   = SHAKE_DECAY
        toward = -self._shake_offset
        self._shake_vel  = glm.vec2(
            self._shake_vel.x + toward.x * spring_k * dt,
            self._shake_vel.y + toward.y * spring_k * dt,
        )
        self._shake_vel  = glm.vec2(
            self._shake_vel.x * (1.0 - damping * dt),
            self._shake_vel.y * (1.0 - damping * dt),
        )
        self._shake_offset = glm.vec2(
            self._shake_offset.x + self._shake_vel.x * dt,
            self._shake_offset.y + self._shake_vel.y * dt,
        )
        # Clamp
        max_shake = 20.0
        mag = math.sqrt(self._shake_offset.x**2 + self._shake_offset.y**2)
        if mag > max_shake:
            self._shake_offset = glm.vec2(
                self._shake_offset.x / mag * max_shake,
                self._shake_offset.y / mag * max_shake,
            )

        # Flash fade
        if self._flash_alpha > 0:
            elapsed = time.time() - self._flash_start
            self._flash_alpha = max(0.0, 1.0 - elapsed / max(0.001, self._flash_duration))

    @property
    def shake_offset(self) -> glm.vec2:
        return self._shake_offset

    @property
    def flash_alpha(self) -> float:
        return self._flash_alpha

"""Optional Web Aim reticle over the mirrored camera frame (visual aid only)."""
import math
import time

import cv2

from config.settings import WINDOW_W, WINDOW_H
from tracking.gesture import Gesture, web_aim, web_origin


class WebAimHUD:
    def __init__(self):
        self.enabled = False
        self._flashes = []

    def flash(self, origin, now=None):
        if self.enabled:
            self._flashes.append(((float(origin.x), float(origin.y)),
                                  time.monotonic() if now is None else now))
            self._flashes = self._flashes[-8:]

    def render(self, bgr, hands, now=None):
        if not self.enabled:
            return bgr
        now = time.monotonic() if now is None else now
        out = bgr.copy()
        height, width = out.shape[:2]
        sx, sy = width / WINDOW_W, height / WINDOW_H
        font = cv2.FONT_HERSHEY_SIMPLEX

        for hand in hands:
            landmarks = hand.aim_landmarks or hand.landmarks
            if len(landmarks) < 9:
                continue
            origin = web_origin(landmarks)
            direction = web_aim(landmarks)
            # Match WebProjectile's predicted landing point, not an object lock.
            tx = max(0.0, min(WINDOW_W - 1.0, origin.x + direction.x * 650.0))
            ty = max(0.0, min(WINDOW_H - 1.0, origin.y + direction.y * 650.0))
            tip = (round(origin.x * sx), round(origin.y * sy))
            target = (round(tx * sx), round(ty * sy))
            ready = hand.gesture in (Gesture.WEB_SHOT, Gesture.PINCH)
            color = (255, 240, 130) if ready else (200, 115, 50)
            radius = max(10, round(min(sx, sy) * (24 + 3 * math.sin(now * 7))))

            cv2.line(out, tip, target, (110, 75, 25), 1, cv2.LINE_AA)
            cv2.circle(out, tip, 4, color, 1, cv2.LINE_AA)
            cv2.circle(out, target, radius, color, 2 if ready else 1, cv2.LINE_AA)
            cv2.circle(out, target, 2, color, -1, cv2.LINE_AA)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                start = (target[0] + dx * (radius + 4), target[1] + dy * (radius + 4))
                end = (target[0] + dx * (radius + 11), target[1] + dy * (radius + 11))
                cv2.line(out, start, end, color, 1, cv2.LINE_AA)
            label = 'AIM READY' if ready else 'AIM TRACKING'
            label_x = max(4, min(width - 118, target[0] - 53))
            label_y = max(14, min(height - 5, target[1] + radius + 27))
            cv2.putText(out, label, (label_x, label_y), font, 0.35, color, 1, cv2.LINE_AA)

        active_flashes = []
        for origin, shot_at in self._flashes:
            age = now - shot_at
            if 0 <= age < 0.42:
                active_flashes.append((origin, shot_at))
                point = (round(origin[0] * sx), round(origin[1] * sy))
                radius = max(2, round((12 + age * 140) * min(sx, sy)))
                brightness = max(0, round(255 * (1 - age / 0.42)))
                cv2.circle(out, point, radius, (255, brightness, brightness), 2, cv2.LINE_AA)
                cv2.circle(out, point, 4, (255, 255, 255), -1, cv2.LINE_AA)
            elif age < 0:
                active_flashes.append((origin, shot_at))
        self._flashes = active_flashes

        cv2.putText(out, 'WEB AIM ON', (max(4, width - 116), height - 15),
                    font, 0.38, (255, 205, 85), 1, cv2.LINE_AA)
        return out

"""Opt-in cinematic war-mode overlay; never an actual danger detector."""
import math

import cv2


class WarModeHUD:
    def render(self, bgr, now: float):
        """Draw a responsive warning theme on a copy of the camera frame."""
        out = bgr.copy()
        height, width = out.shape[:2]
        unit = min(width / 1280.0, height / 720.0)
        inset = max(9, round(15 * unit))
        length = max(24, round(55 * unit))
        glow = 0.5 + 0.5 * math.sin(now * 5.0)
        red = (20, 45, round(160 + 75 * glow))  # BGR, red dominates
        thickness = max(2, round(3 * unit))
        for x, dx in ((inset, 1), (width - inset - 1, -1)):
            for y, dy in ((inset, 1), (height - inset - 1, -1)):
                cv2.line(out, (x, y), (x + dx * length, y), red, thickness, cv2.LINE_AA)
                cv2.line(out, (x, y), (x, y + dy * length), red, thickness, cv2.LINE_AA)
        # Keep central view and upper-right map legible; status is near bottom.
        band_h = max(45, round(56 * unit))
        top = max(0, height - band_h - inset)
        layer = out.copy()
        cv2.rectangle(layer, (inset, top), (width - inset, height - inset),
                      (12, 10, 48), -1)
        cv2.addWeighted(layer, 0.56, out, 0.44, 0, out)
        cv2.line(out, (inset, top), (width - inset, top), red,
                 max(1, round(unit)), cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out, 'WAR MODE - VISUAL SIMULATION',
                    (inset + round(17 * unit), top + round(22 * unit)),
                    font, 0.56 * unit, (110, 180, 255),
                    max(1, round(unit)), cv2.LINE_AA)
        cv2.putText(out, 'FIST TO RETRACT WEB  |  JARVIS WAR MODE OFF',
                    (inset + round(17 * unit), top + round(43 * unit)),
                    font, 0.43 * unit, (100, 145, 220),
                    max(1, round(unit)), cv2.LINE_AA)
        return out

"""Decorative pulse and breathing animation; no physiological measurements."""
import math

import cv2
import numpy as np


class VitalsHUD:
    """Camera-space cinematic effects, never derived from biometric signals."""

    LABEL = 'PULSE FX / BREATH FX - SIMULATED'
    label = LABEL

    def render(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Return a BGR copy with explicitly simulated pulse/breath visuals."""
        out = frame.copy()
        h, w = out.shape[:2]
        x0 = 8
        y0 = max(8, min(215, max(170, round(h * 0.3)), h - 148))
        right = min(w - 8, x0 + 372)
        bottom = min(h - 8, y0 + 140)
        if right <= x0 or bottom <= y0:
            return out

        panel = out[y0:bottom, x0:right]
        shade = np.zeros_like(panel)
        shade[:] = (27, 15, 9)
        cv2.addWeighted(panel, 0.28, shade, 0.72, 0, dst=panel)
        cv2.rectangle(out, (x0, y0), (right - 1, bottom - 1), (175, 118, 60), 1, cv2.LINE_AA)
        cv2.putText(out, self.LABEL, (x0 + 10, y0 + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 225, 175), 1, cv2.LINE_AA)
        cv2.putText(out, 'HEARTBEAT', (x0 + 12, y0 + 53),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (190, 210, 255), 2, cv2.LINE_AA)
        cv2.putText(out, 'BREATHING', (right - 103, bottom - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 220, 165), 1, cv2.LINE_AA)

        # An explicitly decorative beating heart icon; never imply a BPM value.
        beat = 1.0 + 0.12 * max(0.0, math.sin(t * 2 * math.pi * 1.2))
        center_x, center_y = right - 116, y0 + 51
        heart = []
        for angle in np.linspace(0, 2 * math.pi, 40):
            x = 16 * math.sin(angle) ** 3
            y = 13 * math.cos(angle) - 5 * math.cos(2 * angle) - 2 * math.cos(3 * angle) - math.cos(4 * angle)
            heart.append((round(center_x + x * beat * 0.75), round(center_y - y * beat * 0.75)))
        cv2.polylines(out, [np.asarray(heart, dtype=np.int32)], True,
                      (110, 150, 255), 2, cv2.LINE_AA)

        # EKG-inspired graphic: a purely periodic decorative waveform.
        wave_left = x0 + 12
        wave_right = max(wave_left + 1, right - 85)
        baseline = min(bottom - 19, y0 + 82)
        points = []
        for x in range(wave_left, wave_right):
            phase = ((x - wave_left) / 83.0 - t * 1.7) % 1.0
            spike = (22 * max(0.0, 1.0 - abs(phase - 0.25) / 0.045)
                     - 9 * max(0.0, 1.0 - abs(phase - 0.18) / 0.04)
                     - 7 * max(0.0, 1.0 - abs(phase - 0.32) / 0.06))
            points.append((x, round(baseline - spike)))
        cv2.polylines(out, [np.asarray(points, dtype=np.int32)], False,
                      (95, 175, 255), 3, cv2.LINE_AA)

        # Expanding breathing halo; slow visual rhythm, not a breath-rate value.
        center = (right - 46, min(bottom - 42, y0 + 83))
        breath = 0.5 + 0.5 * math.sin(t * 2.3)
        radius = round(15 + 11 * breath)
        cv2.circle(out, center, radius + 5, (110, 80, 30), 2, cv2.LINE_AA)
        cv2.circle(out, center, radius, (255, 205, 110), 2, cv2.LINE_AA)
        cv2.circle(out, center, 4, (255, 255, 220), -1, cv2.LINE_AA)
        return out

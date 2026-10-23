"""
SpiderVision — Hologram Menu
Voice-triggered holographic menu with slide/fade animations.
Rendered onto the camera frame (OpenCV).
"""
import math
import time
import cv2
import numpy as np
from config.settings import MENU_ITEMS, CAMERA_W, CAMERA_H


class HologramMenu:
    """
    Animates a voice-triggered holographic menu panel.
    Say 'Jarvis menu' to show and 'Jarvis close' to hide.
    """

    SLIDE_DURATION = 0.35   # seconds for open/close animation
    ITEM_H         = 52     # pixels per item
    PANEL_W        = 260

    def __init__(self):
        self._open         = False
        self._anim_t       = 0.0    # 0=closed, 1=open
        self._hovered      = -1
        self._selected     = -1
        self._last_toggle  = 0.0

    # ── State ─────────────────────────────────────────────────────────────────

    def show(self):
        if not self._open and time.time() - self._last_toggle > 0.5:
            self._open = True
            self._last_toggle = time.time()

    def hide(self):
        if self._open:
            self._open = False
            self._last_toggle = time.time()

    @property
    def is_open(self) -> bool:
        return self._anim_t > 0.05

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, dt: float, hand_pos=None):
        target = 1.0 if self._open else 0.0
        speed  = 1.0 / self.SLIDE_DURATION
        diff   = target - self._anim_t
        self._anim_t += diff * dt * speed * 10
        self._anim_t  = max(0.0, min(1.0, self._anim_t))

        # Hover detection (based on hand/mouse position)
        self._hovered = -1
        if hand_pos and self._anim_t > 0.5:
            hx, hy = hand_pos
            px, py = self._panel_pos()
            if px <= hx <= px + self.PANEL_W:
                for i in range(len(MENU_ITEMS)):
                    iy = py + 60 + i * self.ITEM_H
                    if iy <= hy <= iy + self.ITEM_H:
                        self._hovered = i
                        break

    def _panel_pos(self):
        """Returns top-left (x, y) of the menu panel."""
        px = int(CAMERA_W * 0.65)
        py = int(CAMERA_H * 0.2)
        return px, py

    def select_hovered(self):
        """Select an item only when the fingertip is inside an active menu row."""
        if self.is_open and self._hovered >= 0:
            return MENU_ITEMS[self._hovered]
        return None

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self, bgr: np.ndarray, t: float) -> np.ndarray:
        if self._anim_t < 0.02:
            return bgr

        a     = self._anim_t
        px, py = self._panel_pos()
        pw    = int(self.PANEL_W * a)
        total_h = 60 + len(MENU_ITEMS) * self.ITEM_H + 20

        # Slide from right
        offset = int((1 - a) * 80)
        draw_x = px + offset

        # ── Panel background ──────────────────────────────────────────────────
        overlay = bgr.copy()
        cv2.rectangle(overlay,
                      (draw_x, py),
                      (draw_x + pw, py + total_h),
                      (2, 8, 18), -1)
        cv2.addWeighted(overlay, 0.82, bgr, 0.18, 0, bgr)

        # Outer border
        col_border = (0, int(180 * a), int(255 * a))
        cv2.rectangle(bgr, (draw_x, py), (draw_x + pw, py + total_h),
                      col_border, 1, cv2.LINE_AA)

        # Glow corner accents
        ca = 16
        for bx2, by2, sx, sy in [
            (draw_x, py, 1, 1), (draw_x + pw, py, -1, 1),
            (draw_x + pw, py + total_h, -1, -1), (draw_x, py + total_h, 1, -1)
        ]:
            cv2.line(bgr, (bx2, by2), (bx2 + sx * ca, by2), col_border, 2, cv2.LINE_AA)
            cv2.line(bgr, (bx2, by2), (bx2, by2 + sy * ca), col_border, 2, cv2.LINE_AA)

        if pw < 30:
            return bgr

        font = cv2.FONT_HERSHEY_DUPLEX

        # Title bar
        title_y = py + 36
        cv2.putText(bgr, "[ SPIDER VISION ]", (draw_x + 12, title_y),
                    font, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.line(bgr, (draw_x + 8, py + 46), (draw_x + pw - 8, py + 46),
                 (0, 100, 180), 1, cv2.LINE_AA)

        # ── Menu items ────────────────────────────────────────────────────────
        for i, label in enumerate(MENU_ITEMS):
            iy    = py + 60 + i * self.ITEM_H
            hov   = (i == self._hovered)
            pulse = 0.7 + 0.3 * math.sin(t * 4.0 + i * 0.8) if hov else 0.0

            # Hover highlight
            if hov:
                hover_overlay = bgr.copy()
                cv2.rectangle(hover_overlay, (draw_x + 4, iy + 2),
                              (draw_x + pw - 4, iy + self.ITEM_H - 4),
                              (0, 60, 120), -1)
                cv2.addWeighted(hover_overlay, 0.7, bgr, 0.3, 0, bgr)
                col = (int(80 + 175 * pulse), int(200 + 55 * pulse), 255)
            else:
                col = (0, 180, 255) if label != "EXIT" else (0, 100, 255)

            # Item number
            num_col = (0, 80, 140)
            cv2.putText(bgr, f"{i+1:02d}", (draw_x + 12, iy + 32),
                        font, 0.36, num_col, 1, cv2.LINE_AA)

            # Item label
            cv2.putText(bgr, label, (draw_x + 42, iy + 32),
                        font, 0.5, col, 1, cv2.LINE_AA)

            # Item separator
            sep_alpha = 0.4
            cv2.line(bgr, (draw_x + 8, iy + self.ITEM_H - 1),
                     (draw_x + pw - 8, iy + self.ITEM_H - 1),
                     (0, 40, 80), 1, cv2.LINE_AA)

            # Arrow indicator for hovered
            if hov:
                ax = draw_x + pw - 22
                ay = iy + self.ITEM_H // 2
                cv2.arrowedLine(bgr, (ax - 12, ay), (ax, ay),
                                col, 1, cv2.LINE_AA, tipLength=0.5)

        # Animated particles on panel edges (digital pixel effect)
        for i in range(4):
            phase = (t * 0.8 + i * 0.25) % 1.0
            dot_x = draw_x + int(phase * pw)
            dot_y = py
            cv2.circle(bgr, (dot_x, dot_y), 1, (0, 200, 255), -1)
            cv2.circle(bgr, (dot_x, py + total_h), 1, (0, 200, 255), -1)

        # Instructions
        cv2.putText(bgr, "SAY JARVIS + OPTION / JARVIS CLOSE",
                    (draw_x + 12, py + total_h - 6),
                    font, 0.3, (0, 80, 120), 1, cv2.LINE_AA)

        return bgr

    def get_selected(self) -> str | None:
        """Returns selected menu item name if clicked, else None."""
        if self._selected >= 0:
            item = MENU_ITEMS[self._selected]
            self._selected = -1
            return item
        return None

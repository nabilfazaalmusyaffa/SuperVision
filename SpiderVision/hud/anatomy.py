"""Illustrated anatomy atlas; landmarks control an illustration, not a scan."""
import math
import time
from pathlib import Path

import cv2
import numpy as np


class AnatomyHUD:
    """Standalone BODY/BRAIN atlas for mirrored normalized camera landmarks."""

    DWELL_SECONDS = 0.65
    _BODY = (
        ('HEART', (610, 260), (235, 75, 245)),
        ('LUNGS', (578, 245), (220, 195, 95)),
        ('LIVER', (585, 300), (95, 165, 225)),
        ('STOMACH', (644, 303), (145, 195, 240)),
        ('KIDNEYS', (622, 323), (190, 150, 225)),
        ('SPINAL CORD', (604, 225), (225, 215, 95)),
        ('PERIPHERAL NERVES', (538, 315), (120, 220, 250)),
    )
    _BRAIN = (
        ('FRONTAL LOBE', (541, 279), (225, 185, 95)),
        ('PARIETAL LOBE', (614, 234), (125, 220, 245)),
        ('TEMPORAL LOBE', (593, 336), (205, 155, 235)),
        ('OCCIPITAL LOBE', (675, 288), (125, 235, 150)),
        ('CEREBELLUM', (674, 382), (125, 170, 250)),
        ('BRAINSTEM', (635, 420), (200, 210, 100)),
        ('SPINAL CORD', (653, 463), (225, 205, 100)),
    )
    _ART = {'BODY': ('body_atlas.png', (519, 138), (162, 445)),
            'BRAIN': ('brain_atlas.png', (420, 195), (350, 281))}
    _FACTS = {
        'HEART': 'Pumps blood throughout the body.',
        'LUNGS': 'Exchange oxygen and carbon dioxide.',
        'LIVER': 'Processes nutrients and produces bile.',
        'STOMACH': 'Begins the breakdown of food.',
        'KIDNEYS': 'Filter blood and produce urine.',
        'SPINAL CORD': 'Carries signals between brain and body.',
        'PERIPHERAL NERVES': 'Carry signals between limbs and spinal cord.',
        'FRONTAL LOBE': 'Supports planning and voluntary movement.',
        'PARIETAL LOBE': 'Processes touch and spatial awareness.',
        'TEMPORAL LOBE': 'Supports hearing and memory.',
        'OCCIPITAL LOBE': 'Processes visual information.',
        'CEREBELLUM': 'Coordinates balance and movement.',
        'BRAINSTEM': 'Regulates breathing and other basic functions.',
    }
    _assets = {}

    @property
    def zoom(self):
        return self._zoom

    @zoom.setter
    def zoom(self, value):
        self._zoom = max(1.0, min(2.5, float(value)))

    def __init__(self):
        self.mode = 'CLOSED'
        self.zoom = 1.0
        self.selected = None
        self._hover = None
        self._hover_since = None
        self._pinch_latched = False
        self._spread = None
        self._scaled_art = None

    def open(self, mode):
        if mode not in ('BODY', 'BRAIN'):
            raise ValueError('mode must be BODY or BRAIN')
        self.mode = mode
        self.zoom = 1.0
        self.selected = None
        self._reset_interaction()

    def close(self):
        self.mode = 'CLOSED'
        self.zoom = 1.0
        self.selected = None
        self._reset_interaction()

    def describe_selected(self):
        """Educational caption for the current selection, never a health result."""
        if self.mode == 'CLOSED' or not self.selected:
            return None
        return self._FACTS.get(self.selected)

    def _reset_interaction(self):
        self._hover = None
        self._hover_since = None
        self._pinch_latched = False
        self._spread = None

    def _anchor(self, pos):
        # Same affine transform as the artwork; invisible points stay invisible.
        return (600 + (pos[0] - 600) * self.zoom,
                355 + (pos[1] - 355) * self.zoom)

    @classmethod
    def _asset(cls, mode):
        """Downsample and tint each original RGBA illustration once."""
        if mode not in cls._assets:
            filename, _, size = cls._ART[mode]
            source = cv2.imread(str(Path(__file__).resolve().parent.parent /
                                    'assets' / 'anatomy' / filename), cv2.IMREAD_UNCHANGED)
            if source is None or source.shape[2] != 4:
                raise FileNotFoundError(f'RGBA anatomy illustration unavailable: {filename}')
            small = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
            luminance = cv2.cvtColor(small[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
            strength = .42 + luminance / 255 * .58
            tint = np.array([245, 225, 90], dtype=np.float32)
            rgb = np.clip(strength[:, :, None] * tint, 0, 255).astype(np.uint8)
            alpha = small[:, :, 3].astype(np.float32) / 255 * .78
            halo = cv2.GaussianBlur(cv2.copyMakeBorder(alpha, 8, 8, 8, 8,
                                                       cv2.BORDER_CONSTANT), (0, 0), 3)
            cls._assets[mode] = (rgb, alpha, halo)
        return cls._assets[mode]

    def _draw_art(self, out, sx, sy):
        rgb, alpha, halo = self._asset(self.mode)
        _, origin, _ = self._ART[self.mode]
        x0 = round((600 + (origin[0] - 600) * self.zoom) * sx)
        y0 = round((355 + (origin[1] - 355) * self.zoom) * sy)
        w = max(1, round(rgb.shape[1] * self.zoom * sx))
        h = max(1, round(rgb.shape[0] * self.zoom * sy))
        # Cached blurred alpha makes a contour halo without per-frame convolution.
        hw = max(1, round(halo.shape[1] * self.zoom * sx))
        hh = max(1, round(halo.shape[0] * self.zoom * sy))
        hx, hy = x0 - round(8 * self.zoom * sx), y0 - round(8 * self.zoom * sy)
        cache_key = (self.mode, w, h, hw, hh)
        if self._scaled_art is None or self._scaled_art[0] != cache_key:
            picture = (rgb if (w, h) == (rgb.shape[1], rgb.shape[0]) else
                       cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR))
            opacity = (alpha if (w, h) == (rgb.shape[1], rgb.shape[0]) else
                       cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR))
            halo_scaled = cv2.resize(halo, (hw, hh), interpolation=cv2.INTER_LINEAR)
            self._scaled_art = (cache_key, picture, opacity, halo_scaled)
        _, picture, opacity, halo_scaled = self._scaled_art
        lx, ly = max(hx, round(365 * sx)), max(hy, round(132 * sy))
        rx, by = min(hx + hw, round(815 * sx)), min(hy + hh, round(610 * sy))
        if lx < rx and ly < by:
            glow = halo_scaled[ly-hy:by-hy, lx-hx:rx-hx, None] * .9
            dest = out[ly:by, lx:rx]
            dest[:] = (dest * (1 - glow) +
                       np.array([230, 160, 45]) * glow).astype(np.uint8)
        x1, y1 = max(x0, round(365 * sx)), max(y0, round(132 * sy))
        x2, y2 = min(x0 + w, round(815 * sx)), min(y0 + h, round(610 * sy))
        if x1 >= x2 or y1 >= y2:
            return
        src = picture[y1-y0:y2-y0, x1-x0:x2-x0]
        mask = opacity[y1-y0:y2-y0, x1-x0:x2-x0, None]
        dest = out[y1:y2, x1:x2]
        dest[:] = (dest * (1 - mask) + src * mask).astype(np.uint8)

    @staticmethod
    def _tip(hand, width, height):
        landmarks = getattr(hand, 'landmarks', ())
        if len(landmarks) < 9:
            return None
        x, y = landmarks[8][:2]
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
            return None
        return (x * width, y * height)

    def _interact(self, hands, now, width, height):
        valid = [(hand, tip) for hand in hands
                 if (tip := self._tip(hand, width, height)) is not None]
        tips = [tip for _, tip in valid]
        if len(tips) >= 2:
            distance = math.dist(tips[0], tips[1]) / min(width, height)
            if self._spread is not None and self._spread > 0:
                self.zoom = max(1.0, min(2.5, self.zoom * distance / self._spread))
            self._spread = distance
            self._hover = self._hover_since = None
            self._pinch_latched = False
            return
        self._spread = None
        if not tips:
            self._hover = self._hover_since = None
            self._pinch_latched = False
            return
        px, py = tips[0][0] * 1280 / width, tips[0][1] * 720 / height
        parts = self._BODY if self.mode == 'BODY' else self._BRAIN
        candidates = [(math.dist((px, py), (ax, ay)), name)
                      for name, anchor, _ in parts
                      for ax, ay in (self._anchor(anchor),)
                      if 365 <= ax < 815 and 132 <= ay < 547]
        nearest, label = min(candidates) if candidates else (float('inf'), None)
        label = label if nearest <= 31 * max(1, self.zoom ** .35) and 365 <= px < 815 and 132 <= py < 547 else None
        if label != self._hover:
            self._hover, self._hover_since = label, now
        gesture = getattr(valid[0][0], 'gesture', '')
        pinching = gesture == 'PINCH'
        if label and ((pinching and not self._pinch_latched) or
                      (gesture == 'POINT' and now - self._hover_since >= self.DWELL_SECONDS)):
            self.selected = label
        self._pinch_latched = pinching

    def render(self, bgr, hands, now=None):
        """Return a separate BGR frame; no hidden-mode changes to the original."""
        if self.mode == 'CLOSED':
            return bgr
        now = time.monotonic() if now is None else now
        height, width = bgr.shape[:2]
        self._interact(hands or (), now, width, height)
        out = bgr.copy()
        sx, sy = width / 1280, height / 720
        point = lambda x, y: (round(x * sx), round(y * sy))
        font = cv2.FONT_HERSHEY_SIMPLEX
        def text(label, x, y, scale=.4, color=(225, 230, 240), thickness=1):
            cv2.putText(out, label, point(x, y), font, scale * min(sx, sy),
                        color, thickness, cv2.LINE_AA)
        def line(a, b, color=(135, 200, 205), thickness=1):
            cv2.line(out, point(*a), point(*b), color, thickness, cv2.LINE_AA)
        # A compact central panel leaves the right-side menu and left telemetry free.
        left, top = point(357, 66)
        right, bottom = point(821, 651)
        panel = out[top:bottom, left:right]
        dark = np.empty_like(panel)
        dark[:] = (17, 17, 25)
        cv2.addWeighted(dark, .78, panel, .22, 0, panel)
        cv2.rectangle(out, point(357, 66), point(821, 651), (135, 135, 85), 1)
        text('ANATOMY ATLAS  /  ' + self.mode, 372, 91, .52, (250, 220, 120))
        text('POINT + DWELL / PINCH: SELECT     TWO HANDS: ZOOM', 372, 115, .32)
        # Grid lies behind the RGBA illustration and stays inside the crop.
        for yy in range(148, 590, 32):
            line((368, yy), (812, yy), (61, 52, 17))
        for xx in range(380, 810, 32):
            line((xx, 135), (xx, 608), (61, 52, 17))
        self._draw_art(out, sx, sy)
        if self.mode == 'BODY':
            # Separate schematic nerve trace, on the bone-dominant arm; never
            # label the blood vessels in the opposite arm as nerves.
            trace = ((551, 238), (545, 265), (541, 290),
                     (538, 315), (538, 341), (530, 369))
            for a, b in zip(trace, trace[1:]):
                start, end = self._anchor(a), self._anchor(b)
                if 365 < start[0] < 815 and 132 < start[1] < 610:
                    line(start, end, (75, 105, 42), 5)
                    line(start, end, (245, 190, 90), 1)

        parts = self._BODY if self.mode == 'BODY' else self._BRAIN
        for i, (label, anchor, color) in enumerate(parts):
            x, y = self._anchor(anchor)
            if not (365 <= x < 815 and 132 <= y < 547):
                continue
            right = i % 2 == 1
            label_x = 735 if right else 370
            label_y = 154 + i * 56
            edge_x = label_x - 9 if right else label_x + 105
            line((x, y), (edge_x, label_y - 4), color)
            cv2.circle(out, point(x, y), 3 if self.selected != label else 6,
                       color, -1, cv2.LINE_AA)
            text(label, label_x, label_y, .39, color)
        if self.selected:
            cv2.rectangle(out, point(357, 654), point(821, 711), (22, 34, 38), -1)
            text('SELECTED: ' + self.selected, 373, 675, .4, (245, 225, 130))
            text(self._FACTS[self.selected], 373, 698, .35, (185, 228, 216))
        else:
            cv2.rectangle(out, point(357, 654), point(821, 711), (22, 34, 38), -1)
            text('POINT TO A PART FOR DETAILS', 373, 675, .37, (185, 228, 216))
            text('SAY JARVIS DESCRIBE TO HEAR IT', 373, 698, .37, (185, 228, 216))
        for hand in hands or ():
            tip = self._tip(hand, width, height)
            if tip is not None:
                px, py = round(tip[0]), round(tip[1])
                if 357 * sx <= px <= 821 * sx and 66 * sy <= py <= 651 * sy:
                    color = (255, 235, 105) if self._hover else (235, 180, 75)
                    cv2.circle(out, (px, py), 7, color, 1, cv2.LINE_AA)
                    cv2.circle(out, (px, py), 2, color, -1, cv2.LINE_AA)
        text('ZOOM %.1fX' % self.zoom, 710, 602, .4, (245, 225, 130))
        cv2.rectangle(out, point(367, 615), point(810, 641), (60, 78, 135), 1)
        text('EDUCATIONAL ILLUSTRATION - NOT A MEDICAL SCAN',
             374, 632, .44, (135, 190, 255))
        return out

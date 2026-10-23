"""
SpiderVision — Math Utilities
GLM helpers, matrix math, easing functions.
"""
import math
import glm
import numpy as np


# ─── Easing ───────────────────────────────────────────────────────────────────

def ease_in_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0


def ease_out_elastic(t: float, amplitude: float = 1.0, period: float = 0.3) -> float:
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    s = period / (2 * math.pi) * math.asin(1.0 / amplitude)
    return amplitude * (2 ** (-10 * t)) * math.sin((t - s) * (2 * math.pi) / period) + 1.0


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def lerp_vec2(a: glm.vec2, b: glm.vec2, t: float) -> glm.vec2:
    t = max(0.0, min(1.0, t))
    return glm.vec2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)


def lerp_vec3(a: glm.vec3, b: glm.vec3, t: float) -> glm.vec3:
    t = max(0.0, min(1.0, t))
    return glm.vec3(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
    )


def ema(current: float, new_val: float, alpha: float) -> float:
    """Exponential moving average for smoothing."""
    return current + alpha * (new_val - current)


def ema_vec2(current: glm.vec2, new_val: glm.vec2, alpha: float) -> glm.vec2:
    return glm.vec2(
        ema(current.x, new_val.x, alpha),
        ema(current.y, new_val.y, alpha),
    )


# ─── Coordinate Helpers ───────────────────────────────────────────────────────

def ndc(x: float, y: float, w: int, h: int) -> glm.vec2:
    """Pixel → Normalized Device Coordinates."""
    return glm.vec2(x / w * 2.0 - 1.0, 1.0 - y / h * 2.0)


def screen_to_ndc(px: float, py: float, w: int, h: int) -> tuple[float, float]:
    return (px / w * 2.0 - 1.0, 1.0 - py / h * 2.0)


def pixel_to_uv(px: float, py: float, w: int, h: int) -> glm.vec2:
    return glm.vec2(px / w, 1.0 - py / h)


# ─── Circle / Polar ───────────────────────────────────────────────────────────

def circle_points(n: int, r: float = 1.0, offset: float = 0.0) -> list[tuple[float, float]]:
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n + offset
        pts.append((math.cos(a) * r, math.sin(a) * r))
    return pts


def polar_to_cartesian(angle_rad: float, radius: float) -> glm.vec2:
    return glm.vec2(math.cos(angle_rad) * radius, math.sin(angle_rad) * radius)


# ─── Matrix Helpers ───────────────────────────────────────────────────────────

def ortho_matrix(w: int, h: int) -> glm.mat4:
    """Orthographic projection for 2D HUD rendering (pixel coords)."""
    return glm.ortho(0.0, float(w), float(h), 0.0, -1.0, 1.0)


def make_transform_2d(cx: float, cy: float, angle_rad: float, scale: float = 1.0) -> glm.mat4:
    m = glm.mat4(1.0)
    m = glm.translate(m, glm.vec3(cx, cy, 0.0))
    m = glm.rotate(m, angle_rad, glm.vec3(0.0, 0.0, 1.0))
    m = glm.scale(m, glm.vec3(scale, scale, 1.0))
    return m


# ─── Noise / Hash ─────────────────────────────────────────────────────────────

def hash21(p: glm.vec2) -> float:
    """Simple deterministic hash → pseudo-random float 0..1."""
    n = math.sin(p.x * 127.1 + p.y * 311.7) * 43758.5453123
    return n - math.floor(n)


# ─── NumPy ────────────────────────────────────────────────────────────────────

def np_circle_strip(cx: float, cy: float, r: float, n: int) -> np.ndarray:
    """Generate ring vertex data as flat float32 array [x, y, ...]."""
    angles = np.linspace(0, 2 * np.pi, n + 1)
    xs = cx + np.cos(angles) * r
    ys = cy + np.sin(angles) * r
    return np.column_stack([xs, ys]).astype(np.float32)


def glm_to_np(m: glm.mat4) -> np.ndarray:
    return np.array(m, dtype=np.float32).reshape(4, 4).T.flatten()

"""
SpiderVision — Spider Web Projectile
Procedural 3D-ish web mesh with physics, elastic animation,
particle trail, shockwave on impact.
"""
import math
import time
import random
import numpy as np
import cv2
import moderngl
import glm

from utils.logger import log
from config.settings import (
    WINDOW_W, WINDOW_H,
    WEB_SPEED, WEB_RINGS, WEB_SPOKES, WEB_LIFETIME, WEB_ELASTIC
)


# ─── Web Shader ───────────────────────────────────────────────────────────────

_WEB_VERT = """
#version 410 core
in vec2 in_pos;
in float in_alpha;

uniform vec2  u_origin;
uniform vec2  u_target;
uniform float u_progress;   // 0..1 launch progress
uniform float u_time;
uniform float u_scale;

out float v_alpha;
out vec2  v_uv;

void main() {
    // Transform web coordinates to world
    vec2 fwd  = u_target - u_origin;
    float len  = length(fwd);
    if (len < 0.001) { gl_Position = vec4(0.0); return; }
    vec2 fwd_n = fwd / len;
    vec2 right = vec2(-fwd_n.y, fwd_n.x);

    // Apply web geometry
    vec2 world = u_origin
               + fwd_n * in_pos.x * len * u_progress
               + right * in_pos.y * u_scale;

    // Elastic wave deformation
    float wave = sin(in_pos.x * 8.0 + u_time * 6.0) * (1.0 - u_progress) * 6.0;
    world += right * wave;

    // Screen → NDC
    vec2 ndc = vec2(world.x / """ + str(float(WINDOW_W)) + """ * 2.0 - 1.0,
                    1.0 - world.y / """ + str(float(WINDOW_H)) + """ * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_alpha = in_alpha;
    v_uv    = in_pos * 0.5 + 0.5;
}
"""

_WEB_FRAG = """
#version 410 core
in float v_alpha;
in vec2  v_uv;

uniform float u_time;
uniform vec3  u_color;
uniform float u_glow;

out vec4 f_color;

void main() {
    // Fresnel-like edge glow
    float edge = abs(v_uv.x - 0.5) * 2.0;
    float glow = pow(edge, 0.5) * u_glow;

    // Animated shimmer along web strands
    float shimmer = 0.7 + 0.3 * sin(v_uv.x * 20.0 + u_time * 4.0);

    vec3 col = u_color * shimmer + vec3(glow);
    float alpha = v_alpha * (0.75 + glow * 0.25);

    f_color = vec4(col, alpha);
}
"""

_SHOCKWAVE_VERT = """
#version 410 core
in vec2 in_pos;
uniform vec2  u_center;
uniform float u_radius;
void main() {
    vec2 world = u_center + in_pos * u_radius;
    vec2 ndc   = vec2(world.x / """ + str(float(WINDOW_W)) + """ * 2.0 - 1.0,
                      1.0 - world.y / """ + str(float(WINDOW_H)) + """ * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

_SHOCKWAVE_FRAG = """
#version 410 core
uniform float u_alpha;
uniform vec3  u_color;
out vec4 f_color;
void main() {
    f_color = vec4(u_color, u_alpha);
}
"""


def _build_web_mesh(rings: int = WEB_RINGS,
                    spokes: int = WEB_SPOKES) -> tuple[np.ndarray, np.ndarray]:
    """
    Build line pairs for longitudinal strands and cross-web connections.
    X = forward 0..1, Y = lateral -0.5..0.5.
    """
    verts = []
    alphas = []

    # Each consecutive pair is an actual line segment (not a point cloud).
    for s in range(spokes):
        lateral = (s / (spokes - 1) - 0.5) if spokes > 1 else 0.0
        for r in range(rings):
            t0, t1 = r / rings, (r + 1) / rings
            verts.extend(((t0, lateral * t0), (t1, lateral * t1)))
            alphas.extend((1.0 - t0 * 0.25, 1.0 - t1 * 0.25))
    for r in range(1, rings + 1):
        t = r / rings
        for s in range(spokes - 1):
            y0 = (s / (spokes - 1) - 0.5) * t
            y1 = ((s + 1) / (spokes - 1) - 0.5) * t
            verts.extend(((t, y0), (t, y1)))
            alphas.extend((0.9, 0.9))

    arr = np.array(verts, dtype=np.float32)
    alp = np.array(alphas, dtype=np.float32)
    return arr, alp


class WebProjectile:
    """One launched web projectile."""

    flight_duration = 0.36  # seconds; fast visual launch, independent of linger time

    def __init__(self, origin: glm.vec2, direction: glm.vec2, side: str,
                 war_mode: bool = False):
        self.origin    = glm.vec2(origin)
        self.side      = side
        self.war_mode  = war_mode
        self.visual_scale = 1.6 if war_mode else 1.0
        spd            = WEB_SPEED * 60.0   # pixels/second
        self.velocity  = glm.vec2(direction.x * spd, direction.y * spd)
        # Keep the entire projected web visible on the camera view.
        self.target = glm.vec2(
            max(0.0, min(WINDOW_W - 1.0, origin.x + direction.x * 650.0)),
            max(0.0, min(WINDOW_H - 1.0, origin.y + direction.y * 650.0)),
        )
        self.progress  = 0.0     # 0..1
        self.alive     = True
        self.impact    = False
        self.shockwave_r = 0.0
        self.shockwave_alive = False
        self._age      = 0.0
        self._impact_emitted = False
        self.retracting = False
        self._retract_age = 0.0
        self._retract_from = 0.0

    def retract(self) -> bool:
        """Animate a war-mode web back toward its visual origin once."""
        if not self.war_mode or not self.alive or self.retracting or self.progress <= 0:
            return False
        self.retracting = True
        self._retract_from = self.progress
        self._retract_age = 0.0
        self.shockwave_alive = False
        return True

    def update(self, dt: float):
        if self.retracting:
            self._retract_age += max(0.0, dt)
            phase = min(1.0, self._retract_age / 0.8)
            self.progress = self._retract_from * (1.0 - phase) ** 2
            if phase >= 1.0:
                self.progress = 0.0
                self.alive = False
            return
        self._age += max(0.0, dt)
        phase = min(1.0, self._age / self.flight_duration)
        self.progress = 1.0 - (1.0 - phase) ** 6
        if self.progress >= 1.0 and not self.impact:
            self.impact           = True
            self.shockwave_alive  = True
        if self.shockwave_alive:
            self.shockwave_r += dt * 110.0  # keep impact halo through the web's linger phase
            if self.shockwave_r > 180:
                self.shockwave_alive = False
        if self._age > (8.0 if self.war_mode else WEB_LIFETIME):
            self.alive = False


class WebProjectileSystem:
    """Manages all active web projectiles and their rendering."""

    def __init__(self, ctx: moderngl.Context, shader_mgr):
        self.ctx        = ctx
        self._webs: list[WebProjectile] = []
        self._verts, self._alphas = _build_web_mesh()
        self._build_gl(shader_mgr)
        log.info(f"WebProjectileSystem ready — {len(self._verts)} web vertices")

    def _build_gl(self, sm):
        self._prog = sm.load_src("web", _WEB_VERT, _WEB_FRAG)
        self._shock_prog = sm.load_src("shockwave", _SHOCKWAVE_VERT, _SHOCKWAVE_FRAG)

        # Build combined vertex data [x, y, alpha]
        data = np.column_stack([self._verts, self._alphas]).astype(np.float32)
        self._web_vbo = self.ctx.buffer(data)
        self._web_vao = self.ctx.vertex_array(
            self._prog,
            [(self._web_vbo, "2f 1f", "in_pos", "in_alpha")]
        )

        # Shockwave ring (unit circle as line loop)
        n = 64
        angles = np.linspace(0, 2 * math.pi, n + 1)
        ring_v  = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)
        self._shock_vbo = self.ctx.buffer(ring_v)
        self._shock_vao = self.ctx.vertex_array(
            self._shock_prog,
            [(self._shock_vbo, "2f", "in_pos")]
        )

    def launch(self, origin: glm.vec2, direction: glm.vec2, side: str,
               war_mode: bool = False):
        if war_mode:
            self._webs = [web for web in self._webs
                          if not (web.war_mode and web.side == side)]
        self._webs.append(WebProjectile(origin, direction, side, war_mode=war_mode))
        log.debug(f"Web launched from {side} hand at ({origin.x:.0f}, {origin.y:.0f})")

    def retract(self, side: str) -> bool:
        """Retract the most recent active war web of this hand, never another's."""
        for web in reversed(self._webs):
            if web.side == side and web.retract():
                return True
        return False

    def clear_war_webs(self):
        """Discard war effects when the mode is disabled or the face locks."""
        self._webs = [web for web in self._webs if not web.war_mode]

    def update(self, dt: float):
        for w in self._webs:
            w.update(dt)
        self._webs = [w for w in self._webs if w.alive]

    def render_overlay(self, bgr: np.ndarray) -> np.ndarray:
        """Composite a bright web on the camera image without requiring GL.

        Web positions use window coordinates; scale them to the actual camera
        frame before uploading its texture. The core is visible from the
        fingertip even during the very first launch frame.
        """
        if not self._webs:
            return bgr
        out = bgr.copy()
        height, width = out.shape[:2]
        sx, sy = width / WINDOW_W, height / WINDOW_H
        for web in self._webs:
            ox, oy = web.origin.x * sx, web.origin.y * sy
            tx = ox + (web.target.x * sx - ox) * web.progress
            ty = oy + (web.target.y * sy - oy) * web.progress
            origin = (round(ox), round(oy))
            tip = (round(tx), round(ty))
            delta = np.array((tx - ox, ty - oy), dtype=float)
            length = float(np.linalg.norm(delta))
            normal = np.array((-delta[1], delta[0])) / max(length, 1.0)
            scale = web.visual_scale
            outer = (25, 85, 255) if web.war_mode else (180, 90, 0)
            core = (165, 215, 255) if web.war_mode else (255, 240, 180)
            cv2.line(out, origin, tip, outer, round(13 * scale), cv2.LINE_AA)
            cv2.line(out, origin, tip, core, round(6 * scale), cv2.LINE_AA)
            cv2.line(out, origin, tip, (255, 255, 255), round(2 * scale), cv2.LINE_AA)
            if length >= 12:
                for side in (-1, 1):
                    for fraction in (0.45, 0.7, 0.95):
                        along = np.array((ox, oy)) + delta * fraction
                        spread = normal * side * min(22.0 * scale, length * 0.13 * scale) * fraction
                        start = tuple(np.rint(along).astype(int))
                        end = tuple(np.rint(along + spread).astype(int))
                        cv2.line(out, start, end, core, round(2 * scale), cv2.LINE_AA)
            cv2.circle(out, tip, round(6 * scale), (255, 255, 255), -1, cv2.LINE_AA)
            if web.retracting:
                # Animated collar follows the returning tip; the beam shortens
                # toward its actual origin, not a fixed screen-space point.
                pulse = 0.5 + 0.5 * math.sin(web._retract_age * 17.0)
                radius = max(10, round(22 * min(sx, sy) * scale))
                color = (40, round(125 + 80 * pulse), 255)
                cv2.circle(out, tip, radius, color, 4, cv2.LINE_AA)
                cv2.circle(out, tip, radius + 7, (10, 60, 140), 2, cv2.LINE_AA)
                label = 'WEB RETRACTING'
                font = cv2.FONT_HERSHEY_SIMPLEX
                label_scale = 0.56
                (label_w, label_h), _ = cv2.getTextSize(label, font, label_scale, 2)
                label_x = max(5, min(width - label_w - 5, tip[0] - label_w // 2))
                label_y = max(label_h + 9, min(height - 7, tip[1] - radius - 13))
                cv2.rectangle(out, (label_x - 4, label_y - label_h - 5),
                              (label_x + label_w + 4, label_y + 5), (12, 10, 30), -1)
                cv2.putText(out, label, (label_x, label_y), font,
                            label_scale, color, 2, cv2.LINE_AA)
            if web.shockwave_alive:
                radius = max(1, round(web.shockwave_r * min(sx, sy)))
                center = (round(web.target.x * sx), round(web.target.y * sy))
                cv2.circle(out, center, radius, outer, round(3 * scale), cv2.LINE_AA)
        return out

    def render(self, t: float, particles=None):
        if not self._webs:
            return

        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.ctx.disable(moderngl.DEPTH_TEST)


        for w in self._webs:
            # Web mesh
            self._prog["u_origin"].value  = (w.origin.x, w.origin.y)
            self._prog["u_target"].value  = (w.target.x, w.target.y)
            self._prog["u_progress"].value= w.progress
            self._prog["u_time"].value    = t
            self._prog["u_scale"].value   = 60.0 * w.visual_scale
            self._prog["u_color"].value   = (1.0, 0.55, 0.36) if w.war_mode else (0.6, 0.85, 1.0)
            self._prog["u_glow"].value    = 0.5 + 0.3 * math.sin(t * 8.0)
            self._web_vao.render(moderngl.LINES)

            # Shockwave ring on impact
            if w.shockwave_alive:
                alpha = max(0.0, 1.0 - w.shockwave_r / 180.0)
                self._shock_prog["u_center"].value = (w.target.x, w.target.y)
                self._shock_prog["u_radius"].value = w.shockwave_r
                self._shock_prog["u_alpha"].value  = alpha * 0.9
                self._shock_prog["u_color"].value  = (0.3, 0.7, 1.0)
                self._shock_vao.render(moderngl.LINE_LOOP)

                # Emit particles on impact
                if particles and w.impact and not w._impact_emitted:
                    from effects.particle_engine import PType
                    particles.emit_burst(w.target.x, w.target.y,
                        count=48, ptype=PType.SPARK,
                        color=(0.4, 0.8, 1.0, 1.0),
                        speed=200, life_range=(0.3, 0.8))
                    w._impact_emitted = True

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    @property
    def active_count(self) -> int:
        return len(self._webs)

    def release(self):
        self._web_vbo.release()
        self._web_vao.release()
        self._shock_vbo.release()
        self._shock_vao.release()

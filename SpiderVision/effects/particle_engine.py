"""
SpiderVision — GPU Particle Engine
Up to 20,000 particles using GPU instancing via ModernGL.
Particle types: Dust, Glow, Spark, Smoke, Electric, DigitalPixel,
                Hexagon, LightTrail, EnergyOrb
"""
import math
import time
import random
import numpy as np
import moderngl
import glm

from utils.logger import log
from config.settings import (
    WINDOW_W, WINDOW_H, MAX_PARTICLES, PARTICLE_GRAVITY
)


# ─── Particle Vertex Layout (per-instance, interleaved float32) ───────────────
# [pos_x, pos_y, vel_x, vel_y, life, max_life, size, type_id, r, g, b, a, angle, spin]
#   0       1      2     3      4       5        6       7      8  9 10 11   12    13
PARTICLE_STRIDE = 14   # floats per particle
PARTICLE_DTYPE  = np.float32


# ─── GLSL Shaders ─────────────────────────────────────────────────────────────

_PARTICLE_VERT = """
#version 410 core
// Per-vertex (quad corner)
in vec2 in_corner;   // [-0.5..0.5]
// Per-instance
in vec2  i_pos;
in vec2  i_vel;
in float i_life;
in float i_max_life;
in float i_size;
in float i_type;
in vec4  i_color;
in float i_angle;
in float i_spin;

uniform float u_time;

out vec4  v_color;
out vec2  v_uv;
out float v_type;
out float v_life_frac;

void main() {
    float frac = clamp(i_life / max(i_max_life, 0.001), 0.0, 1.0);
    v_life_frac = frac;

    // Fade in/out
    float alpha_fade = frac < 0.1 ? frac * 10.0 : (frac > 0.8 ? (1.0 - frac) * 5.0 : 1.0);
    v_color = vec4(i_color.rgb, i_color.a * alpha_fade);
    v_type  = i_type;

    // Velocity-based stretch (also keeps i_vel active — prevents GLSL optimizer removal)
    float speed   = length(i_vel);
    vec2 vel_dir  = speed > 0.5 ? normalize(i_vel) : vec2(1.0, 0.0);
    float stretch = 1.0 + speed * 0.004;

    // Rotation: trail particles align to velocity
    float a = i_angle + i_spin * (1.0 - frac);
    bool is_trail = i_type > 6.5;
    if (is_trail && speed > 0.5) {
        a = atan(vel_dir.y, vel_dir.x);
    }
    float ca = cos(a), sa = sin(a);
    vec2 rot = vec2(in_corner.x * ca - in_corner.y * sa,
                    in_corner.x * sa + in_corner.y * ca);

    // Scale: trail elongated + velocity stretch; sparks get subtle stretch
    vec2 sz = vec2(i_size, i_size);
    if (is_trail) {
        sz.x *= 3.0 + stretch;
    } else if (i_type > 1.5 && i_type < 3.5) {
        sz.x *= max(1.0, stretch * 0.5);   // Spark: subtle motion blur
    }

    vec2 world = i_pos + rot * sz;

    // Screen coords -> NDC
    vec2 ndc = vec2(world.x / float(""" + str(WINDOW_W) + """) * 2.0 - 1.0,
                    1.0 - world.y / float(""" + str(WINDOW_H) + """) * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_corner + 0.5;
}
"""

_PARTICLE_FRAG = """
#version 410 core
in vec4  v_color;
in vec2  v_uv;
in float v_type;
in float v_life_frac;
out vec4 f_color;

float circle(vec2 uv) { return 1.0 - smoothstep(0.3, 0.5, length(uv - 0.5)); }

float hex(vec2 uv) {
    vec2 p = abs(uv - 0.5) * 2.0;
    return 1.0 - step(1.0, max(p.x * 0.866025 + p.y * 0.5, p.y));
}

float ring(vec2 uv) {
    float d = length(uv - 0.5);
    return smoothstep(0.2, 0.22, d) * (1.0 - smoothstep(0.45, 0.5, d));
}

void main() {
    float mask;
    int t = int(v_type);
    if      (t == 0) mask = circle(v_uv);          // Dust
    else if (t == 1) mask = circle(v_uv);          // Glow (additive)
    else if (t == 2) mask = circle(v_uv);          // Spark
    else if (t == 3) mask = circle(v_uv) * 0.6;   // Smoke (soft)
    else if (t == 4) mask = circle(v_uv);          // Electric
    else if (t == 5) {                              // DigitalPixel (square)
        mask = step(0.1, v_uv.x) * step(0.1, v_uv.y) *
               step(v_uv.x, 0.9) * step(v_uv.y, 0.9);
    }
    else if (t == 6) mask = ring(v_uv);            // Hexagon ring
    else if (t == 7) mask = circle(v_uv);          // LightTrail
    else             mask = circle(v_uv);          // EnergyOrb

    // Glow types use additive blend (handled in main loop blend mode)
    f_color = vec4(v_color.rgb, v_color.a * mask);
}
"""


# ─── Particle Type Constants ───────────────────────────────────────────────────

class PType:
    DUST        = 0
    GLOW        = 1
    SPARK       = 2
    SMOKE       = 3
    ELECTRIC    = 4
    DIGITAL     = 5
    HEXAGON     = 6
    LIGHT_TRAIL = 7
    ENERGY_ORB  = 8


# ─── Particle Engine ──────────────────────────────────────────────────────────

class ParticleEngine:
    """
    GPU Particle Engine with instanced rendering.
    Particle data lives in a numpy array; updated on CPU, rendered on GPU.
    """

    # Quad geometry (two triangles = strip)
    _QUAD = np.array([
        -0.5, -0.5,
         0.5, -0.5,
        -0.5,  0.5,
         0.5,  0.5,
    ], dtype=np.float32)

    def __init__(self, ctx: moderngl.Context, shader_mgr):
        self.ctx    = ctx
        self._count = 0
        self._pool  = np.zeros((MAX_PARTICLES, PARTICLE_STRIDE), dtype=PARTICLE_DTYPE)
        self._build_gl(shader_mgr)
        log.info(f"ParticleEngine ready — capacity {MAX_PARTICLES}")

    def _build_gl(self, shader_mgr):
        sm = shader_mgr
        self._prog = sm.load_src("particle", _PARTICLE_VERT, _PARTICLE_FRAG)

        # Quad VBO (static)
        self._quad_vbo = self.ctx.buffer(self._QUAD)

        # Instance VBO (dynamic, updated every frame)
        self._inst_vbo = self.ctx.buffer(
            reserve=MAX_PARTICLES * PARTICLE_STRIDE * 4,   # 4 bytes/float
            dynamic=True
        )

        # VAO: quad corner (per-vertex) + all instance attrs
        stride = PARTICLE_STRIDE * 4   # bytes
        self._vao = self.ctx.vertex_array(
            self._prog,
            [
                (self._quad_vbo, "2f",    "in_corner"),
                (self._inst_vbo, "2f 2f f f f f 4f f f /i",
                 "i_pos", "i_vel", "i_life", "i_max_life",
                 "i_size", "i_type", "i_color", "i_angle", "i_spin"),
            ]
        )

    # ── Emitter API ───────────────────────────────────────────────────────────

    def emit(self,
             x: float, y: float,
             count: int = 16,
             ptype: int = PType.GLOW,
             color: tuple = (0.0, 0.75, 1.0, 1.0),
             speed: float = 120.0,
             spread: float = math.pi,
             life_range: tuple = (0.4, 1.2),
             size_range: tuple = (4.0, 12.0),
             direction: float = -math.pi / 2):  # upward default
        """
        Emit `count` particles from position (x, y).
        """
        n = min(count, MAX_PARTICLES - self._count)
        if n <= 0:
            return

        for i in range(n):
            slot = self._find_slot()
            if slot < 0:
                break
            ang   = direction + (random.random() - 0.5) * spread * 2.0
            spd   = speed * (0.5 + random.random() * 0.5)
            life  = random.uniform(*life_range)
            size  = random.uniform(*size_range)
            spin  = (random.random() - 0.5) * math.pi * 4.0

            p = self._pool[slot]
            p[0]  = x + random.uniform(-4, 4)
            p[1]  = y + random.uniform(-4, 4)
            p[2]  = math.cos(ang) * spd
            p[3]  = math.sin(ang) * spd
            p[4]  = life
            p[5]  = life
            p[6]  = size
            p[7]  = float(ptype)
            p[8]  = color[0]
            p[9]  = color[1]
            p[10] = color[2]
            p[11] = color[3]
            p[12] = random.uniform(0, math.pi * 2)
            p[13] = spin
            self._count = max(self._count, slot + 1)

    def emit_ring(self, cx: float, cy: float, radius: float,
                  count: int = 32, **kwargs):
        """Emit particles along a ring circumference."""
        for i in range(count):
            a = 2 * math.pi * i / count
            x = cx + math.cos(a) * radius
            y = cy + math.sin(a) * radius
            # Outward direction
            self.emit(x, y, count=1, direction=a, spread=0.3, **kwargs)

    def emit_burst(self, cx: float, cy: float, **kwargs):
        """Isotropic burst in all directions."""
        self.emit(cx, cy, spread=math.pi, direction=0, **kwargs)

    def _find_slot(self) -> int:
        """Find a dead particle slot (life <= 0)."""
        for i in range(MAX_PARTICLES):
            if self._pool[i][4] <= 0.0:
                return i
        return -1

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, dt: float):
        """Integrate all live particles."""
        alive = self._pool[:, 4] > 0.0
        if not np.any(alive):
            self._count = 0
            return

        # Physics
        self._pool[alive, 2] += PARTICLE_GRAVITY.x * dt * 60.0
        self._pool[alive, 3] += PARTICLE_GRAVITY.y * dt * 60.0
        # Drag
        drag = 0.985
        self._pool[alive, 2] *= drag
        self._pool[alive, 3] *= drag
        # Integrate position
        self._pool[alive, 0] += self._pool[alive, 2] * dt
        self._pool[alive, 1] += self._pool[alive, 3] * dt
        # Age
        self._pool[alive, 4] -= dt
        # Kill out-of-bounds
        oob = (
            (self._pool[:, 0] < -100) | (self._pool[:, 0] > WINDOW_W + 100) |
            (self._pool[:, 1] < -100) | (self._pool[:, 1] > WINDOW_H + 100)
        )
        self._pool[oob, 4] = 0.0

        self._count = int(np.sum(self._pool[:, 4] > 0.0))

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self, t: float):
        if self._count == 0:
            return

        live_mask = self._pool[:, 4] > 0.0
        live_data = self._pool[live_mask]
        if len(live_data) == 0:
            return

        # Upload instance data
        self._inst_vbo.write(live_data.astype(np.float32).tobytes())

        # Additive blend for glow/spark types, normal for others
        self.ctx.blend_equation = moderngl.FUNC_ADD
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE   # additive
        self.ctx.disable(moderngl.DEPTH_TEST)

        self._vao.render(moderngl.TRIANGLE_STRIP, instances=len(live_data))

        # Restore
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    @property
    def alive_count(self) -> int:
        return self._count

    def release(self):
        self._quad_vbo.release()
        self._inst_vbo.release()
        self._vao.release()

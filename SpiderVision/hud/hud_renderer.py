"""
SpiderVision — Main Holographic HUD Renderer
Renders the central hologram: rings, hexagon, radar, crosshair,
grid, particle ring, animated lines — all via GLSL.
"""
import math
import time
import numpy as np
import moderngl
import glm

from utils.logger import log
from utils.math_utils import circle_points, ortho_matrix
from config.settings import (
    WINDOW_W, WINDOW_H,
    HUD_CENTER_X, HUD_CENTER_Y, HUD_OUTER_R, HUD_INNER_R, HUD_RING_W,
    RADAR_SPEED, HEX_SPEED, PULSE_PERIOD,
    C_RED, C_BLUE, C_WHITE
)


# ─── HUD Shaders ─────────────────────────────────────────────────────────────

_HUD_VERT = """
#version 410 core
in  vec2 in_pos;
uniform mat4 u_proj;
uniform mat4 u_model;
out vec2 v_pos;   // local position for fragment effects
void main() {
    vec4 world   = u_model * vec4(in_pos, 0.0, 1.0);
    gl_Position  = u_proj  * world;
    v_pos        = in_pos;
}
"""

_RING_FRAG = """
#version 410 core
in vec2 v_pos;
uniform vec4  u_color;
uniform float u_time;
uniform float u_inner_r;
uniform float u_outer_r;
uniform float u_dash_n;    // number of dashes (0 = solid)
uniform float u_rot;       // rotation offset
out vec4 f_color;

void main() {
    float d = length(v_pos);
    if (d < u_inner_r || d > u_outer_r) discard;

    float t  = (atan(v_pos.y, v_pos.x) + u_rot) / (2.0 * 3.14159);
    t        = fract(t);

    float alpha = u_color.a;

    // Dash pattern
    if (u_dash_n > 0.0) {
        float dash = fract(t * u_dash_n);
        if (dash > 0.55) discard;
    }

    // Pulse glow
    float pulse = 0.6 + 0.4 * sin(u_time * 3.0 + d * 0.05);

    // Edge soften
    float edge = 1.0 - abs(d - (u_inner_r + u_outer_r) * 0.5) /
                       ((u_outer_r - u_inner_r) * 0.5);
    edge = smoothstep(0.0, 0.3, edge);

    f_color = vec4(u_color.rgb * pulse, alpha * edge);
}
"""

_LINE_VERT = """
#version 410 core
in vec2 in_pos;
uniform mat4 u_proj;
out vec2 v_pos;
void main() {
    gl_Position = u_proj * vec4(in_pos, 0.0, 1.0);
    v_pos = in_pos;
}
"""

_LINE_FRAG = """
#version 410 core
in vec2  v_pos;
uniform vec4  u_color;
uniform float u_time;
uniform float u_glow;
out vec4 f_color;
void main() {
    float flicker = 0.85 + 0.15 * sin(u_time * 7.3 + v_pos.x * 0.02);
    vec3  glow    = u_color.rgb * u_glow;
    f_color = vec4(u_color.rgb * flicker + glow, u_color.a);
}
"""

_RADAR_FRAG = """
#version 410 core
in vec2 v_pos;
uniform vec4  u_color;
uniform float u_sweep_angle;   // current sweep angle in radians
uniform float u_time;
uniform float u_radius;
out vec4 f_color;

void main() {
    float d = length(v_pos);
    if (d > u_radius) discard;

    float angle = atan(v_pos.y, v_pos.x);

    // Sweep trail: how far behind the sweep head is this pixel?
    float diff  = mod(angle - u_sweep_angle + 6.28318, 6.28318);
    float trail = max(0.0, 1.0 - diff / (3.14159 * 0.7));

    if (trail < 0.01) discard;

    // Radar grid circles
    float circle = fract(d / (u_radius / 4.0));
    float ring   = smoothstep(0.05, 0.0, abs(circle - 1.0)) * 0.15;

    float alpha = u_color.a * trail * (0.6 + ring);
    f_color = vec4(u_color.rgb, alpha);
}
"""

_GRID_FRAG = """
#version 410 core
in vec2 v_pos;
uniform vec4  u_color;
uniform float u_time;
uniform vec2  u_center;
out vec4 f_color;

void main() {
    vec2  p     = v_pos - u_center;
    // Perspective grid (simulated)
    float y_off = p.y;
    float scale = max(0.1, (600.0 + y_off) / 600.0);

    // Grid lines
    float gx = abs(fract(p.x / (60.0 * scale) - 0.5) - 0.5) / fwidth(p.x / (60.0 * scale));
    float gy = abs(fract(y_off / 40.0 - 0.5) - 0.5) / fwidth(y_off / 40.0);
    float grid = 1.0 - min(min(gx, gy), 1.0);

    // Fade toward horizon
    float fade = smoothstep(0.0, 0.4, scale) * smoothstep(1.0, 0.4, scale);

    // Animated data pulse
    float pulse = step(0.995, sin(u_time * 2.0 - y_off * 0.02));

    f_color = vec4(u_color.rgb, u_color.a * grid * fade * 0.5 + pulse * 0.3);
}
"""

_HEX_FRAG = """
#version 410 core
in vec2 v_pos;
uniform vec4  u_color;
uniform float u_time;
uniform float u_size;
out vec4 f_color;

float hex_dist(vec2 p) {
    p = abs(p);
    return max(dot(p, normalize(vec2(1.0, 1.732))), p.x);
}

void main() {
    float d  = hex_dist(v_pos / u_size);
    float edge = abs(d - 1.0);
    float alpha = smoothstep(0.05, 0.0, edge) * u_color.a;
    float pulse = 0.7 + 0.3 * sin(u_time * 4.0);
    f_color = vec4(u_color.rgb * pulse, alpha);
}
"""


class HUDRenderer:
    """
    Renders the complete holographic HUD overlay using GLSL shaders.
    Designed to render on top of the camera feed (post-processed scene).
    """

    def __init__(self, ctx: moderngl.Context, shader_mgr):
        self.ctx   = ctx
        self.sm    = shader_mgr
        self.cx    = WINDOW_W * HUD_CENTER_X
        self.cy    = WINDOW_H * HUD_CENTER_Y
        self._spider_mode   = False
        self._mode_transition = 0.0   # 0..1
        self._build_progs()
        self._build_geo()
        self._proj = ortho_matrix(WINDOW_W, WINDOW_H)
        # Animated data lines state
        import random
        self._lines = [
            self._new_line(random.random())
            for _ in range(16)
        ]

    def _new_line(self, phase: float) -> dict:
        import random
        return {
            "x":     random.uniform(self.cx - 380, self.cx + 380),
            "y":     random.uniform(self.cy - 220, self.cy + 220),
            "len":   random.uniform(40, 130),
            "speed": random.uniform(0.3, 1.2),
            "phase": phase,
            "alpha": random.uniform(0.2, 0.7),
        }

    # ── GL Setup ──────────────────────────────────────────────────────────────

    def _build_progs(self):
        sm = self.sm
        self._prog_ring  = sm.load_src("hud_ring",  _HUD_VERT, _RING_FRAG)
        self._prog_line  = sm.load_src("hud_line",  _LINE_VERT, _LINE_FRAG)
        self._prog_radar = sm.load_src("hud_radar", _HUD_VERT, _RADAR_FRAG)
        self._prog_grid  = sm.load_src("hud_grid",  _LINE_VERT, _GRID_FRAG)
        self._prog_hex   = sm.load_src("hud_hex",   _HUD_VERT, _HEX_FRAG)

    def _build_geo(self):
        """Build all static geometry VBOs."""
        # Full-quad for radar / grid (large)
        W, H = float(WINDOW_W), float(WINDOW_H)
        quad_v = np.array([
            0.0, 0.0,
            W,   0.0,
            0.0, H,
            W,   H,
        ], dtype=np.float32)
        self._quad_vbo = self.ctx.buffer(quad_v)

        # Ring geometry (unit circle disk approximation)
        n  = 256
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
        cx, cy = self.cx, self.cy
        ring_v = np.empty(n * 2, dtype=np.float32)
        ring_v[0::2] = np.cos(angles) * HUD_OUTER_R
        ring_v[1::2] = np.sin(angles) * HUD_OUTER_R
        self._ring_vbo = self.ctx.buffer(ring_v)
        self._ring_n   = n

        # Line VBO (dynamic, 2 vertices = 1 line)
        self._dyn_line_vbo = self.ctx.buffer(
            reserve=4 * 2 * 4, dynamic=True   # 2 pts * 2 floats * 4 bytes
        )

        # Crosshair lines (4 segments)
        CH = 40  # half length
        GAP = 20
        cross_v = np.array([
            cx - CH, cy,   cx - GAP, cy,   # left
            cx + GAP, cy,  cx + CH,  cy,   # right
            cx, cy - CH,   cx, cy - GAP,   # up
            cx, cy + GAP,  cx, cy + CH,    # down
        ], dtype=np.float32)
        self._cross_vbo = self.ctx.buffer(cross_v)
        self._cross_n   = 8

        # Target corner brackets (4 corners)
        br = HUD_OUTER_R + 30
        bl = 28
        corners = self._build_brackets(cx, cy, br, bl)
        self._bracket_vbo = self.ctx.buffer(corners)
        self._bracket_n   = len(corners) // 2

    def _build_brackets(self, cx, cy, r, l):
        """Build corner bracket lines around a circle of radius r."""
        pts = []
        offsets = [
            (-1, -1), (1, -1), (1, 1), (-1, 1)
        ]
        for ox, oy in offsets:
            bx, by = cx + ox * r, cy + oy * r
            # Horizontal arm
            pts += [bx, by, bx + ox * l * (-1), by]
            # Vertical arm
            pts += [bx, by, bx, by + oy * l * (-1)]
        return np.array(pts, dtype=np.float32)

    # ── VAO helpers ───────────────────────────────────────────────────────────

    def _make_ring_vao(self, prog):
        return self.ctx.vertex_array(
            prog, [(self._ring_vbo, "2f", "in_pos")]
        )

    def _make_quad_vao(self, prog):
        return self.ctx.vertex_array(
            prog, [(self._quad_vbo, "2f", "in_pos")]
        )

    def _proj_bytes(self):
        import numpy as np
        return np.array(self._proj, dtype=np.float32).reshape(4, 4).T.flatten().tobytes()

    # ── Render ────────────────────────────────────────────────────────────────

    def set_spider_mode(self, active: bool):
        self._spider_mode = active

    def render(self, t: float, dt: float):
        """Full HUD render pass. Called after scene is rendered."""
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Smooth mode transition
        target = 1.0 if self._spider_mode else 0.0
        self._mode_transition += (target - self._mode_transition) * dt * 2.5

        m = self._mode_transition
        # Spider mode colors
        ring_col = (
            m * C_RED.x + (1-m) * C_BLUE.x,
            m * C_RED.y + (1-m) * C_BLUE.y,
            m * C_RED.z + (1-m) * C_BLUE.z,
            0.9
        )
        ring_speed_mult = 1.0 + m * 2.5

        # ── Grid ──────────────────────────────────────────────────────────────
        self._render_grid(t)

        # ── Radar sweep ───────────────────────────────────────────────────────
        self._render_radar(t, ring_col, ring_speed_mult)

        # ── Outer ring ────────────────────────────────────────────────────────
        self._render_ring(t, HUD_OUTER_R - 2, HUD_OUTER_R + 2, ring_col,
                          dash_n=24, rot_speed=RADAR_SPEED * 0.5 * ring_speed_mult)

        # ── Inner ring ────────────────────────────────────────────────────────
        inner_col = (0.8, 0.9, 1.0, 0.7)
        self._render_ring(t, HUD_INNER_R - 2, HUD_INNER_R + 2, inner_col,
                          dash_n=12, rot_speed=-RADAR_SPEED * 0.8 * ring_speed_mult)

        # ── Hexagon ───────────────────────────────────────────────────────────
        self._render_hexagon(t, ring_speed_mult)

        # ── Energy pulse circles ───────────────────────────────────────────────
        self._render_energy_pulses(t, ring_col)

        # ── Crosshair ─────────────────────────────────────────────────────────
        self._render_crosshair(t, ring_col)

        # ── Corner brackets ───────────────────────────────────────────────────
        self._render_brackets(t, ring_col)

        # ── Animated data lines ───────────────────────────────────────────────
        self._render_data_lines(t, dt)

        self.ctx.enable(moderngl.DEPTH_TEST)

    # ── Sub-render methods ────────────────────────────────────────────────────

    def _set_proj(self, prog):
        proj_flat = np.array(self._proj, dtype=np.float32).reshape(4, 4).T.flatten()
        prog["u_proj"].write(proj_flat.tobytes())

    def _set_model(self, prog, cx, cy, angle=0.0, scale=1.0):
        m = glm.mat4(1.0)
        m = glm.translate(m, glm.vec3(cx, cy, 0.0))
        m = glm.rotate(m, angle, glm.vec3(0, 0, 1))
        m = glm.scale(m, glm.vec3(scale, scale, 1.0))
        flat = np.array(m, dtype=np.float32).reshape(4,4).T.flatten()
        prog["u_model"].write(flat.tobytes())

    def _render_ring(self, t, inner_r, outer_r, color, dash_n=0, rot_speed=1.0):
        prog = self._prog_ring
        vao  = self._make_ring_vao(prog)
        self._set_proj(prog)
        self._set_model(prog, self.cx, self.cy)
        prog["u_color"].value    = color
        prog["u_time"].value     = t
        prog["u_inner_r"].value  = inner_r
        prog["u_outer_r"].value  = outer_r
        prog["u_dash_n"].value   = float(dash_n)
        prog["u_rot"].value      = t * rot_speed * 2.0 * math.pi
        vao.render(moderngl.TRIANGLE_FAN)
        vao.release()

    def _render_radar(self, t, color, speed_mult):
        prog = self._prog_radar
        vao  = self._make_ring_vao(prog)
        sweep_angle = t * RADAR_SPEED * speed_mult * 2.0 * math.pi
        self._set_proj(prog)
        self._set_model(prog, self.cx, self.cy)
        prog["u_color"].value       = color
        prog["u_sweep_angle"].value = sweep_angle % (2 * math.pi)
        prog["u_radius"].value      = float(HUD_OUTER_R)
        vao.render(moderngl.TRIANGLE_FAN)
        vao.release()

    def _render_hexagon(self, t, speed_mult):
        prog = self._prog_hex
        # Unit hexagon (6 vertices + center for fan)
        n = 7
        verts = []
        for i in range(7):
            a = i * math.pi / 3.0
            verts += [math.cos(a), math.sin(a)]
        vbo = self.ctx.buffer(np.array(verts, dtype=np.float32))
        vao = self.ctx.vertex_array(prog, [(vbo, "2f", "in_pos")])
        self._set_proj(prog)
        angle = t * HEX_SPEED * speed_mult * 2.0 * math.pi
        self._set_model(prog, self.cx, self.cy, angle, 75.0)
        prog["u_color"].value = (0.0, 0.78, 1.0, 0.85)
        prog["u_time"].value  = t
        prog["u_size"].value  = 1.0
        vao.render(moderngl.LINE_LOOP)
        vao.release()
        vbo.release()

    def _render_energy_pulses(self, t, color):
        for i in range(3):
            phase  = (t / PULSE_PERIOD + i / 3.0) % 1.0
            radius = HUD_OUTER_R * 0.3 + phase * HUD_OUTER_R * 1.5
            alpha  = (1.0 - phase) * 0.4 * color[3]
            if alpha < 0.01:
                continue
            n = 64
            angles = np.linspace(0, 2 * math.pi, n + 1)
            verts  = np.column_stack([np.cos(angles) * radius, np.sin(angles) * radius]).astype(np.float32)
            vbo    = self.ctx.buffer(verts)
            vao    = self.ctx.vertex_array(self._prog_line, [(vbo, "2f", "in_pos")])
            self._set_proj(self._prog_line)
            self._prog_line["u_color"].value = (*color[:3], alpha)
            self._prog_line["u_time"].value  = t
            self._prog_line["u_glow"].value  = 0.2
            # Translate to center
            vao.render(moderngl.LINE_LOOP)
            vao.release()
            vbo.release()

    def _render_crosshair(self, t, color):
        prog = self._prog_line
        vao  = self.ctx.vertex_array(prog, [(self._cross_vbo, "2f", "in_pos")])
        self._set_proj(prog)
        alpha = 0.8 + 0.2 * math.sin(t * 5.0)
        prog["u_color"].value = (*color[:3], alpha)
        prog["u_time"].value  = t
        prog["u_glow"].value  = 0.3
        vao.render(moderngl.LINES)
        vao.release()

    def _render_brackets(self, t, color):
        prog = self._prog_line
        vao  = self.ctx.vertex_array(prog, [(self._bracket_vbo, "2f", "in_pos")])
        self._set_proj(prog)
        pulse = 0.6 + 0.4 * math.sin(t * 2.0)
        prog["u_color"].value = (*color[:3], pulse)
        prog["u_time"].value  = t
        prog["u_glow"].value  = 0.4
        vao.render(moderngl.LINES)
        vao.release()

    def _render_grid(self, t):
        prog = self._prog_grid
        vao  = self._make_quad_vao(prog)
        self._set_proj(prog)
        prog["u_color"].value  = (0.0, 0.6, 1.0, 0.25)
        prog["u_time"].value   = t
        prog["u_center"].value = (self.cx, self.cy)
        vao.render(moderngl.TRIANGLE_STRIP)
        vao.release()

    def _render_data_lines(self, t, dt):
        import random
        prog = self._prog_line
        self._set_proj(prog)
        for ln in self._lines:
            # Animate line scrolling
            ln["phase"] = (ln["phase"] + dt * ln["speed"]) % 1.0
            alive = math.sin(ln["phase"] * math.pi)
            if alive < 0.05:
                # Reset line
                ln["x"]     = random.uniform(self.cx - 380, self.cx + 380)
                ln["y"]     = random.uniform(self.cy - 220, self.cy + 220)
                ln["len"]   = random.uniform(30, 120)
                ln["alpha"] = random.uniform(0.15, 0.6)
                continue
            x1 = ln["x"]
            x2 = x1 + ln["len"] * alive
            y  = ln["y"]
            verts = np.array([x1, y, x2, y], dtype=np.float32)
            self._dyn_line_vbo.write(verts.tobytes())
            vao = self.ctx.vertex_array(prog, [(self._dyn_line_vbo, "2f", "in_pos")])
            prog["u_color"].value = (0.0, 0.75, 1.0, ln["alpha"] * alive)
            prog["u_time"].value  = t
            prog["u_glow"].value  = 0.1
            vao.render(moderngl.LINES)
            vao.release()

    def release(self):
        self._ring_vbo.release()
        self._quad_vbo.release()
        self._dyn_line_vbo.release()
        self._cross_vbo.release()
        self._bracket_vbo.release()

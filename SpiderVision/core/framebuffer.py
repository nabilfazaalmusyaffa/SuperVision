"""
SpiderVision — HDR Framebuffer & Post-Processing Chain
Manages all off-screen render targets and the full post-processing pipeline.
"""
import moderngl
import numpy as np
from config.settings import (
    WINDOW_W, WINDOW_H, MSAA_SAMPLES,
    BLOOM_THRESHOLD, BLOOM_INTENSITY, BLOOM_BLUR_PASSES,
    MOTION_BLUR_SAMPLES, MOTION_BLUR_STRENGTH,
    CHROMATIC_STRENGTH, VIGNETTE_STRENGTH, EXPOSURE, GAMMA
)
from core.shader_manager import ShaderManager
from utils.logger import log

# ─── Fullscreen Quad Shaders ──────────────────────────────────────────────────

_QUAD_VERT = """
#version 410 core
in vec2 in_position;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

_BLOOM_BRIGHT_FRAG = """
#version 410 core
uniform sampler2D u_hdr;
uniform float u_threshold;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec3 col = texture(u_hdr, v_uv).rgb;
    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
    float w = max(0.0, lum - u_threshold) / max(lum, 0.001);
    f_color = vec4(col * w, 1.0);
}
"""

_BLUR_H_FRAG = """
#version 410 core
uniform sampler2D u_tex;
uniform vec2 u_resolution;
in vec2 v_uv;
out vec4 f_color;
const float kernel[9] = float[](0.0093,0.028002,0.065984,0.121703,0.175713,
                                 0.121703,0.065984,0.028002,0.0093);
void main() {
    float px = 1.0 / u_resolution.x;
    vec3 c = vec3(0.0);
    for (int i = 0; i < 9; i++) {
        c += texture(u_tex, v_uv + vec2(float(i - 4) * px, 0.0)).rgb * kernel[i];
    }
    f_color = vec4(c, 1.0);
}
"""

_BLUR_V_FRAG = """
#version 410 core
uniform sampler2D u_tex;
uniform vec2 u_resolution;
in vec2 v_uv;
out vec4 f_color;
const float kernel[9] = float[](0.0093,0.028002,0.065984,0.121703,0.175713,
                                 0.121703,0.065984,0.028002,0.0093);
void main() {
    float py = 1.0 / u_resolution.y;
    vec3 c = vec3(0.0);
    for (int i = 0; i < 9; i++) {
        c += texture(u_tex, v_uv + vec2(0.0, float(i - 4) * py)).rgb * kernel[i];
    }
    f_color = vec4(c, 1.0);
}
"""

_COMPOSITE_FRAG = """
#version 410 core
uniform sampler2D u_hdr;
uniform sampler2D u_bloom;
uniform float u_bloom_intensity;
uniform float u_exposure;
uniform float u_gamma;
uniform float u_vignette;
uniform float u_chromatic;
uniform vec2  u_resolution;
in vec2 v_uv;
out vec4 f_color;

vec3 aces_tonemap(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

void main() {
    // Chromatic aberration
    float ca = u_chromatic;
    vec2 dir = v_uv - 0.5;
    float r = texture(u_hdr, v_uv + dir * ca).r;
    float g = texture(u_hdr, v_uv).g;
    float b2= texture(u_hdr, v_uv - dir * ca).b;
    vec3 hdr = vec3(r, g, b2);
    // Bloom
    vec3 bloom = texture(u_bloom, v_uv).rgb;
    hdr += bloom * u_bloom_intensity;
    // Exposure
    hdr *= u_exposure;
    // ACES tone mapping
    hdr = aces_tonemap(hdr);
    // Gamma correction
    hdr = pow(hdr, vec3(1.0 / u_gamma));
    // Vignette
    float dist = length(dir) * 1.41421;
    float vig  = 1.0 - smoothstep(0.5, 1.0, dist) * u_vignette;
    hdr *= vig;
    f_color = vec4(hdr, 1.0);
}
"""

_FXAA_FRAG = """
#version 410 core
uniform sampler2D u_tex;
uniform vec2 u_resolution;
in vec2 v_uv;
out vec4 f_color;
// Simplified FXAA
#define FXAA_REDUCE_MIN   (1.0/128.0)
#define FXAA_REDUCE_MUL   (1.0/8.0)
#define FXAA_SPAN_MAX     8.0
void main() {
    vec2 inv = 1.0 / u_resolution;
    vec3 luma = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(texture(u_tex, v_uv + vec2(-1,-1)*inv).rgb, luma);
    float lumaNE = dot(texture(u_tex, v_uv + vec2( 1,-1)*inv).rgb, luma);
    float lumaSW = dot(texture(u_tex, v_uv + vec2(-1, 1)*inv).rgb, luma);
    float lumaSE = dot(texture(u_tex, v_uv + vec2( 1, 1)*inv).rgb, luma);
    float lumaM  = dot(texture(u_tex, v_uv).rgb, luma);
    float lumaMin = min(lumaM, min(min(lumaNW,lumaNE), min(lumaSW,lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW,lumaNE), max(lumaSW,lumaSE)));
    vec2 dir;
    dir.x = -((lumaNW+lumaNE)-(lumaSW+lumaSE));
    dir.y =  ((lumaNW+lumaSW)-(lumaNE+lumaSE));
    float rcp = 1.0/(min(abs(dir.x),abs(dir.y)) + max((lumaNW+lumaNE+lumaSW+lumaSE)*0.25*FXAA_REDUCE_MUL, FXAA_REDUCE_MIN));
    dir = clamp(dir*rcp, vec2(-FXAA_SPAN_MAX), vec2(FXAA_SPAN_MAX)) * inv;
    vec3 A = 0.5*(texture(u_tex, v_uv+dir*( 1.0/3.0-0.5)).rgb + texture(u_tex, v_uv+dir*(2.0/3.0-0.5)).rgb);
    vec3 B = A*0.5 + 0.25*(texture(u_tex, v_uv+dir*(-0.5)).rgb + texture(u_tex, v_uv+dir*0.5).rgb);
    float lumaB = dot(B, luma);
    f_color = vec4((lumaB < lumaMin || lumaB > lumaMax) ? A : B, 1.0);
}
"""


class Framebuffer:
    """
    Manages HDR FBO chain:
      scene FBO → bloom extraction → blur passes → composite → FXAA → screen
    """

    def __init__(self, ctx: moderngl.Context, shader_mgr: ShaderManager):
        self.ctx = ctx
        self.sm  = shader_mgr
        self.w   = WINDOW_W
        self.h   = WINDOW_H
        self._build_fbos()
        self._build_quad()
        self._build_shaders()
        log.info("Framebuffer chain ready")

    # ── FBO Creation ──────────────────────────────────────────────────────────

    def _make_fbo(self, w: int, h: int, components: int = 4,
                  dtype: str = "f2") -> tuple[moderngl.Framebuffer, moderngl.Texture]:
        tex = self.ctx.texture((w, h), components, dtype=dtype)
        tex.filter  = moderngl.LINEAR, moderngl.LINEAR
        tex.repeat_x = False
        tex.repeat_y = False
        fbo = self.ctx.framebuffer(color_attachments=[tex])
        return fbo, tex

    def _build_fbos(self):
        w, h = self.w, self.h
        hw, hh = w // 2, h // 2
        # Main HDR scene buffer (float16)
        self.scene_fbo,   self.scene_tex   = self._make_fbo(w,  h)
        # Bloom buffers (half resolution for performance)
        self.bloom_a_fbo, self.bloom_a_tex = self._make_fbo(hw, hh, 4, "f2")
        self.bloom_b_fbo, self.bloom_b_tex = self._make_fbo(hw, hh, 4, "f2")
        # Composite output
        self.comp_fbo,    self.comp_tex    = self._make_fbo(w,  h)
        # Depth attachment for scene (shared)
        self.depth_buf = self.ctx.depth_renderbuffer((w, h))

    def _build_quad(self):
        """Fullscreen quad [-1,1] with UV [0,1]."""
        verts = np.array([
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
            -1.0,  1.0,  0.0, 1.0,
             1.0,  1.0,  1.0, 1.0,
        ], dtype=np.float32)
        vbo = self.ctx.buffer(verts)
        self._quad_vbo = vbo

    def _build_shaders(self):
        sm = self.sm
        self._prog_bright   = sm.load_src("bloom_bright",  _QUAD_VERT, _BLOOM_BRIGHT_FRAG)
        self._prog_blur_h   = sm.load_src("blur_h",        _QUAD_VERT, _BLUR_H_FRAG)
        self._prog_blur_v   = sm.load_src("blur_v",        _QUAD_VERT, _BLUR_V_FRAG)
        self._prog_composite= sm.load_src("composite",     _QUAD_VERT, _COMPOSITE_FRAG)
        self._prog_fxaa     = sm.load_src("fxaa",          _QUAD_VERT, _FXAA_FRAG)

        def make_vao(prog):
            return self.ctx.vertex_array(
                prog, [(self._quad_vbo, "2f 2f", "in_position", "in_uv")]
            )

        self._vao_bright    = make_vao(self._prog_bright)
        self._vao_blur_h    = make_vao(self._prog_blur_h)
        self._vao_blur_v    = make_vao(self._prog_blur_v)
        self._vao_composite = make_vao(self._prog_composite)
        self._vao_fxaa      = make_vao(self._prog_fxaa)

    # ── Render API ────────────────────────────────────────────────────────────

    def bind_scene(self):
        """Bind scene FBO as render target."""
        self.scene_fbo.use()
        self.ctx.viewport = (0, 0, self.w, self.h)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

    def process(self):
        """Run full post-processing chain → write to screen."""
        hw, hh = self.w // 2, self.h // 2
        sm = ShaderManager.set_uniform

        # 1. Bright pass (bloom extraction)
        self.bloom_a_fbo.use()
        self.ctx.viewport = (0, 0, hw, hh)
        self.scene_tex.use(location=0)
        sm(self._prog_bright, "u_hdr",       0)
        sm(self._prog_bright, "u_threshold", BLOOM_THRESHOLD)
        self._vao_bright.render(moderngl.TRIANGLE_STRIP)

        # 2. Blur passes (ping-pong)
        for i in range(BLOOM_BLUR_PASSES):
            # Horizontal
            self.bloom_b_fbo.use()
            self.bloom_a_tex.use(location=0)
            sm(self._prog_blur_h, "u_tex",        0)
            sm(self._prog_blur_h, "u_resolution", (float(hw), float(hh)))
            self._vao_blur_h.render(moderngl.TRIANGLE_STRIP)
            # Vertical
            self.bloom_a_fbo.use()
            self.bloom_b_tex.use(location=0)
            sm(self._prog_blur_v, "u_tex",        0)
            sm(self._prog_blur_v, "u_resolution", (float(hw), float(hh)))
            self._vao_blur_v.render(moderngl.TRIANGLE_STRIP)

        # 3. Composite (HDR + bloom + tone mapping + vignette + chromatic)
        self.comp_fbo.use()
        self.ctx.viewport = (0, 0, self.w, self.h)
        self.scene_tex.use(location=0)
        self.bloom_a_tex.use(location=1)
        sm(self._prog_composite, "u_hdr",           0)
        sm(self._prog_composite, "u_bloom",          1)
        sm(self._prog_composite, "u_bloom_intensity",BLOOM_INTENSITY)
        sm(self._prog_composite, "u_exposure",       EXPOSURE)
        sm(self._prog_composite, "u_gamma",          GAMMA)
        sm(self._prog_composite, "u_vignette",       VIGNETTE_STRENGTH)
        sm(self._prog_composite, "u_chromatic",      CHROMATIC_STRENGTH)
        sm(self._prog_composite, "u_resolution",     (float(self.w), float(self.h)))
        self._vao_composite.render(moderngl.TRIANGLE_STRIP)

        # 4. FXAA → screen
        self.ctx.screen.use()
        
        # Use physical framebuffer dimensions if provided (for Retina)
        fw = getattr(self, "fb_width", self.w)
        fh = getattr(self, "fb_height", self.h)
        self.ctx.viewport = (0, 0, fw, fh)
        
        self.comp_tex.use(location=0)
        sm(self._prog_fxaa, "u_tex",        0)
        sm(self._prog_fxaa, "u_resolution", (float(self.w), float(self.h)))
        self._vao_fxaa.render(moderngl.TRIANGLE_STRIP)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def release(self):
        for attr in ("scene_fbo","scene_tex","bloom_a_fbo","bloom_a_tex",
                     "bloom_b_fbo","bloom_b_tex","comp_fbo","comp_tex","depth_buf"):
            obj = getattr(self, attr, None)
            if obj:
                obj.release()
        log.info("Framebuffer released")

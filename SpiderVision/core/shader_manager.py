"""
SpiderVision — Shader Manager
Load, compile, cache, and optionally hot-reload GLSL shader programs.
"""
import os
import moderngl
from utils.logger import log
from config.settings import SHADERS_DIR


class ShaderManager:
    """Loads and caches ModernGL programs from GLSL source files."""

    def __init__(self, ctx: moderngl.Context):
        self.ctx      = ctx
        self._cache: dict[str, moderngl.Program] = {}
        self._mtimes: dict[str, float]           = {}

    # ── Loading ───────────────────────────────────────────────────────────────

    def _read(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _path(self, name: str) -> str:
        return os.path.join(SHADERS_DIR, name)

    def load(self, vert: str, frag: str, geom: str | None = None) -> moderngl.Program:
        """
        Load a shader program by filename stem.
        Names are relative to assets/shaders/.
        Cached by (vert, frag, geom) tuple key.
        """
        key = f"{vert}|{frag}|{geom}"
        if key in self._cache:
            return self._cache[key]
        return self._compile(key, vert, frag, geom)

    def _compile(self, key: str, vert: str, frag: str, geom: str | None) -> moderngl.Program:
        try:
            vp = self._path(vert)
            fp = self._path(frag)
            vs = self._read(vp)
            fs = self._read(fp)
            gs = self._read(self._path(geom)) if geom else None

            prog = self.ctx.program(
                vertex_shader   = vs,
                fragment_shader = fs,
                geometry_shader = gs,
            )
            self._cache[key] = prog
            log.info(f"Shader compiled: {vert} + {frag}" + (f" + {geom}" if geom else ""))
            return prog
        except Exception as e:
            log.error(f"Shader compilation failed [{vert},{frag}]: {e}")
            raise

    # ── Inline source shortcut (for procedural shaders) ───────────────────────

    def load_src(self, key: str, vert_src: str, frag_src: str,
                 geom_src: str | None = None) -> moderngl.Program:
        """Compile from raw GLSL strings directly (no file I/O)."""
        if key in self._cache:
            return self._cache[key]
        try:
            prog = self.ctx.program(
                vertex_shader   = vert_src,
                fragment_shader = frag_src,
                geometry_shader = geom_src,
            )
            self._cache[key] = prog
            log.debug(f"Inline shader compiled: {key}")
            return prog
        except Exception as e:
            log.error(f"Inline shader failed [{key}]: {e}")
            raise

    # ── Uniform helper ────────────────────────────────────────────────────────

    @staticmethod
    def set_uniform(prog: moderngl.Program, name: str, value):
        """Safely set a uniform, ignoring missing uniforms silently."""
        try:
            if name in prog:
                prog[name].value = value
        except Exception:
            pass

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def release(self):
        for prog in self._cache.values():
            prog.release()
        self._cache.clear()
        log.info("ShaderManager released")

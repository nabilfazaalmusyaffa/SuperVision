"""
SpiderVision — Core Renderer
ModernGL context creation via GLFW, main window management,
primary render loop orchestration.
"""
import glfw
import moderngl
import time
import numpy as np
from utils.logger import log
from config.settings import (
    WINDOW_TITLE, WINDOW_W, WINDOW_H, TARGET_FPS, FULLSCREEN,
    MSAA_SAMPLES, OPENGL_VERSION
)


class Renderer:
    """
    Central rendering context. Owns the GLFW window and ModernGL ctx.
    All rendering subsystems receive this object.
    """

    def __init__(self):
        self.width  = WINDOW_W
        self.height = WINDOW_H
        self.ctx    = None
        self.window = None
        self._frame_time = 1.0 / TARGET_FPS
        self._last_time  = 0.0
        self.delta_time  = 0.0
        self.elapsed     = 0.0
        self.frame_count = 0
        self.fps         = 0.0
        self._fps_counter    = 0
        self._fps_last_time  = 0.0
        self._init_glfw()
        self._init_moderngl()
        log.info(f"Renderer ready — OpenGL {self.ctx.version_code} | {WINDOW_W}×{WINDOW_H}")

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _init_glfw(self):
        if not glfw.init():
            raise RuntimeError("GLFW failed to initialise")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, OPENGL_VERSION[0])
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, OPENGL_VERSION[1])
        glfw.window_hint(glfw.OPENGL_PROFILE,        glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)   # macOS required
        glfw.window_hint(glfw.SAMPLES,               MSAA_SAMPLES)
        glfw.window_hint(glfw.DOUBLEBUFFER,           True)
        glfw.window_hint(glfw.RESIZABLE,              False)

        monitor = glfw.get_primary_monitor()
        if monitor is None:
            glfw.terminate()
            raise RuntimeError("No display monitor available; launch from a graphical desktop")
        window_w, window_h = WINDOW_W, WINDOW_H
        window_x, window_y = 0, 0
        if FULLSCREEN:
            glfw.window_hint(glfw.DECORATED, False)  # Borderless
            # Avoid glfw.get_video_mode: it segfaults in some macOS display sessions.
            try:
                area_x, area_y, area_w, area_h = glfw.get_monitor_workarea(monitor)
                if area_w > 0 and area_h > 0:
                    scale = min(1.0, area_w / WINDOW_W, area_h / WINDOW_H)
                    window_w = max(1, int(WINDOW_W * scale))
                    window_h = max(1, int(WINDOW_H * scale))
                    window_x, window_y = area_x, area_y
            except Exception as exc:
                log.warning(f"Monitor workarea unavailable; using configured window size: {exc}")
        self.window = glfw.create_window(window_w, window_h, WINDOW_TITLE, None, None)

        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")
        if FULLSCREEN:
            glfw.set_window_pos(self.window, window_x, window_y)

        # Update width/height to actual window size
        # We will keep self.width and self.height as WINDOW_W and WINDOW_H 
        # so internal rendering stays at 1920x1080 (resolution scaling).
        self.width = WINDOW_W
        self.height = WINDOW_H
        
        # Store physical framebuffer size for viewport mapping
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        self.fb_width = fb_w
        self.fb_height = fb_h
        log.info(f"Window fitted to {window_w}x{window_h}; framebuffer {fb_w}x{fb_h}")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)   # VSync on

        # Key callback
        glfw.set_key_callback(self.window, self._on_key)
        self._key_callbacks: list = []

    def _init_moderngl(self):
        self.ctx = moderngl.create_context()
        # Enable blending globally
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        # Enable depth test for 3D passes
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<="

    # ── Key Handling ───────────────────────────────────────────────────────────

    def _on_key(self, window, key, scancode, action, mods):
        for cb in self._key_callbacks:
            cb(key, action, mods)

    def add_key_callback(self, cb):
        self._key_callbacks.append(cb)

    # ── Frame Timing ───────────────────────────────────────────────────────────

    def begin_frame(self) -> bool:
        """
        Returns False if window should close.
        Updates delta_time, elapsed, fps.
        """
        if glfw.window_should_close(self.window):
            return False

        now = glfw.get_time()
        self.delta_time  = now - self._last_time
        self._last_time  = now
        self.elapsed     = now
        self.frame_count += 1

        # FPS calculation
        self._fps_counter += 1
        if now - self._fps_last_time >= 0.5:
            self.fps            = self._fps_counter / (now - self._fps_last_time)
            self._fps_counter   = 0
            self._fps_last_time = now

        # Clear default framebuffer
        self.ctx.screen.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        return True

    def end_frame(self):
        """Swap buffers and poll events."""
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    # ── Viewport ───────────────────────────────────────────────────────────────

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def use_screen(self):
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, self.fb_width, self.fb_height)

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def close(self):
        if self.ctx:
            self.ctx.release()
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()
        log.info("Renderer closed")

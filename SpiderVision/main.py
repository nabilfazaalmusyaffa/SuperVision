"""
SpiderVision HUD — Cinematic Edition
Main application entry point.

Architecture:
  Thread 1 (main)  : OpenGL rendering
  Thread 2         : Camera capture
  Thread 3         : MediaPipe inference (face + hand)
  Thread 4         : Performance monitoring

State machine: STARTUP → IDLE → SCANNING → TRACKING → SPIDER_MODE / MENU_OPEN / CAPTURE
"""
import sys
import os
import time
import math
import threading
import random
import queue
import numpy as np
import cv2
import glfw
import moderngl
import glm

# ── Ensure project root in path ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    WINDOW_W, WINDOW_H, TARGET_FPS, STARTUP_DURATION, AppState,
    CAMERA_W, CAMERA_H, GESTURE_HOLD_SEC
)
from core.renderer       import Renderer
from core.framebuffer    import Framebuffer
from core.shader_manager import ShaderManager
from core.camera         import CameraStream
from tracking.hand_tracker import HandTracker
from tracking.face_tracker import FaceTracker, FaceData
from tracking.gesture     import Gesture, HandState, GestureLatch, web_aim, web_origin, is_retraction_pose
from vision.mask_detector import MaskDetector
from vision.sface_identity import SFaceIdentity
from vision.face_session import FaceSession
from hud.hud_renderer     import HUDRenderer
from hud.face_scanner     import FaceScannerHUD
from hud.performance_monitor import PerformanceMonitor
from hud.menu             import HologramMenu
from hud.vitals           import VitalsHUD
from hud.web_aim          import WebAimHUD
from hud.anatomy          import AnatomyHUD
from hud.war_mode         import WarModeHUD
from hud.location_map     import LocationMapHUD
from location.ip_area     import IPAreaProvider
from location.device      import ManualLocationProvider
from sound.voice_commands import VoiceCommandListener
from effects.particle_engine import ParticleEngine, PType
from effects.web_projectile  import WebProjectileSystem
from effects.camera_effect   import CameraEffect
from utils.sound_generator   import generate_all as gen_sounds
from utils.screenshot        import ScreenshotManager
from utils.logger            import log


# ─── Camera Texture Shader (fullscreen background) ────────────────────────────

_CAM_VERT = """
#version 410 core
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_CAM_FRAG = """
#version 410 core
uniform sampler2D u_cam;
uniform float u_brightness;
uniform float u_time;
uniform float u_spider_mode;  // 0..1
in  vec2 v_uv;
out vec4 f_color;

void main() {
    vec4 cam = texture(u_cam, v_uv);

    // Spider mode: tint red
    vec3 col = cam.rgb;
    col = mix(col, vec3(dot(col, vec3(0.3,0.59,0.11))) * vec3(1.0,0.1,0.1), u_spider_mode * 0.7);

    // Slight scanline effect, animating downwards
    float scanline = 1.0 - 0.04 * mod(gl_FragCoord.y - u_time * 20.0, 2.0);

    // CRT-style vignette (very subtle)
    vec2 uv2 = v_uv - 0.5;
    float vig = 1.0 - dot(uv2, uv2) * 0.6;

    col *= u_brightness * scanline * vig;
    f_color = vec4(col, 1.0);
}
"""

_FLASH_VERT = """
#version 410 core
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""
_FLASH_FRAG = """
#version 410 core
uniform float u_alpha;
out vec4 f_color;
void main() { f_color = vec4(1.0, 1.0, 1.0, u_alpha); }
"""

_HAND_VERT = """
#version 410 core
in vec2 in_pos;
uniform mat4 u_proj;
void main() {
    gl_Position = u_proj * vec4(in_pos, 0.0, 1.0);
}
"""
_HAND_FRAG = """
#version 410 core
uniform vec4 u_color;
out vec4 f_color;
void main() { f_color = u_color; }
"""

# Fullscreen quad verts
_FS_QUAD = np.array([
    -1.0, -1.0,  0.0, 1.0,
     1.0, -1.0,  1.0, 1.0,
    -1.0,  1.0,  0.0, 0.0,
     1.0,  1.0,  1.0, 0.0,
], dtype=np.float32)


class SpiderVisionApp:
    """Main application — owns all subsystems, drives the render loop."""

    # ── Init ──────────────────────────────────────────────────────────────────

    def __init__(self):
        log.info("=== Spider Vision HUD — Cinematic Edition ===")
        log.info("Initialising subsystems...")

        # 1. Generate sounds (if needed)
        gen_sounds()

        # 2. Renderer & GL
        self.renderer   = Renderer()
        self.ctx        = self.renderer.ctx
        self.shader_mgr = ShaderManager(self.ctx)
        self.fbo        = Framebuffer(self.ctx, self.shader_mgr)
        self.fbo.fb_width  = self.renderer.fb_width
        self.fbo.fb_height = self.renderer.fb_height

        # 3. Camera
        self.camera = CameraStream()
        self.camera.wait_for_first_frame(timeout=4.0)

        # 4. Tracking (in background thread)
        self.face_tracker = FaceTracker()
        self.hand_tracker = HandTracker()
        self.mask_detector = MaskDetector()
        self.identity = SFaceIdentity(os.path.join(os.path.dirname(__file__), ".face_identity.npz"))
        self.session = FaceSession()
        self._face_data: FaceData = FaceData()
        self._hands: list[HandState] = []
        self._track_lock = threading.Lock()
        self._track_running = True
        self._track_sequence = 0
        self._handled_track_sequence = 0
        self._track_thread = threading.Thread(
            target=self._track_loop, daemon=True, name="TrackingThread"
        )

        # 5. HUD systems
        self.hud      = HUDRenderer(self.ctx, self.shader_mgr)
        self.face_hud = FaceScannerHUD()
        self.perf_mon = PerformanceMonitor()
        self.menu     = HologramMenu()
        self.vitals   = VitalsHUD()
        self.web_aim_hud = WebAimHUD()
        self.anatomy  = AnatomyHUD()
        self.war_hud  = WarModeHUD()
        self.location_map = LocationMapHUD()
        self._map_visible = True
        self._map_consent_pending = False
        self._map_generation = 0
        self._map_lookup_running = False
        self._map_reply = None
        self._map_was_unlocked = False
        self.show_performance = True
        self.show_particles = True

        # 6. Effects
        self.particles   = ParticleEngine(self.ctx, self.shader_mgr)
        self.web_system  = WebProjectileSystem(self.ctx, self.shader_mgr)
        self.cam_effect  = CameraEffect()

        # 7. Audio
        self._init_audio()
        self.voice = VoiceCommandListener(self.audio)
        self.voice.start()  # microphone recognition stays off the render thread

        # 8. Screenshot
        self.screenshot  = ScreenshotManager(self.audio)

        # 9. GL resources for camera background
        self._build_cam_gl()

        # 10. State
        self.state          = AppState.STARTUP
        self._startup_t     = 0.0
        self._spider_mode   = False
        self._war_mode      = False  # opt-in visual effect; never inferred from a face
        self._prev_gestures: dict = {}   # side → last gesture + hold time
        self._web_latch = GestureLatch(hold_seconds=0.08)
        self._last_web_shot: dict[str, float] = {}
        self._exit_after_speech = None
        self._pending_audio_mute = False
        self._voice_notice = ''
        self._last_web_side = None
        self._hand_gl_ready = False
        self._proj_2d       = None

        # 11. Key callbacks
        self.renderer.add_key_callback(self._on_key)

        # 12. Start tracking thread
        self._track_thread.start()

        # 13. Play startup sound
        if self.audio:
            self.audio.play("startup")

        log.info("All subsystems ready. Starting render loop.")

    # ── Audio ─────────────────────────────────────────────────────────────────

    def _init_audio(self):
        try:
            from sound.audio_manager import AudioManager
            self.audio = AudioManager()
        except Exception as e:
            log.warning(f"Audio init failed: {e}. Running without audio.")
            self.audio = None

    # ── Camera GL ─────────────────────────────────────────────────────────────

    def _build_cam_gl(self):
        self._cam_prog = self.shader_mgr.load_src("cam_bg", _CAM_VERT, _CAM_FRAG)
        self._flash_prog = self.shader_mgr.load_src("flash", _FLASH_VERT, _FLASH_FRAG)
        self._hand_prog = self.shader_mgr.load_src("hand_hud", _HAND_VERT, _HAND_FRAG)

        vbo = self.ctx.buffer(_FS_QUAD)
        self._cam_vbo = vbo
        self._cam_vao = self.ctx.vertex_array(
            self._cam_prog, [(vbo, "2f 2f", "in_pos", "in_uv")]
        )
        self._flash_vao = self.ctx.vertex_array(
            self._flash_prog, [(vbo, "2f 8x", "in_pos")]
        )

        # Camera texture (RGB, dynamically updated)
        self._cam_tex = self.ctx.texture((CAMERA_W, CAMERA_H), 3)
        self._cam_tex.filter = moderngl.LINEAR, moderngl.LINEAR

        # Dynamic VBO for hand skeleton lines
        self._hand_vbo = self.ctx.buffer(reserve=256 * 2 * 4, dynamic=True)
        self._hand_vao = self.ctx.vertex_array(
            self._hand_prog, [(self._hand_vbo, "2f", "in_pos")]
        )

    # ── Tracking Thread ───────────────────────────────────────────────────────

    def _track_loop(self):
        """Background thread: runs MediaPipe on each camera frame."""
        while self._track_running:
            rgb = self.camera.read_rgb()
            if rgb is None:
                time.sleep(0.01)
                continue

            face_data = self.face_tracker.process(rgb, 0.033)
            hands     = self.hand_tracker.process(rgb)
            # Let the reticle follow fresh fingertips while face matching runs.
            # Gesture actions still wait for the final authorized sequence below.
            with self._track_lock:
                self._hands = hands

            # Mask detection (every 3rd frame is enough)
            bgr = self.camera.read()
            if bgr is not None:
                self.mask_detector.detect(bgr, face_data)
                box = None
                if face_data.detected:
                    h, w = bgr.shape[:2]
                    box = (face_data.box_x * w, face_data.box_y * h,
                           face_data.box_w * w, face_data.box_h * h)
                self._apply_session(face_data, self.identity.process(bgr, box))
                face_data.enroll_progress = f"{self.identity.sample_count}/{self.identity.samples_required}"
            else:
                self._apply_session(face_data, "NO_FACE")

            with self._track_lock:
                self._face_data = face_data
                self._track_sequence += 1

    def _apply_session(self, face_data: FaceData, raw_status: str):
        face_data.identity_status = raw_status
        face_data.session_unlocked = self.session.observe(raw_status)

    # ── Key Handler ───────────────────────────────────────────────────────────

    def _on_key(self, key, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
            glfw.set_window_should_close(self.renderer.window, True)
        elif key == glfw.KEY_BACKSPACE and self.anatomy.mode != 'CLOSED':
            self.anatomy.close()
        elif (key == glfw.KEY_SPACE and self._face_data.session_unlocked
              and self.anatomy.mode == 'CLOSED'):
            # Manual web launch (debug)
            self._fire_web("Right")
        elif key == glfw.KEY_S and self._face_data.session_unlocked:
            self.screenshot.start()
        elif key == glfw.KEY_E:
            self.identity.start_enrollment()
            log.info("Face enrollment started (local-only model)")
        elif key == glfw.KEY_D:
            self.identity.delete_enrollment()
            log.info("Face enrollment deleted")
        elif key == glfw.KEY_M and self._face_data.session_unlocked:
            if self.menu.is_open:
                self.menu.hide()
            else:
                self.menu.show()
        elif (glfw.KEY_1 <= key <= glfw.KEY_9 and self.menu.is_open and
              self._face_data.session_unlocked):
            from config.settings import MENU_ITEMS
            self._menu_action(MENU_ITEMS[key - glfw.KEY_1])

    # ── Web Projectile ────────────────────────────────────────────────────────

    def _fire_web(self, side: str, hand=None):
        now = time.monotonic()
        if now - self._last_web_shot.get(side, -float('inf')) < 0.7:
            return
        # Find hand
        hand = hand or self.hand_tracker.get_by_side(side)
        if hand:
            aim_landmarks = hand.aim_landmarks or hand.landmarks
            origin = web_origin(aim_landmarks)
            direction = web_aim(aim_landmarks)
            self.web_system.launch(origin, direction, side,
                                   war_mode=getattr(self, '_war_mode', False))
            if self.audio:
                self.audio.play("web")
            self.cam_effect.shake_web()
            # Burst particles from palm
            self.particles.emit_burst(
                origin.x, origin.y,
                count=32, ptype=PType.SPARK,
                color=(0.4, 0.85, 1.0, 1.0),
                speed=180, life_range=(0.2, 0.6)
            )
        else:
            # No hand tracked, fire from screen edge
            if side == "Right":
                origin = glm.vec2(WINDOW_W * 0.75, WINDOW_H * 0.6)
                direction = glm.vec2(1.0, -0.3)
            else:
                origin = glm.vec2(WINDOW_W * 0.25, WINDOW_H * 0.6)
                direction = glm.vec2(-1.0, -0.3)
            self.web_system.launch(origin, direction, side,
                                   war_mode=getattr(self, '_war_mode', False))
            if self.audio:
                self.audio.play("web")
            self.cam_effect.shake_web()
        self.web_aim_hud.flash(origin)
        self._last_web_shot[side] = now

    # ── Gesture Processing ────────────────────────────────────────────────────

    def _menu_action(self, item):
        if item == "AUDIO" and self.audio:
            self.audio.muted = not self.audio.muted
            if self.audio.muted:
                self.audio.stop_all()
        elif item == "PERFORMANCE":
            self.show_performance = not self.show_performance
        elif item == "WEB AIM":
            self.web_aim_hud.enabled = not self.web_aim_hud.enabled
        elif item == "BODY ATLAS":
            self.anatomy.open('BODY')
        elif item == "BRAIN ATLAS":
            self.anatomy.open('BRAIN')
        elif item == "EFFECTS":
            self.show_particles = not self.show_particles
        elif item == "SCREENSHOT":
            self.screenshot.start()
        elif item == "ENROLL FACE":
            self.identity.start_enrollment()
        elif item == "EXIT":
            glfw.set_window_should_close(self.renderer.window, True)
        self.menu.hide()

    def _process_voice_commands(self, session_unlocked: bool):
        """Dispatch voice commands while the verified session is active."""
        if not session_unlocked and self._war_mode:
            self._war_mode = False
            self.web_system.clear_war_webs()
        for _ in range(4):
            command = self.voice.pop_command()
            if command is None:
                break
            if not session_unlocked:
                self._voice_notice = 'VERIFY FACE FIRST'
                continue
            self._voice_notice = ''
            if command == "MENU":
                self.menu.show()
                if self.audio:
                    self.audio.play("menu")
                    self.audio.speak("Menu open, Mister. Say Jarvis and a menu name.")
            elif command == "CLOSE":
                if self.anatomy.mode != 'CLOSED':
                    self.anatomy.close()
                    self.menu.hide()
                    if self.audio and not self.audio.muted:
                        self.audio.speak("Anatomy view closed, Mister.")
                else:
                    self.menu.hide()
                    if self.audio:
                        self.audio.speak("Menu closed, Mister.")
            elif command == "HELP":
                if self.audio:
                    self.audio.speak("I can help with web aim, war mode, anatomy, and an approximate map. Say Jarvis war mode on, body, describe, map, or status.")
            elif command == "WAR MODE ON":
                if self.anatomy.mode != 'CLOSED':
                    if self.audio and not self.audio.muted:
                        self.audio.speak("Close anatomy first, then activate war mode.")
                elif not self._war_mode:
                    self._war_mode = True
                    self.menu.hide()
                    if self.audio and not self.audio.muted:
                        self.audio.speak("War mode activated. Visual simulation only. Make a fist to retract the web.")
            elif command == "WAR MODE OFF":
                if self._war_mode:
                    self._war_mode = False
                    self.web_system.clear_war_webs()
                if self.audio and not self.audio.muted:
                    self.audio.speak("War mode off.")
            elif command == "STATUS":
                if self.audio and not self.audio.muted:
                    mode = self.anatomy.mode.lower()
                    self.audio.speak("Spider Vision online. " +
                                     (f"{mode} atlas open." if mode != 'closed' else "Anatomy atlas closed."))
            elif command == "DESCRIBE":
                if self.audio and not self.audio.muted:
                    description = self.anatomy.describe_selected()
                    if description:
                        self.audio.speak(f"{self.anatomy.selected.title()}. {description} Educational illustration only.")
                    else:
                        self.audio.speak("Point to a part in the body or brain atlas first.")
            elif command == "SHOW MAP":
                self._map_visible = True
                configured = self._use_manual_map_area()
                if self.audio and not self.audio.muted:
                    if configured is True:
                        self.audio.speak("Chosen area shown. This is not GPS.")
                    elif configured is False:
                        self.audio.speak("Manual map coordinates are invalid.")
                    else:
                        self.audio.speak("Map open. Say Jarvis locate me for an approximate area.")
            elif command == "HIDE MAP":
                self._map_visible = False
                self._map_consent_pending = False
                self._map_generation += 1
                self._map_lookup_running = False
                self._map_reply = None
                self.location_map.set_location(None, None)
                if self.audio and not self.audio.muted:
                    self.audio.speak("Map hidden. Active area cleared. Cached map tiles may remain on this device.")
            elif command == "LOCATE":
                self._map_visible = True
                self._map_consent_pending = True
                if self.audio and not self.audio.muted:
                    self.audio.speak("I can estimate your area using your public IP. The external ipapi dot co service will see that IP. Say Jarvis allow location to continue.")
            elif command == "ALLOW LOCATION":
                if self._map_consent_pending:
                    self._map_consent_pending = False
                    self._start_map_lookup()
                elif self.audio and not self.audio.muted:
                    self.audio.speak("Say Jarvis locate me first.")
            elif command in ("BODY", "BRAIN"):
                self.menu.hide()
                self.anatomy.open(command)
                if self.audio and not self.audio.muted:
                    self.audio.speak(f"{command.title()} atlas open. Educational illustration only.")
            elif command == "CLOSE ANATOMY":
                self.anatomy.close()
                if self.audio and not self.audio.muted:
                    self.audio.speak("Anatomy view closed, Mister.")
            elif command in ("ZOOM IN", "ZOOM OUT"):
                if self.anatomy.mode == 'CLOSED':
                    if self.audio and not self.audio.muted:
                        self.audio.speak("Open the body or brain atlas first.")
                else:
                    factor = 1.25 if command == "ZOOM IN" else 0.8
                    self.anatomy.zoom = max(1.0, min(2.5, self.anatomy.zoom * factor))
                    if self.audio and not self.audio.muted:
                        self.audio.speak("Zoomed in." if factor > 1 else "Zoomed out.")
            elif command == "AUDIO" and self.audio and not self.audio.muted:
                self.menu.hide()
                self.audio.speak("Audio off, Mister.")
                self._pending_audio_mute = True
            elif command == "EXIT":
                self.menu.hide()
                if self.audio and not self.audio.muted:
                    self.audio.speak("Closing Spider Vision. Goodbye, Mister.")
                now = time.monotonic()
                self._exit_after_speech = (now + 0.25, now + 10.0)
            elif command in ("AUDIO", "PERFORMANCE", "EFFECTS", "SCREENSHOT", "ENROLL FACE", "WEB AIM"):
                self._menu_action(command)
                if self.audio and not self.audio.muted:
                    replies = {
                        "AUDIO": "Audio on, Mister.",
                        "PERFORMANCE": "Performance display on." if self.show_performance else "Performance display off.",
                        "EFFECTS": "Effects on." if self.show_particles else "Effects off.",
                        "SCREENSHOT": "Screenshot in three seconds, Mister.",
                        "ENROLL FACE": "Face enrollment started. Look into the camera.",
                        "WEB AIM": "Web aim on, Mister." if self.web_aim_hud.enabled else "Web aim off, Mister.",
                    }
                    self.audio.speak(replies[command])

    def _finish_voice_replies(self):
        speaking = bool(self.audio and self.audio.is_speaking)
        if self._pending_audio_mute and not speaking:
            self.audio.muted = True
            self.audio.stop_all()
            self._pending_audio_mute = False
        if self._exit_after_speech is not None:
            ready, deadline = self._exit_after_speech
            now = time.monotonic()
            if now >= ready and (not speaking or now >= deadline):
                glfw.set_window_should_close(self.renderer.window, True)
                self._exit_after_speech = None

    def _voice_status_text(self, session_unlocked: bool) -> str:
        if self._voice_notice and not session_unlocked:
            return self._voice_notice
        if self.audio and self.audio.is_speaking:
            return 'JARVIS SPEAKING - WAIT'
        if self.voice.available:
            return self.voice.last_feedback or 'MIC READY'
        return 'MIC OFF - PRESS M' if self.voice.error else 'MIC LOADING'

    def _process_gestures(self, hands: list[HandState], dt: float):
        if self.anatomy.mode != 'CLOSED':
            self._web_latch.forget_missing(set())
            self._prev_gestures.clear()
            return
        seen_sides = set()
        now = time.monotonic()

        for hand in hands:
            side = hand.handedness
            seen_sides.add(side)
            g = hand.gesture

            prev = self._prev_gestures.get(side)
            # Raw landmarks react before the skeleton's EMA has caught up.
            # Synthetic/recovery states without landmarks use the old classifier.
            raw = hand.aim_landmarks
            retract_pose = (is_retraction_pose(raw) if isinstance(raw, (list, tuple))
                            and len(raw) >= 21 else g == Gesture.FIST)
            previous_retract_pose = (prev.get('retract_pose', prev['gesture'] == Gesture.FIST)
                                     if prev else False)
            if prev is None or g != prev["gesture"]:
                prev = {"gesture": g, "since": now, "fired": False}
            prev['retract_pose'] = retract_pose
            self._prev_gestures[side] = prev
            hold = now - prev["since"]

            # One pull per transition, even if the smoothed gesture changes later.
            if getattr(self, '_war_mode', False) and retract_pose and not previous_retract_pose:
                if self.web_system.retract(side):
                    log.debug('War web retracted from %s hand', side)
                    if self.audio:
                        self.audio.play('retract')
                    self.particles.emit_burst(
                        hand.palm_pos.x, hand.palm_pos.y,
                        count=18, ptype=PType.SPARK,
                        color=(1.0, 0.55, 0.25, 0.9),
                        speed=70, life_range=(0.12, 0.4))

            # Do not fire a second web while the slow classifier still reports
            # WEB_SHOT/PINCH after the raw hand has closed.
            cooling = now - self._last_web_shot.get(side, -float('inf')) < 0.7
            if retract_pose and g in (Gesture.WEB_SHOT, Gesture.PINCH):
                self._web_latch.update(side, Gesture.NONE, now)
                web_triggered = False
            else:
                web_triggered = (False if cooling and g in (Gesture.WEB_SHOT, Gesture.PINCH)
                                 else self._web_latch.update(side, g, now))
            if g in (Gesture.WEB_SHOT, Gesture.PINCH) and web_triggered:
                self._fire_web(side, hand)
                # Palm glow effect
                self.particles.emit_burst(
                    hand.palm_pos.x, hand.palm_pos.y,
                    count=16, ptype=PType.ELECTRIC,
                    color=(0.2, 0.6, 1.0, 0.9),
                    speed=60, life_range=(0.1, 0.35)
                )

            # Menu and screenshots are voice/keyboard controlled. Hand gestures
            # are reserved for the web shot so ordinary motion does not open UI.
            # ── Continuous particle from palm (web shot charge) ────────────────
            if g == Gesture.WEB_SHOT:
                charge = min(1.0, hold / GESTURE_HOLD_SEC)
                self.particles.emit(
                    hand.palm_pos.x, hand.palm_pos.y,
                    count=2, ptype=PType.ELECTRIC,
                    color=(0.2 + 0.8 * charge, 0.5, 1.0, 0.6),
                    speed=30, spread=math.pi, life_range=(0.1, 0.25)
                )

        # Clear gestures for hands that disappeared
        for side in list(self._prev_gestures.keys()):
            if side not in seen_sides:
                del self._prev_gestures[side]
        self._web_latch.forget_missing(seen_sides)

    # ── Camera Texture Upload ─────────────────────────────────────────────────

    def _upload_cam_frame(self, bgr: np.ndarray):
        """Upload BGR camera frame to OpenGL texture (resize to CAMERA_W×H)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (CAMERA_H, CAMERA_W):
            rgb = cv2.resize(rgb, (CAMERA_W, CAMERA_H))
        self._cam_tex.write(rgb.tobytes())

    def _render_web_aim(self, bgr, hands, session_unlocked):
        if (self.state == AppState.STARTUP or not session_unlocked or
                self.anatomy.mode != 'CLOSED'):
            return bgr
        return self.web_aim_hud.render(bgr, hands)

    def _render_anatomy(self, bgr, hands, session_unlocked):
        if not session_unlocked:
            if self.anatomy.mode != 'CLOSED':
                self.anatomy.close()
            return bgr
        if self.state == AppState.STARTUP or self.anatomy.mode == 'CLOSED':
            return bgr
        return self.anatomy.render(bgr, hands)

    def _render_war_mode(self, bgr, session_unlocked, t):
        if (not self._war_mode or not session_unlocked or
                self.state == AppState.STARTUP or self.anatomy.mode != 'CLOSED'):
            return bgr
        return self.war_hud.render(bgr, t)

    def _render_location_map(self, bgr, session_unlocked):
        if not session_unlocked:
            if getattr(self, '_map_was_unlocked', False):
                self._map_generation += 1
                self.location_map.set_location(None, None)
                self._map_consent_pending = False
                self._map_lookup_running = False
                self._map_reply = None
                self._map_was_unlocked = False
            return bgr
        self._map_was_unlocked = True
        if self.state == AppState.STARTUP or not self._map_visible or self.menu.is_open:
            return bgr
        return self.location_map.render(bgr)

    def _start_map_lookup(self):
        """Only after an explicit second voice confirmation; never block render."""
        if self._map_lookup_running:
            if self.audio and not self.audio.muted:
                self.audio.speak("Location request already in progress.")
            return
        self._map_generation += 1
        generation = self._map_generation
        self._map_lookup_running = True
        self._map_reply = None
        threading.Thread(target=self._fetch_map_area, args=(generation,),
                         name='SpiderVisionLocation', daemon=True).start()

    def _use_manual_map_area(self):
        """Use coordinates deliberately supplied at launch, or leave map off."""
        supplied = os.environ.get('SPIDERVISION_MAP_COORDS', '').strip()
        if not supplied:
            return None
        try:
            values = supplied.split(',')
            if len(values) != 2:
                raise ValueError('expected latitude,longitude')
            result = ManualLocationProvider(float(values[0]), float(values[1])).request_location()
            self.location_map.set_location(result['lat'], result['lon'], 'USER CHOSEN')
            return True
        except (TypeError, ValueError):
            return False

    def _fetch_map_area(self, generation):
        try:
            result = IPAreaProvider().request_location(opt_in=True)
            if generation != self._map_generation:
                return
            if result['error'] is None:
                self.location_map.set_location(result['lat'], result['lon'], 'IP ESTIMATE',
                                               accuracy_m=result['accuracy_m'])
                message = "Approximate network area shown. This is not GPS."
            else:
                message = "Area unavailable. Check your internet connection. This map is not GPS."
            self._map_reply = (generation, message)
        except Exception:
            if generation == self._map_generation:
                self._map_reply = (generation, "Area unavailable. The map is not GPS.")
        finally:
            if generation == self._map_generation:
                self._map_lookup_running = False

    def _finish_map_lookup(self, session_unlocked):
        reply = self._map_reply
        self._map_reply = None
        if (reply and reply[0] == self._map_generation and session_unlocked
                and self._map_visible and self.audio and not self.audio.muted):
            self.audio.speak(reply[1])

    def _combat_hud_visible(self, session_unlocked):
        return not (session_unlocked and self.anatomy.mode != 'CLOSED')

    # ── 2D Projection Matrix ──────────────────────────────────────────────────

    def _get_proj2d(self):
        if self._proj_2d is None:
            from utils.math_utils import ortho_matrix
            m = ortho_matrix(WINDOW_W, WINDOW_H)
            self._proj_2d = np.array(m, dtype=np.float32).reshape(4, 4).T.flatten()
        return self._proj_2d

    # ── Hand Skeleton (OpenGL) ────────────────────────────────────────────────

    def _render_hand_gl(self, hands: list[HandState], t: float):
        """Render hand skeleton using OpenGL lines."""
        if not hands:
            return
        from mediapipe.python.solutions import hands as mp_h
        CONNECTIONS = mp_h.HAND_CONNECTIONS

        prog = self._hand_prog
        prog["u_proj"].write(self._get_proj2d().tobytes())

        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.ctx.disable(moderngl.DEPTH_TEST)

        for hand in hands:
            lm    = hand.landmarks
            pulse = 0.6 + 0.4 * math.sin(t * 6.0)
            is_web = hand.gesture == Gesture.WEB_SHOT

            # Skeleton lines
            line_pts = []
            for a_idx, b_idx in CONNECTIONS:
                if a_idx < len(lm) and b_idx < len(lm):
                    ax = lm[a_idx][0] * WINDOW_W
                    ay = lm[a_idx][1] * WINDOW_H
                    bx = lm[b_idx][0] * WINDOW_W
                    by = lm[b_idx][1] * WINDOW_H
                    line_pts += [ax, ay, bx, by]

            if line_pts:
                pts = np.array(line_pts, dtype=np.float32)
                self._hand_vbo.orphan(pts.nbytes)
                self._hand_vbo.write(pts.tobytes())
                vao = self.ctx.vertex_array(
                    self._hand_prog, [(self._hand_vbo, "2f", "in_pos")]
                )
                if is_web:
                    prog["u_color"].value = (0.2, 0.6, 1.0, pulse * 0.9)
                else:
                    prog["u_color"].value = (0.0, 0.78, 1.0, 0.6)
                vao.render(moderngl.LINES)
                vao.release()

            # Joint dots as particles (fingertips)
            tips = [4, 8, 12, 16, 20]
            for i, tip_idx in enumerate(tips):
                if tip_idx < len(lm):
                    px = lm[tip_idx][0] * WINDOW_W
                    py = lm[tip_idx][1] * WINDOW_H
                    if is_web and i in (1, 4):  # index + pinky
                        self.particles.emit(
                            px, py, count=1, ptype=PType.ELECTRIC,
                            color=(0.0, 0.8, 1.0, 0.7),
                            speed=15, life_range=(0.05, 0.15)
                        )

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    # ── Spider Mode Particles ─────────────────────────────────────────────────

    def _emit_spider_ambient(self, t: float, dt: float):
        """Continuous ambient particles in spider mode."""
        if random.random() < dt * 30:
            side = random.choice([-1, 1])
            x = WINDOW_W * 0.5 + side * random.uniform(100, 400)
            y = random.uniform(WINDOW_H * 0.3, WINDOW_H * 0.8)
            self.particles.emit(x, y,
                count=1, ptype=PType.GLOW,
                color=(1.0, 0.1, 0.1, 0.5),
                speed=40, life_range=(0.5, 1.5),
                spread=math.pi * 0.3, direction=-math.pi / 2
            )

    # ── Startup Animation Particles ───────────────────────────────────────────

    def _emit_startup_particles(self, frac: float):
        cx, cy = WINDOW_W / 2, WINDOW_H / 2
        if random.random() < 0.4:
            r = HUDRenderer if False else 220
            self.particles.emit_ring(cx, cy, r,
                count=2, ptype=PType.GLOW,
                color=(0.0, 0.75, 1.0, 0.7),
                speed=60, life_range=(0.4, 1.0),
                size_range=(3, 8)
            )

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def run(self):
        _HUD_OUTER_R = 220   # startup particle ring radius

        while self.renderer.begin_frame():
            t  = self.renderer.elapsed
            dt = min(self.renderer.delta_time, 0.05)   # cap at 50ms
            effect_dt = min(self.renderer.delta_time, 0.25)  # real time, even when frames are slow

            # ── Read tracking data ─────────────────────────────────────────────
            with self._track_lock:
                face_data = self._face_data
                hands     = list(self._hands)
                sequence = self._track_sequence

            self.menu.update(dt)  # no fingertip selection: voice operates the menu
            self._process_voice_commands(face_data.session_unlocked)
            self._finish_voice_replies()
            self._finish_map_lookup(face_data.session_unlocked)
            combat_hud = self._combat_hud_visible(face_data.session_unlocked)

            # ── Update state machine ───────────────────────────────────────────
            if self.state == AppState.STARTUP:
                self._startup_t += dt
                self._emit_startup_particles(self._startup_t / STARTUP_DURATION)
                if self._startup_t >= STARTUP_DURATION:
                    self.state = AppState.IDLE
                    log.info("Startup complete → IDLE")

            if self.state != AppState.STARTUP:
                # Detect spider mode
                prev_spider = self._spider_mode
                if self.mask_detector.is_active and face_data.session_unlocked:
                    self._spider_mode = True
                    if not prev_spider and self.audio:
                        self.audio.play("lock")
                else:
                    self._spider_mode = False

                self.hud.set_spider_mode(self._spider_mode or self._war_mode)

                # Spider ambient particles
                if self._spider_mode:
                    self._emit_spider_ambient(t, dt)

                # Process gestures
                if sequence != self._handled_track_sequence:
                    # Unknown faces may be shown, but cannot operate the HUD.
                    active_hands = hands if face_data.session_unlocked else []
                    self._process_gestures(active_hands, dt)
                    if not face_data.session_unlocked:
                        self.menu.hide()
                    self._handled_track_sequence = sequence

            # ── Update effects ────────────────────────────────────────────────
            self.particles.update(dt)
            self.web_system.update(effect_dt)
            self.cam_effect.update(dt)

            # Screenshot update
            capture_now = self.screenshot.update(dt)

            # ── Read camera frame ─────────────────────────────────────────────
            bgr = self.camera.read()
            if bgr is not None:
                # Apply face scanner overlay (OpenCV)
                if self.state != AppState.STARTUP:
                    bgr = self.face_hud.render(bgr, face_data, dt, self.audio)
                    if face_data.session_unlocked:
                        bgr = self.vitals.render(bgr, t)
                if combat_hud:
                    bgr = self.web_system.render_overlay(bgr)
                bgr = self._render_war_mode(bgr, face_data.session_unlocked, t)
                bgr = self._render_web_aim(bgr, hands, face_data.session_unlocked)
                bgr = self._render_anatomy(bgr, hands, face_data.session_unlocked)

                # Apply performance monitor
                self.perf_mon.face_confidence = face_data.confidence
                self.perf_mon.hand_confidence = (
                    hands[0].confidence if hands else 0.0
                )
                self.perf_mon.camera_latency  = self.camera.latency_ms
                if self.show_performance:
                    bgr = self.perf_mon.render(bgr, self.renderer.fps,
                                               self.camera.latency_ms)

                # Menu overlay
                if self.menu.is_open:
                    bgr = self.menu.render(bgr, t)
                bgr = self._render_location_map(bgr, face_data.session_unlocked)

                # Voice is optional if the microphone permission/device is unavailable.
                voice_text = self._voice_status_text(face_data.session_unlocked)
                cv2.putText(bgr, voice_text, (CAMERA_W - 290, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)

                # Screenshot countdown overlay
                if self.screenshot.is_active:
                    bgr = self.screenshot.render_countdown(bgr, t)

                # Upload to GL texture
                self._upload_cam_frame(bgr)

            # ── GL Rendering ─────────────────────────────────────────────────
            # Bind scene FBO
            self.fbo.bind_scene()

            # 1. Camera background quad
            self.ctx.disable(moderngl.DEPTH_TEST)
            self._cam_tex.use(location=0)
            spider_frac = 1.0 if (self._spider_mode or self._war_mode) and face_data.session_unlocked else 0.0
            self._cam_prog["u_cam"].value          = 0
            self._cam_prog["u_brightness"].value   = 0.75
            self._cam_prog["u_time"].value         = t
            self._cam_prog["u_spider_mode"].value  = spider_frac
            self._cam_vao.render(moderngl.TRIANGLE_STRIP)

            # 2. HUD hologram (center rings, hexagon, radar, etc.)
            self.ctx.enable(moderngl.BLEND)
            if combat_hud and (self.state != AppState.STARTUP or self._startup_t > 0.5):
                startup_alpha = min(1.0, (self._startup_t - 0.5) / 1.5) if self.state == AppState.STARTUP else 1.0
                self.hud.render(t, dt)

            # 3. Hand skeleton (GL)
            if combat_hud and hands and self.state != AppState.STARTUP:
                self._render_hand_gl(hands, t)

            # 4. Particles
            if combat_hud and self.show_particles:
                self.particles.render(t)

            # 5. Web projectiles
            if combat_hud:
                self.web_system.render(t, self.particles)

            # 6. White flash
            if combat_hud and self.cam_effect.flash_alpha > 0.01:
                self._flash_prog["u_alpha"].value = self.cam_effect.flash_alpha
                self._flash_vao.render(moderngl.TRIANGLE_STRIP)

            # ── Post-processing (bloom, FXAA, tonemap) ──────────────────────
            self.fbo.process()

            if capture_now:
                finished = self.screenshot.capture_screen(
                    self.ctx, self.renderer.fb_width, self.renderer.fb_height
                )
                self.screenshot.save(finished, self.audio)
                self.cam_effect.flash_white(0.4)
                self.cam_effect.shake_capture()

            self.renderer.end_frame()

        self._shutdown()

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _shutdown(self):
        log.info("Shutting down...")
        self._track_running = False
        self.voice.stop()
        self.perf_mon.stop()
        self.camera.stop()
        if self.audio:
            self.audio.shutdown()
        self.face_tracker.close()
        self.hand_tracker.close()
        self.particles.release()
        self.web_system.release()
        self.fbo.release()
        self.shader_mgr.release()
        self.renderer.close()
        log.info("Spider Vision HUD — Goodbye.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        app = SpiderVisionApp()
        app.run()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — exiting")
    except Exception as e:
        log.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

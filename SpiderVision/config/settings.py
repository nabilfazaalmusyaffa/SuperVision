"""
SpiderVision HUD — Configuration & Settings
All constants, colors, thresholds, paths centralized here.
"""
import os
import glm

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR      = os.path.join(BASE_DIR, "assets")
SHADERS_DIR     = os.path.join(ASSETS_DIR, "shaders")
SOUNDS_DIR      = os.path.join(ASSETS_DIR, "sounds")
FONTS_DIR       = os.path.join(ASSETS_DIR, "fonts")
TEXTURES_DIR    = os.path.join(ASSETS_DIR, "textures")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# ─── Window ───────────────────────────────────────────────────────────────────
WINDOW_TITLE    = "Spider Vision HUD — Cinematic Edition"
WINDOW_W        = 1920
WINDOW_H        = 1080
TARGET_FPS      = 60
FULLSCREEN      = True           # Set True for production
MSAA_SAMPLES    = 4
OPENGL_VERSION  = (4, 1)

# ─── Camera ───────────────────────────────────────────────────────────────────
CAMERA_SOURCE   = 0             # 0, 1, 2 or IP string
CAMERA_W        = 1280
CAMERA_H        = 720
CAMERA_FPS      = 60

# ─── Color Palette (RGBA float 0..1) ─────────────────────────────────────────
# Neon Red
C_RED           = glm.vec4(1.0,  0.10, 0.10, 1.0)
C_RED_DIM       = glm.vec4(0.6,  0.05, 0.05, 0.7)
C_RED_GLOW      = glm.vec4(1.0,  0.20, 0.15, 0.4)
# Neon Blue
C_BLUE          = glm.vec4(0.0,  0.75, 1.0,  1.0)
C_BLUE_DIM      = glm.vec4(0.0,  0.40, 0.65, 0.6)
C_BLUE_GLOW     = glm.vec4(0.05, 0.60, 1.0,  0.35)
# White / Near-white
C_WHITE         = glm.vec4(0.95, 0.98, 1.0,  1.0)
C_WHITE_DIM     = glm.vec4(0.70, 0.75, 0.82, 0.5)
# Black / Background
C_BLACK         = glm.vec4(0.0,  0.0,  0.0,  1.0)
C_BG            = glm.vec4(0.01, 0.01, 0.02, 0.95)
# Accent Orange (warning)
C_ORANGE        = glm.vec4(1.0,  0.55, 0.05, 1.0)
# Green (confidence bar)
C_GREEN         = glm.vec4(0.10, 1.0,  0.35, 1.0)

# ─── HUD Layout ───────────────────────────────────────────────────────────────
HUD_CENTER_X    = 0.5           # fraction of window width
HUD_CENTER_Y    = 0.5           # fraction of window height
HUD_OUTER_R     = 180           # pixels, outer ring radius
HUD_INNER_R     = 130           # pixels, inner ring radius
HUD_RING_W      = 3.0           # ring stroke width

RADAR_SPEED     = 0.8           # rotations/second
HEX_SPEED       = 0.4           # rotations/second (inner hexagon)
PULSE_PERIOD    = 2.0           # seconds

# ─── Face Scanner ─────────────────────────────────────────────────────────────
FACE_SCAN_SPEED         = 2.0   # beam sweeps per second
FACE_TRACK_SMOOTH       = 0.25  # EMA alpha (lower = smoother)
FACE_CONFIDENCE_MIN     = 0.5

# ─── Hand Tracking ────────────────────────────────────────────────────────────
HAND_SMOOTH             = 0.30  # EMA alpha
HAND_MIN_CONFIDENCE     = 0.5
HAND_MIN_PRESENCE       = 0.5
MAX_HANDS               = 2

# ─── Gesture Thresholds ───────────────────────────────────────────────────────
FINGER_RATIO            = 1.12  # tip/pip distance ratio to consider "open"
THUMB_RATIO             = 1.15
# Spider-Man web gesture: index + pinky + thumb OUT, middle + ring CLOSED
GESTURE_HOLD_SEC        = 0.20  # seconds to hold before trigger

# ─── Spider Mode ──────────────────────────────────────────────────────────────
MASK_RED_RATIO          = 0.35  # fraction of face area that must be red
MASK_BLACK_RATIO        = 0.15  # fraction that is near-black

# ─── Particle Engine ──────────────────────────────────────────────────────────
MAX_PARTICLES           = 20_000
PARTICLE_BATCH_SIZE     = 512   # emitted per burst
PARTICLE_GRAVITY        = glm.vec3(0.0, -0.15, 0.0)

# ─── Web Projectile ───────────────────────────────────────────────────────────
WEB_SPEED               = 3.5   # units/second
WEB_RINGS               = 12    # radial resolution
WEB_SPOKES              = 16    # angular resolution
WEB_LIFETIME            = 2.5   # seconds
WEB_ELASTIC             = 0.7

# ─── Post Processing ──────────────────────────────────────────────────────────
BLOOM_THRESHOLD         = 0.75
BLOOM_INTENSITY         = 1.4
BLOOM_BLUR_PASSES       = 6
FXAA_ENABLED            = True
MOTION_BLUR_SAMPLES     = 8
MOTION_BLUR_STRENGTH    = 0.15
CHROMATIC_STRENGTH      = 0.004
VIGNETTE_STRENGTH       = 0.45
EXPOSURE                = 1.0
GAMMA                   = 2.2

# ─── Screen Shake ─────────────────────────────────────────────────────────────
SHAKE_DECAY             = 8.0   # per second damping
SHAKE_WEB_INTENSITY     = 0.015
SHAKE_CAPTURE_INTENSITY = 0.008

# ─── Audio ────────────────────────────────────────────────────────────────────
AUDIO_FREQ              = 48000
AUDIO_CHANNELS          = 2
AUDIO_BUFFER            = 1024
MASTER_VOLUME           = 0.75

# ─── TTS ──────────────────────────────────────────────────────────────────────
TTS_VOICE               = "en-US-GuyNeural"  # male synthetic assistant voice; not a Jarvis impersonation
TTS_RATE                = "+5%"               # slightly faster
TTS_VOLUME              = "+0%"

# ─── Performance Monitor ─────────────────────────────────────────────────────
PERF_UPDATE_INTERVAL    = 0.5   # seconds between perf reads
PERF_X                  = 12    # pixels from left
PERF_Y                  = 16    # pixels from top

# ─── Screenshot ───────────────────────────────────────────────────────────────
SCREENSHOT_COUNTDOWN    = 3     # seconds
SCREENSHOT_PREFIX       = "SpiderVision"

# ─── State Machine ────────────────────────────────────────────────────────────
class AppState:
    STARTUP       = "STARTUP"
    IDLE          = "IDLE"
    SCANNING      = "SCANNING"
    TRACKING      = "TRACKING"
    SPIDER_MODE   = "SPIDER_MODE"
    WEB_LAUNCH    = "WEB_LAUNCH"
    MENU_OPEN     = "MENU_OPEN"
    CAPTURE       = "CAPTURE"
    SHUTDOWN      = "SHUTDOWN"

STARTUP_DURATION = 2.5   # seconds for startup animation

# ─── Menu Items ───────────────────────────────────────────────────────────────
MENU_ITEMS = ["AUDIO", "PERFORMANCE", "EFFECTS", "SCREENSHOT", "ENROLL FACE", "WEB AIM", "BODY ATLAS", "BRAIN ATLAS", "EXIT"]

"""
SpiderVision — Post Processor (thin wrapper)
The core HDR post-processing chain lives in core/framebuffer.py.
This module provides any runtime-adjustable post-process effects.
"""
import math
import moderngl
import numpy as np
from config.settings import (
    MOTION_BLUR_STRENGTH, MOTION_BLUR_SAMPLES,
    CHROMATIC_STRENGTH
)
from utils.logger import log


class PostProcessor:
    """
    Optional extra post-process effects layered on top of the
    Framebuffer chain. Currently unused in main pipeline but
    available for future expansion (depth of field, etc).
    """

    def __init__(self, ctx: moderngl.Context, shader_mgr):
        self.ctx        = ctx
        self.sm         = shader_mgr
        self.motion_blur_strength = MOTION_BLUR_STRENGTH
        self.chromatic_strength   = CHROMATIC_STRENGTH
        log.info("PostProcessor ready")

    def set_motion_blur(self, strength: float):
        self.motion_blur_strength = max(0.0, min(1.0, strength))

    def release(self):
        pass

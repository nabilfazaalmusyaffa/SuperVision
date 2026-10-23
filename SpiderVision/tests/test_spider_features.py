import tempfile
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import glm
from unittest.mock import patch

from effects.web_projectile import _build_web_mesh, WebProjectile
from tracking.gesture import Gesture, classify_gesture, GestureLatch, web_aim, web_origin, detect_gesture
from vision.face_identity import FaceIdentity
from vision.sface_identity import SFaceIdentity
from hud.face_scanner import FaceScannerHUD
from tracking.face_tracker import FaceData
from utils.screenshot import ScreenshotManager
from hud.menu import HologramMenu
from sound.audio_manager import AudioManager
from core.renderer import Renderer


class WebTests(unittest.TestCase):
    def test_hand_position_is_mirrored_once_not_twice(self):
        # CameraStream already flips the frame before MediaPipe sees it.
        from tracking.hand_tracker import HandTracker
        # Assert coordinate conversion helper (no MediaPipe model required).
        self.assertEqual(HandTracker.screen_x(0.25, 'Right'), 480)
        self.assertEqual(HandTracker.screen_x(0.25, 'Left'), 480)

    def test_web_shot_requires_fresh_hold_and_aims_with_index_finger(self):
        latch = GestureLatch(hold_seconds=0.2)
        self.assertFalse(latch.update('Right', Gesture.WEB_SHOT, 1.0))
        self.assertFalse(latch.update('Right', Gesture.WEB_SHOT, 1.1))
        self.assertTrue(latch.update('Right', Gesture.WEB_SHOT, 1.21))
        self.assertFalse(latch.update('Right', Gesture.WEB_SHOT, 2.0))
        latch.update('Right', Gesture.NONE, 2.1)
        self.assertFalse(latch.update('Right', Gesture.WEB_SHOT, 2.2))
        self.assertTrue(latch.update('Right', Gesture.WEB_SHOT, 2.42))
        landmarks = [(0.5, 0.5, 0)] * 21
        landmarks[5] = (0.5, 0.5, 0)
        landmarks[8] = (0.7, 0.3, 0)
        direction = web_aim(landmarks)
        self.assertGreater(direction.x, 0)
        self.assertLess(direction.y, 0)
        origin = web_origin(landmarks)
        self.assertEqual(origin.x, 0.7 * 1920)
        self.assertEqual(origin.y, 0.3 * 1080)

    def test_thumb_index_pinching_fires_alternative_gesture(self):
        landmarks = [(0.5, 0.5, 0)] * 21
        landmarks[4] = (0.54, 0.41, 0)
        landmarks[8] = (0.55, 0.41, 0)
        self.assertEqual(detect_gesture(landmarks), Gesture.PINCH)

    def test_mesh_is_pairs_of_real_lines_and_reaches_web_tip(self):
        vertices, alphas = _build_web_mesh()
        self.assertEqual(len(vertices) % 2, 0)
        self.assertEqual(len(vertices), len(alphas))
        self.assertTrue(np.any((vertices[:, 0] == 0) & (vertices[:, 1] == 0)))
        self.assertTrue(np.any(vertices[:, 0] == 1))
        self.assertTrue(np.all(np.isfinite(vertices)))

    def test_web_progress_uses_delta_and_target_stays_onscreen(self):
        web = WebProjectile(glm.vec2(950, 540), glm.vec2(1, -0.2), 'Right')
        self.assertLessEqual(web.target.x, 1920)
        web.update(0.2)
        self.assertGreater(web.progress, 0)
        self.assertFalse(web.impact)
        for _ in range(11):
            web.update(0.1)
        self.assertTrue(web.impact)
        self.assertTrue(web.shockwave_alive)


class IdentityTests(unittest.TestCase):
    def test_application_uses_embedding_recognizer_not_legacy_lbph(self):
        import main
        self.assertIs(main.SFaceIdentity, SFaceIdentity)

    def test_embedding_model_accepts_owner_and_rejects_different_face(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'owner.npz'
            extractor = lambda vector: vector
            owner = SFaceIdentity(path, extractor=extractor, samples_required=3, sample_interval=0)
            owner.start_enrollment()
            for f in ([1.0, 0.0, 0.0], [0.98, 0.1, 0.0], [0.96, 0.2, 0.0]):
                owner.process(np.array(f, dtype=np.float32), None)
            self.assertTrue(path.exists())
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            again = SFaceIdentity(path, extractor=extractor)
            self.assertTrue(again.enrolled)
            for _ in range(3):
                status = again.process(np.array([0.95, 0.25, 0.0], dtype=np.float32), None)
            self.assertEqual(status, 'AUTHORIZED')
            for _ in range(3):
                status = again.process(np.array([0.0, 0.0, 1.0], dtype=np.float32), None)
            self.assertEqual(status, 'DANGER')
            self.assertEqual(again.process(None, None), 'NO_FACE')
            again.delete_enrollment()
            self.assertFalse(path.exists())

    def test_loaded_face_threshold_adapts_to_training_variation(self):
        # A fixed 50 rejected the user's real camera frames (79-84) even though
        # nearest-neighbor distances among their enrollment samples were 76-79.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'face.yml'
            enrolled = FaceIdentity(path, samples_required=6, sample_interval=0)
            enrolled.start_enrollment()
            base = np.random.default_rng(4).integers(0, 256, (100, 100, 3), dtype=np.uint8)
            for shift in range(6):
                enrolled.process(np.roll(base, shift * 12, axis=1), (0, 0, 100, 100))
            loaded = FaceIdentity(path)
            hist = loaded._model.getHistograms()
            neighbor_max = max(min(cv2.compareHist(hist[i], hist[j], cv2.HISTCMP_CHISQR_ALT)
                                   for j in range(len(hist)) if j != i) for i in range(len(hist)))
            self.assertGreater(loaded.threshold, neighbor_max)
            self.assertLessEqual(loaded.threshold, 90)

    def test_unenrolled_face_is_not_called_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = FaceIdentity(Path(directory) / 'face.yml')
            self.assertEqual(identity.process(np.zeros((100, 100, 3), dtype=np.uint8), None), 'NO_FACE')
            self.assertEqual(identity.process(np.zeros((100, 100, 3), dtype=np.uint8), (10, 10, 70, 70)), 'NOT_ENROLLED')

    def test_enrollment_requires_multiple_samples_and_persists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'face.yml'
            identity = FaceIdentity(path, samples_required=3, sample_interval=0)
            face = np.random.default_rng(22).integers(0, 256, (100, 100, 3), dtype=np.uint8)
            identity.start_enrollment()
            for _ in range(3):
                state = identity.process(face, (10, 10, 80, 80))
            self.assertEqual(state, 'SCANNING')
            self.assertTrue(path.exists())
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            self.assertTrue(FaceIdentity(path).enrolled)
            for _ in range(5):
                state = identity.process(face, (10, 10, 80, 80))
            self.assertEqual(state, 'AUTHORIZED')
            identity.delete_enrollment()
            self.assertFalse(path.exists())
            self.assertFalse(identity.enrolled)

    def test_unknown_face_enters_danger_only_after_stable_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = FaceIdentity(Path(directory) / 'face.yml', samples_required=3, sample_interval=0)
            owner = np.random.default_rng(1).integers(0, 256, (100, 100, 3), dtype=np.uint8)
            stranger = np.random.default_rng(2).integers(0, 256, (100, 100, 3), dtype=np.uint8)
            identity.start_enrollment()
            for _ in range(3):
                identity.process(owner, (10, 10, 80, 80))
            self.assertEqual(identity.process(stranger, (10, 10, 80, 80)), 'SCANNING')
            identity.process(stranger, (10, 10, 80, 80))
            self.assertEqual(identity.process(stranger, (10, 10, 80, 80)), 'DANGER')
            self.assertEqual(identity.process(stranger, None), 'NO_FACE')


class FaceHudTests(unittest.TestCase):
    def test_unknown_face_warns_once_without_welcome(self):
        class Audio:
            def __init__(self): self.sounds, self.words = [], []
            def play(self, name): self.sounds.append(name)
            def speak(self, text): self.words.append(text)
        audio, hud = Audio(), FaceScannerHUD()
        face = FaceData()
        face.detected = True
        face.box_x, face.box_y, face.box_w, face.box_h = 0.3, 0.2, 0.3, 0.5
        face.identity_status = 'DANGER'
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        hud.render(frame, face, 0.1, audio)
        hud.render(frame, face, 0.1, audio)
        self.assertEqual(audio.sounds.count('warning'), 1)
        self.assertEqual(audio.words, [])

    def test_welcome_only_after_authorized_scan(self):
        class Audio:
            spoken = []
            def play(self, name): pass
            def speak(self, text): self.spoken.append(text)

        audio = Audio()
        hud = FaceScannerHUD()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        face = FaceData()
        face.detected = True
        face.box_x, face.box_y = 0.3, 0.2
        face.box_w, face.box_h = 0.3, 0.5
        face.identity_status = 'DANGER'
        hud.render(frame, face, 3.0, audio)
        self.assertEqual(audio.spoken, [])
        face.identity_status = 'AUTHORIZED'
        hud.render(frame, face, 3.0, audio)
        self.assertEqual(len(audio.spoken), 1)
        self.assertIn('Welcome', audio.spoken[0])
        hud.render(frame, face, 0.1, audio)
        self.assertEqual(len(audio.spoken), 1)


class ScreenshotTests(unittest.TestCase):
    def test_capture_uses_final_framebuffer_and_flips_gl_orientation(self):
        class Screen:
            def read(self, **kwargs):
                return bytes([0, 0, 255, 255, 0, 0])  # bottom blue, top red RGB
        class Context:
            screen = Screen()
        frame = ScreenshotManager.capture_screen(Context(), 1, 2)
        self.assertEqual(tuple(frame[0, 0]), (0, 0, 255))  # top red BGR
        self.assertEqual(tuple(frame[1, 0]), (255, 0, 0))  # bottom blue BGR


class RendererTests(unittest.TestCase):
    def test_fullscreen_avoids_crashing_video_mode_probe(self):
        import glfw
        renderer = Renderer.__new__(Renderer)
        renderer._key_callbacks = []
        with patch.object(glfw, 'init', return_value=True), \
             patch.object(glfw, 'get_primary_monitor', return_value=object()), \
             patch.object(glfw, 'get_video_mode', side_effect=AssertionError('native segfault')), \
             patch.object(glfw, 'window_hint'), patch.object(glfw, 'create_window', return_value=object()), \
             patch.object(glfw, 'set_window_pos'), patch.object(glfw, 'get_framebuffer_size', return_value=(1920, 1080)), \
             patch.object(glfw, 'make_context_current'), patch.object(glfw, 'swap_interval'), \
             patch.object(glfw, 'set_key_callback'):
            renderer._init_glfw()

    def test_missing_monitor_fails_cleanly_instead_of_segfault(self):
        import glfw
        renderer = Renderer.__new__(Renderer)
        with patch.object(glfw, 'init', return_value=True), \
             patch.object(glfw, 'get_primary_monitor', return_value=None), \
             patch.object(glfw, 'get_video_mode', side_effect=AssertionError('NULL MONITOR')), \
             patch.object(glfw, 'window_hint'), patch.object(glfw, 'terminate'):
            with self.assertRaisesRegex(RuntimeError, 'monitor'):
                renderer._init_glfw()


class MenuTests(unittest.TestCase):
    def test_hover_and_select_uses_camera_frame_coordinates(self):
        menu = HologramMenu()
        menu.show()
        menu.update(0.1, (880, 230))  # first row at x~832, y~204
        self.assertEqual(menu.select_hovered(), 'AUDIO')
        menu.update(0.1, (10, 10))
        self.assertIsNone(menu.select_hovered())

    def test_audio_mute_disables_sfx_and_voice(self):
        audio = AudioManager.__new__(AudioManager)
        audio.muted = True
        audio._sounds = {'web': type('S', (), {'play': lambda self, **kw: self.fail()})()}
        audio.play('web')
        audio.speak('Hello')
        self.assertFalse(hasattr(audio, '_tts_thread'))

    def test_voice_has_offline_macos_fallback(self):
        audio = AudioManager.__new__(AudioManager)
        with patch('sound.audio_manager.subprocess.run') as run:
            audio._speak_offline('Welcome, Mister.')
            run.assert_called_once_with(
                ['say', '-v', 'Daniel', 'Welcome, Mister.'], check=True, timeout=25
            )


if __name__ == '__main__':
    unittest.main()

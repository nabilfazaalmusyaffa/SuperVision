"""Web Aim overlays the camera only when explicitly enabled."""
import importlib.util
import unittest
import time
from unittest.mock import Mock, patch
from types import SimpleNamespace
import threading

import numpy as np
import cv2
import glm
from config.settings import AppState
from effects.web_projectile import WebProjectile
from tracking.gesture import Gesture, HandState, web_aim, web_origin, detect_gesture, GestureLatch
from tracking.face_tracker import FaceData
from vision.face_session import FaceSession


class WebAimModeTests(unittest.TestCase):
    def test_off_does_not_change_camera_frame(self):
        self.assertIsNotNone(importlib.util.find_spec('hud.web_aim'), 'Web Aim HUD is missing')
        from hud.web_aim import WebAimHUD
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        hud = WebAimHUD()
        result = hud.render(frame, [], now=1.0)
        self.assertTrue(np.array_equal(result, frame))
        self.assertFalse(hud.enabled)

    def test_ready_reticle_follows_real_projectile_target_without_mutating_input(self):
        from hud.web_aim import WebAimHUD
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        landmarks = [(0.5, 0.5, 0.0)] * 21
        landmarks[5] = (0.5, 0.5, 0.0)
        landmarks[8] = (0.6, 0.45, 0.0)
        hand = HandState(gesture=Gesture.WEB_SHOT, landmarks=landmarks)
        projectile = WebProjectile(web_origin(landmarks), web_aim(landmarks), 'Right')
        cx = round(projectile.target.x * frame.shape[1] / 1920)
        cy = round(projectile.target.y * frame.shape[0] / 1080)
        hud = WebAimHUD()
        hud.enabled = True
        rendered = hud.render(frame, [hand], now=1.0)
        self.assertGreater(np.count_nonzero(rendered[cy-22:cy+23, cx-22:cx+23]), 40)
        self.assertEqual(np.count_nonzero(frame), 0)

    def test_shot_flash_is_visible_then_expires_without_hand(self):
        from hud.web_aim import WebAimHUD
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        hud = WebAimHUD()
        hud.enabled = True
        hud.flash(glm.vec2(960, 540), now=1.0)
        active = hud.render(frame, [], now=1.1)
        expired = hud.render(frame, [], now=1.6)
        self.assertGreater(np.count_nonzero(active[145:215, 285:355]), 50)
        self.assertEqual(np.count_nonzero(expired[145:215, 285:355]), 0)

    def test_both_web_muzzles_flash_when_two_hands_fire_together(self):
        from hud.web_aim import WebAimHUD
        hud = WebAimHUD(); hud.enabled = True
        hud.flash(glm.vec2(480, 540), now=1.0)
        hud.flash(glm.vec2(1440, 540), now=1.0)
        frame = hud.render(np.zeros((360, 640, 3), np.uint8), [], now=1.1)
        self.assertGreater(np.count_nonzero(frame[155:205, 135:185]), 20)
        self.assertGreater(np.count_nonzero(frame[155:205, 455:505]), 20)

    def test_app_camera_overlay_requires_an_unlocked_face_session(self):
        from hud.web_aim import WebAimHUD
        from main import SpiderVisionApp
        app = SpiderVisionApp.__new__(SpiderVisionApp)
        app.web_aim_hud = WebAimHUD()
        app.web_aim_hud.enabled = True
        app.anatomy = Mock(mode='CLOSED')
        app.state = AppState.IDLE
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        landmarks = [(0.5, 0.5, 0.0)] * 21
        landmarks[8] = (0.6, 0.45, 0.0)
        hand = HandState(gesture=Gesture.WEB_SHOT, landmarks=landmarks)
        self.assertTrue(np.array_equal(app._render_web_aim(frame, [hand], False), frame))
        self.assertGreater(np.count_nonzero(app._render_web_aim(frame, [hand], True)), 50)
        app.anatomy.mode = 'BODY'
        self.assertTrue(np.array_equal(app._render_web_aim(frame, [hand], True), frame))

    def test_firing_with_tracked_hand_produces_brief_muzzle_flash(self):
        from hud.web_aim import WebAimHUD
        from main import SpiderVisionApp
        app = SpiderVisionApp.__new__(SpiderVisionApp)
        app.web_aim_hud = WebAimHUD()
        app.web_aim_hud.enabled = True
        app._last_web_shot = {}
        app.web_system = Mock()
        app.particles = Mock()
        app.audio = None
        app.cam_effect = Mock()
        landmarks = [(0.5, 0.5, 0.0)] * 21
        landmarks[8] = (0.6, 0.45, 0.0)
        hand = HandState(gesture=Gesture.WEB_SHOT, landmarks=landmarks)
        app._fire_web('Right', hand)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        rendered = app.web_aim_hud.render(frame, [], now=time.monotonic())
        self.assertGreater(np.count_nonzero(rendered[145:180, 360:405]), 20)

    def test_downward_spiderman_pose_stays_a_web_shot_and_projects_below_hand(self):
        landmarks = [(0.5, 0.5, 0.0)] * 21
        for index, xy in {0:(0.5,0.8), 2:(0.49,0.63), 4:(0.35,0.48),
                          5:(0.46,0.55), 6:(0.46,0.63), 8:(0.46,0.74),
                          9:(0.52,0.55), 10:(0.54,0.54), 12:(0.51,0.56),
                          13:(0.58,0.55), 14:(0.60,0.54), 16:(0.58,0.56),
                          17:(0.62,0.55), 18:(0.65,0.48), 20:(0.70,0.40)}.items():
            landmarks[index] = (*xy, 0.0)
        self.assertEqual(detect_gesture(landmarks), Gesture.WEB_SHOT)
        projectile = WebProjectile(web_origin(landmarks), web_aim(landmarks), 'Right')
        self.assertGreater(projectile.target.y, projectile.origin.y)

    def test_web_near_screen_edge_never_reverses_a_downward_or_rightward_aim(self):
        down = WebProjectile(glm.vec2(960, 1040), glm.vec2(0, 1), 'Right')
        right = WebProjectile(glm.vec2(1880, 540), glm.vec2(1, 0), 'Right')
        self.assertGreater(down.target.y, down.origin.y)
        self.assertLessEqual(down.target.y, 1079)
        self.assertGreater(right.target.x, right.origin.x)
        self.assertLessEqual(right.target.x, 1919)

    def test_bottom_edge_reticle_agrees_with_downward_web_target(self):
        from hud.web_aim import WebAimHUD
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        landmarks = [(0.5, 0.5, 0.0)] * 21
        landmarks[5] = (0.5, 0.65, 0.0)
        landmarks[8] = (0.5, 0.90, 0.0)
        hand = HandState(gesture=Gesture.WEB_SHOT, landmarks=landmarks)
        target = WebProjectile(web_origin(landmarks), web_aim(landmarks), 'Right').target
        self.assertGreater(target.y, web_origin(landmarks).y)
        hud = WebAimHUD(); hud.enabled = True
        with patch('hud.web_aim.cv2.circle', wraps=cv2.circle) as circles:
            rendered = hud.render(frame, [hand], now=1)
        target_pixel = (round(target.x / 3), round(target.y / 3))
        self.assertIn(target_pixel,
                      [call.args[1] for call in circles.call_args_list if call.args[2] >= 10])
        self.assertGreater(np.count_nonzero(rendered[-5:, 312:329]), 12)

    def test_two_spiderman_hands_launch_independently_in_normal_and_aim_modes(self):
        from hud.web_aim import WebAimHUD
        from main import SpiderVisionApp
        for aim_enabled in (False, True):
            app = SpiderVisionApp.__new__(SpiderVisionApp)
            app._prev_gestures = {}
            app._web_latch = GestureLatch(hold_seconds=0)
            app._last_web_shot = {}
            app.anatomy = Mock(mode='CLOSED')
            app.web_system, app.particles, app.cam_effect = Mock(), Mock(), Mock()
            app.audio = None
            app.web_aim_hud = WebAimHUD()
            app.web_aim_hud.enabled = aim_enabled
            left_lm = [(0.3, 0.5, 0.0)] * 21
            right_lm = [(0.7, 0.5, 0.0)] * 21
            left_lm[8] = (0.3, 0.4, 0.0)
            right_lm[8] = (0.7, 0.6, 0.0)
            hands = [HandState(handedness='Left', gesture=Gesture.WEB_SHOT, landmarks=left_lm),
                     HandState(handedness='Right', gesture=Gesture.WEB_SHOT, landmarks=right_lm)]
            app._process_gestures(hands, 0.016)
            self.assertEqual(app.web_system.launch.call_count, 2)

    def test_hand_tracking_exposes_current_unsmoothed_tip_for_web_aim(self):
        from tracking.hand_tracker import HandTracker
        tracker = HandTracker.__new__(HandTracker)
        tracker._lock = threading.Lock()
        tracker._smooth = {'Right': [(0.5, 0.5, 0.0)] * 21}
        raw = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
        raw[8].x = 0.8
        result = SimpleNamespace(
            multi_hand_landmarks=[SimpleNamespace(landmark=raw)],
            multi_handedness=[SimpleNamespace(classification=[SimpleNamespace(label='Right', score=0.9)])],
        )
        tracker._hands_sol = SimpleNamespace(process=lambda frame: result)
        state = tracker.process(np.zeros((2, 2, 3), dtype=np.uint8))[0]
        self.assertEqual(state.aim_landmarks[8][0], 0.8)
        self.assertLess(state.landmarks[8][0], 0.8)

    def test_reticle_uses_current_tip_without_waiting_for_skeleton_smoothing(self):
        from hud.web_aim import WebAimHUD
        slow = [(0.5, 0.5, 0.0)] * 21
        current = list(slow)
        slow[8] = (0.55, 0.45, 0.0)
        current[8] = (0.80, 0.45, 0.0)
        hand = HandState(gesture=Gesture.WEB_SHOT, landmarks=slow,
                         aim_landmarks=current)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        target = WebProjectile(web_origin(current), web_aim(current), 'Right').target
        cx, cy = round(target.x / 3), round(target.y / 3)
        hud = WebAimHUD(); hud.enabled = True
        rendered = hud.render(frame, [hand], now=1.0)
        self.assertGreater(np.count_nonzero(rendered[cy-21:cy+22, cx-21:cx+22]), 40)

    def test_launched_web_uses_same_current_tip_as_reticle(self):
        from hud.web_aim import WebAimHUD
        from main import SpiderVisionApp
        app = SpiderVisionApp.__new__(SpiderVisionApp)
        app._last_web_shot = {}
        app.web_system, app.particles, app.cam_effect = Mock(), Mock(), Mock()
        app.audio = None
        app.web_aim_hud = WebAimHUD()
        app.web_aim_hud.enabled = True
        slow = [(0.5, 0.5, 0.0)] * 21
        current = list(slow)
        slow[8] = (0.55, 0.45, 0.0)
        current[8] = (0.80, 0.45, 0.0)
        app._fire_web('Right', HandState(landmarks=slow, aim_landmarks=current))
        shot_origin, shot_direction, _ = app.web_system.launch.call_args.args
        self.assertAlmostEqual(shot_origin.x, web_origin(current).x)
        self.assertAlmostEqual(shot_direction.x, web_aim(current).x)

    def test_tracking_publishes_new_aim_before_slower_identity_matching(self):
        from main import SpiderVisionApp
        app = SpiderVisionApp.__new__(SpiderVisionApp)
        app._track_running = True
        app._track_lock = threading.Lock()
        app._hands = []
        app._face_data = FaceData()
        app._track_sequence = 0
        app.session = FaceSession()
        hand = HandState(handedness='Right')
        frame = np.zeros((12, 12, 3), dtype=np.uint8)
        calls = 0
        def read_rgb():
            nonlocal calls
            calls += 1
            if calls > 1:
                app._track_running = False
                return None
            return frame
        app.camera = SimpleNamespace(read_rgb=read_rgb, read=lambda: frame)
        app.face_tracker = SimpleNamespace(process=lambda image, dt: FaceData())
        app.hand_tracker = SimpleNamespace(process=lambda image: [hand])
        app.mask_detector = SimpleNamespace(detect=lambda image, face: None)
        def identity_process(image, box):
            self.assertEqual(app._hands, [hand])
            return 'NO_FACE'
        app.identity = SimpleNamespace(process=identity_process, sample_count=0, samples_required=8)
        app._track_loop()
        self.assertEqual(app._track_sequence, 1)


if __name__ == '__main__':
    unittest.main()

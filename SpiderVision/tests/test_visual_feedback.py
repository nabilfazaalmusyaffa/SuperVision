"""Camera-frame visual feedback; no GL context or biometric data required."""
import unittest
from unittest.mock import patch

import cv2
import glm
import numpy as np

from effects.web_projectile import WebProjectile, WebProjectileSystem
from hud.vitals import VitalsHUD


class WebFeedbackTests(unittest.TestCase):
    def test_war_web_is_visibly_wider_than_standard_web(self):
        ordinary = WebProjectile(glm.vec2(480, 540), glm.vec2(1, 0), 'Right')
        war = WebProjectile(glm.vec2(480, 540), glm.vec2(1, 0), 'Right', war_mode=True)
        ordinary.update(0.12)
        war.update(0.12)
        system = WebProjectileSystem.__new__(WebProjectileSystem)
        frame = np.zeros((720, 1280, 3), np.uint8)
        system._webs = [ordinary]
        normal_pixels = np.count_nonzero(system.render_overlay(frame))
        system._webs = [war]
        war_pixels = np.count_nonzero(system.render_overlay(frame))
        self.assertGreater(war_pixels, normal_pixels * 1.25)
        self.assertEqual(np.count_nonzero(frame), 0)

    def test_war_web_remains_retractable_after_normal_web_has_expired(self):
        origin, direction = glm.vec2(480, 540), glm.vec2(1, 0)
        ordinary = WebProjectile(origin, direction, 'Right')
        war = WebProjectile(origin, direction, 'Right', war_mode=True)
        for web in (ordinary, war):
            web.update(0.5)
            web.update(2.2)
        self.assertFalse(ordinary.alive)
        self.assertTrue(war.alive)
        self.assertTrue(war.retract())

    def test_retraction_stays_visible_for_half_second_and_finishes_cleanly(self):
        web = WebProjectile(glm.vec2(480, 540), glm.vec2(1, 0), 'Right', war_mode=True)
        web.update(0.38)
        self.assertTrue(web.retract())
        web.update(0.5)
        self.assertTrue(web.alive)
        self.assertGreater(web.progress, 0.05)
        web.update(0.4)
        self.assertFalse(web.alive)
        self.assertEqual(web.progress, 0.0)

    def test_war_web_retracts_to_fingertip_without_a_second_impact(self):
        web = WebProjectile(glm.vec2(480, 540), glm.vec2(1, 0), 'Right', war_mode=True)
        web.update(0.38)
        self.assertTrue(web.impact)
        self.assertTrue(web.retract())
        before = web.progress
        web.update(0.1)
        self.assertLess(web.progress, before)
        self.assertTrue(web.alive)
        self.assertFalse(web.shockwave_alive)
        web.update(0.2)
        self.assertTrue(web.alive)
        web.update(0.6)
        self.assertFalse(web.alive)
        self.assertEqual(web.progress, 0.0)
        self.assertFalse(web.retract())

    def test_retraction_has_distinct_moving_tip_halo_and_label(self):
        web = WebProjectile(glm.vec2(480, 540), glm.vec2(1, 0), 'Right', war_mode=True)
        web.update(0.38)
        self.assertTrue(web.retract())
        web.update(0.1)
        system = WebProjectileSystem.__new__(WebProjectileSystem)
        system._webs = [web]
        frame = np.zeros((720, 1280, 3), np.uint8)
        tip_x = round((web.origin.x + (web.target.x - web.origin.x) * web.progress) * 1280 / 1920)
        tip_y = round(web.origin.y * 720 / 1080)
        with patch('cv2.putText', wraps=cv2.putText) as text:
            result = system.render_overlay(frame)
        labels = [call for call in text.call_args_list if 'RETRACT' in call.args[1]]
        self.assertEqual(len(labels), 1)
        self.assertGreaterEqual(labels[0].args[4], 0.55)
        self.assertGreater(result[tip_y + 23, tip_x, 2], 80)
        self.assertEqual(np.count_nonzero(frame), 0)

    def test_second_war_shot_replaces_only_prior_web_from_same_hand(self):
        system = WebProjectileSystem.__new__(WebProjectileSystem)
        system._webs = []
        origin = glm.vec2(480, 540)
        direction = glm.vec2(1, 0)
        system.launch(origin, direction, 'Right', war_mode=True)
        first = system._webs[-1]
        system.launch(origin, direction, 'Left', war_mode=True)
        system.launch(origin, direction, 'Right', war_mode=True)
        self.assertEqual(len(system._webs), 2)
        self.assertNotIn(first, system._webs)
        self.assertEqual({web.side for web in system._webs}, {'Right', 'Left'})

    def test_retraction_targets_only_latest_war_web_of_same_hand(self):
        system = WebProjectileSystem.__new__(WebProjectileSystem)
        ordinary = WebProjectile(glm.vec2(480, 540), glm.vec2(1, 0), 'Right')
        right = WebProjectile(glm.vec2(480, 540), glm.vec2(1, 0), 'Right', war_mode=True)
        left = WebProjectile(glm.vec2(480, 540), glm.vec2(-1, 0), 'Left', war_mode=True)
        system._webs = [ordinary, right, left]
        for web in system._webs:
            web.update(0.15)
        self.assertTrue(system.retract('Right'))
        self.assertFalse(system.retract('Right'))
        self.assertFalse(ordinary.retracting)
        self.assertFalse(left.retracting)
        system.clear_war_webs()
        self.assertEqual(system._webs, [ordinary])

    def test_flight_starts_visibly_and_finishes_before_half_second(self):
        web = WebProjectile(glm.vec2(900, 500), glm.vec2(1, 0), 'Right')
        self.assertLess(web.flight_duration, 0.5)
        web.update(0.016)
        self.assertGreater(web.progress, 0.2)
        self.assertFalse(web.impact)
        web.update(web.flight_duration)
        self.assertTrue(web.impact)

    def test_overlay_draws_bright_beam_from_fingertip_without_gl(self):
        web = WebProjectile(glm.vec2(480, 540), glm.vec2(1, 0), 'Right')
        web.update(0.016)
        system = WebProjectileSystem.__new__(WebProjectileSystem)
        system._webs = [web]
        frame = np.zeros((720, 1280, 3), np.uint8)
        result = system.render_overlay(frame)
        self.assertGreater(np.count_nonzero(result), 1000)
        # WINDOW_W / CAMERA_W == WINDOW_H / CAMERA_H == 1.5
        self.assertGreater(result[360, 320:400].max(), 180)
        self.assertEqual(np.count_nonzero(frame), 0)
        self.assertGreater(result[:, :, 0].max(), 180)  # cyan / white in BGR

    def test_war_hud_is_red_pulsing_and_explicitly_visual(self):
        from hud.war_mode import WarModeHUD
        hud = WarModeHUD()
        frame = np.zeros((720, 1280, 3), np.uint8)
        with patch('cv2.putText', wraps=cv2.putText) as text:
            first = hud.render(frame, 0.0)
        labels = [call.args[1] for call in text.call_args_list]
        self.assertTrue(any('WAR MODE' in label and 'VISUAL' in label for label in labels))
        self.assertTrue(any('FIST' in label and 'RETRACT' in label for label in labels))
        self.assertGreater(first[15, 15, 2], first[15, 15, 0])
        self.assertFalse(np.array_equal(first, hud.render(frame, 0.35)))
        self.assertEqual(np.count_nonzero(frame), 0)


class SimulatedVitalsTests(unittest.TestCase):
    def test_label_is_explicit_and_animation_changes_with_time(self):
        hud = VitalsHUD()
        frame = np.zeros((360, 640, 3), np.uint8)
        with patch('cv2.putText', wraps=cv2.putText) as text:
            first = hud.render(frame, 0.0)
        labels = [call.args[1] for call in text.call_args_list]
        self.assertIn('PULSE FX / BREATH FX - SIMULATED', labels)
        self.assertIn('HEARTBEAT', labels)
        self.assertIn('BREATHING', labels)
        self.assertFalse(any('BPM' in label or 'SpO2' in label for label in labels))
        later = hud.render(frame, 0.35)
        self.assertGreater(np.count_nonzero(first), 1000)
        self.assertFalse(np.array_equal(first, later))
        self.assertEqual(np.count_nonzero(frame), 0)


if __name__ == '__main__':
    unittest.main()

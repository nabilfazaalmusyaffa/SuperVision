import unittest
from unittest.mock import Mock

import numpy as np
import glm

from sound.voice_commands import parse_command, VoiceCommandListener
from effects.web_projectile import WebProjectile, WebProjectileSystem
from hud.vitals import VitalsHUD


class VoiceCommandTests(unittest.TestCase):
    def test_short_english_commands_and_bare_menu_fallback(self):
        self.assertEqual(parse_command('Jarvis menu'), 'MENU')
        self.assertEqual(parse_command('hey Jarvis show menu'), 'MENU')
        self.assertEqual(parse_command('Jarvis screenshot'), 'SCREENSHOT')
        self.assertEqual(parse_command('Jarvis close menu'), 'CLOSE')
        self.assertEqual(parse_command('Jarvis audio'), 'AUDIO')
        self.assertEqual(parse_command('Jarvis performance'), 'PERFORMANCE')
        self.assertEqual(parse_command('Jarvis effects'), 'EFFECTS')
        self.assertEqual(parse_command('Jarvis exit'), 'EXIT')
        self.assertEqual(parse_command('menu'), 'MENU')
        self.assertIsNone(parse_command('exit'))
        self.assertIsNone(parse_command('Jarvis random sentence'))

    def test_listener_queue_only_accepts_command_when_authorized(self):
        listener = VoiceCommandListener(audio=None)
        listener.accept_text('Jarvis menu')
        self.assertEqual(listener.pop_command(), 'MENU')
        self.assertIsNone(listener.pop_command())


class WebFeedbackTests(unittest.TestCase):
    def test_web_visible_immediately_in_camera_overlay(self):
        web = WebProjectile(glm.vec2(400, 350), glm.vec2(1, -0.1), 'Right')
        web.update(0.016)
        self.assertGreater(web.progress, 0.2)
        self.assertLess(web.flight_duration, 0.5)
        system = WebProjectileSystem.__new__(WebProjectileSystem)
        system._webs = [web]
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = system.render_overlay(frame)
        self.assertGreater(np.count_nonzero(result), 1000)


class VitalsTests(unittest.TestCase):
    def test_simulated_pulse_and_breath_are_labeled(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        hud = VitalsHUD()
        result = hud.render(frame, 1.0)
        self.assertGreater(np.count_nonzero(result), 100)
        self.assertEqual(hud.label, 'PULSE FX / BREATH FX - SIMULATED')


if __name__ == '__main__':
    unittest.main()

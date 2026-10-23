import unittest
from unittest.mock import Mock
import glfw

from main import SpiderVisionApp
from tracking.face_tracker import FaceData
from vision.face_session import FaceSession


class AppSessionIntegrationTests(unittest.TestCase):
    def test_verified_session_survives_absence_but_locks_on_other_face(self):
        app = SpiderVisionApp.__new__(SpiderVisionApp)
        app.session = FaceSession()
        face = FaceData()
        app._apply_session(face, 'AUTHORIZED')
        self.assertTrue(face.session_unlocked)
        app._apply_session(face, 'NO_FACE')
        self.assertTrue(face.session_unlocked)
        app._apply_session(face, 'DANGER')
        self.assertFalse(face.session_unlocked)
        app._apply_session(face, 'AUTHORIZED')
        self.assertTrue(face.session_unlocked)

    def test_menu_stays_available_after_owner_steps_out_of_view(self):
        app = SpiderVisionApp.__new__(SpiderVisionApp)
        app.session = FaceSession()
        face = FaceData()
        app._apply_session(face, 'AUTHORIZED')
        app._apply_session(face, 'NO_FACE')
        app.menu, app.audio = Mock(), Mock(muted=False)
        app.voice = Mock()
        app.voice.pop_command.side_effect = ['MENU', None]
        app._voice_notice = ''
        app._process_voice_commands(face.session_unlocked)
        app.menu.show.assert_called_once()

    def test_keyboard_fallback_remains_available_without_repeating_face_scan(self):
        app = SpiderVisionApp.__new__(SpiderVisionApp)
        app.menu = Mock(is_open=False)
        app._face_data = FaceData()
        app._face_data.identity_status = 'NO_FACE'
        app._face_data.session_unlocked = True
        app._on_key(glfw.KEY_M, glfw.PRESS, 0)
        app.menu.show.assert_called_once()


if __name__ == '__main__':
    unittest.main()

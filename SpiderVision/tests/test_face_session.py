"""Session-latch and scanner regression tests for visual face matching."""
import unittest
from unittest.mock import patch
import cv2

import numpy as np

from vision.face_session import FaceSession
from tracking.face_tracker import FaceData
from hud.face_scanner import FaceScannerHUD


class FaceSessionTests(unittest.TestCase):
    def test_authorized_latches_through_absence_and_scanning(self):
        session = FaceSession()
        self.assertFalse(session.unlocked)
        self.assertTrue(session.observe("AUTHORIZED"))
        for raw_status in ("NO_FACE", "SCANNING", "NO_FACE"):
            self.assertTrue(session.observe(raw_status))
            self.assertTrue(session.unlocked)

    def test_stranger_and_enrollment_revoke_until_fresh_authorization(self):
        for revoked_status in ("DANGER", "DANGER_OTHER", "NOT_ENROLLED", "ENROLLING"):
            with self.subTest(status=revoked_status):
                session = FaceSession()
                session.observe("AUTHORIZED")
                self.assertFalse(session.observe(revoked_status))
                self.assertFalse(session.observe("NO_FACE"))
                self.assertFalse(session.observe("SCANNING"))
                self.assertTrue(session.observe("AUTHORIZED"))


class FaceDataTests(unittest.TestCase):
    def test_session_flag_defaults_locked_and_can_be_updated(self):
        face = FaceData()
        self.assertFalse(face.session_unlocked)
        face.session_unlocked = True
        self.assertTrue(face.session_unlocked)


class FaceHudSessionTests(unittest.TestCase):
    class Audio:
        def __init__(self):
            self.sounds, self.words = [], []

        def play(self, sound):
            self.sounds.append(sound)

        def speak(self, words):
            self.words.append(words)

    @staticmethod
    def face(session, status, detected=True):
        face = FaceData()
        face.identity_status = status
        face.session_unlocked = session.observe(status)
        face.detected = detected
        face.box_x, face.box_y, face.box_w, face.box_h = 0.2, 0.2, 0.4, 0.5
        face.confidence = 0.9
        return face

    def test_absence_does_not_rearm_completed_scan_or_greeting(self):
        session, hud, audio = FaceSession(), FaceScannerHUD(), self.Audio()
        frame = np.zeros((120, 220, 3), dtype=np.uint8)
        with patch('hud.face_scanner.time.monotonic') as clock:
            clock.return_value = 10.0
            hud.render(frame, self.face(session, 'AUTHORIZED'), 3.0, audio)
            self.assertEqual(len(audio.words), 1)
            clock.return_value = 11.0
            with patch.object(cv2, 'putText', wraps=cv2.putText) as text:
                hud.render(frame, self.face(session, 'NO_FACE', False), 10.0, audio)
            self.assertIn('SESSION ACTIVE - FACE OUT OF VIEW',
                          [call.args[1] for call in text.call_args_list])
            self.assertTrue(hud.is_locked)
            self.assertEqual(hud.scan_pct, 100.0)
            clock.return_value = 30.0
            with patch.object(cv2, 'putText', wraps=cv2.putText) as text:
                hud.render(frame, self.face(session, 'AUTHORIZED'), 0.1, audio)
            self.assertIn('SESSION ACTIVE', [call.args[1] for call in text.call_args_list])
        self.assertEqual(len(audio.words), 1)
        self.assertEqual(audio.sounds.count('lock'), 1)
        self.assertEqual(hud.scan_pct, 100.0)


    def test_stranger_requires_new_authorized_scan_and_greeting_once(self):
        session, hud, audio = FaceSession(), FaceScannerHUD(), self.Audio()
        frame = np.zeros((120, 220, 3), dtype=np.uint8)
        hud.render(frame, self.face(session, 'AUTHORIZED'), 3.0, audio)
        hud.render(frame, self.face(session, 'DANGER_OTHER', False), 0.1, audio)
        self.assertFalse(session.unlocked)
        self.assertFalse(hud.is_locked)
        self.assertEqual(hud.scan_pct, 0.0)
        hud.render(frame, self.face(session, 'NO_FACE', False), 0.1, audio)
        hud.render(frame, self.face(session, 'SCANNING'), 3.0, audio)
        self.assertEqual(len(audio.words), 2)  # welcome and stranger warning only
        self.assertFalse(hud.is_locked)
        hud.render(frame, self.face(session, 'AUTHORIZED'), 0.1, audio)
        self.assertTrue(session.unlocked)
        self.assertEqual(len(audio.words), 2)
        hud.render(frame, self.face(session, 'AUTHORIZED'), 3.0, audio)
        hud.render(frame, self.face(session, 'AUTHORIZED'), 3.0, audio)
        self.assertEqual(len([word for word in audio.words if word.startswith('Welcome')]), 2)
        self.assertEqual(audio.sounds.count('lock'), 2)


if __name__ == "__main__":
    unittest.main()

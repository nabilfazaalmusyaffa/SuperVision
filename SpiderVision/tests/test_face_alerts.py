"""Behavioral coverage for SFace multi-face alerts and the scanner HUD."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from vision.sface_identity import SFaceIdentity
from hud.face_scanner import FaceScannerHUD


OWNER = [1.0, 0.0, 0.0]
STRANGER = [0.0, 1.0, 0.0]
FRAME = np.zeros((120, 220, 3), dtype=np.uint8)


class IdentityAlertsTests(unittest.TestCase):
    def make_identity(self, directory):
        identity = SFaceIdentity(Path(directory) / 'face.npz', extractor=lambda frame: frame,
                                 samples_required=1, sample_interval=0)
        identity.start_enrollment()
        self.assertEqual(identity.process(OWNER), 'SCANNING')
        return identity

    def test_owner_and_stranger_require_three_consecutive_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = self.make_identity(directory)
            self.assertEqual(identity.process([OWNER, STRANGER]), 'SCANNING')
            self.assertEqual(identity.process([OWNER, STRANGER]), 'SCANNING')
            self.assertEqual(identity.process([OWNER, STRANGER]), 'DANGER_OTHER')
            self.assertEqual(identity.process([OWNER, STRANGER]), 'DANGER_OTHER')
            self.assertEqual(identity.process(None), 'NO_FACE')
            self.assertEqual(identity.process([OWNER, STRANGER]), 'SCANNING')

    def test_enrollment_ignores_multiple_person_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = self.make_identity(directory)
            identity.start_enrollment()
            self.assertEqual(identity.process([OWNER, STRANGER]), 'ENROLLING')
            self.assertEqual(identity.sample_count, 0)
            self.assertEqual(identity.process(OWNER), 'SCANNING')
            self.assertEqual(identity.sample_count, 0)

    def test_yunet_extracts_each_detected_face(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = self.make_identity(directory)
            class Detector:
                def setInputSize(self, size):
                    self.size = size
                def detect(self, frame):
                    return None, np.array([[1, 0, 0, 0, 0.99], [0, 1, 0, 0, 0.85]])
            class Recognizer:
                def alignCrop(self, frame, face):
                    return face
                def feature(self, aligned):
                    return np.array([OWNER if aligned[0] else STRANGER], dtype=np.float32)
            identity._detector = Detector()
            identity._recognizer = Recognizer()
            features = identity._extract_camera_feature(FRAME)
            self.assertEqual(features.shape, (2, 3))
            self.assertEqual(identity._detector.size, (220, 120))
            np.testing.assert_array_equal(features, [OWNER, STRANGER])


class ScannerAlertsTests(unittest.TestCase):
    class Audio:
        def __init__(self):
            self.sounds = []
            self.words = []
        def play(self, sound):
            self.sounds.append(sound)
        def speak(self, words):
            self.words.append(words)

    @staticmethod
    def face(status='AUTHORIZED', detected=True):
        from types import SimpleNamespace
        return SimpleNamespace(identity_status=status, detected=detected,
                               box_x=0.2, box_y=0.2, box_w=0.4, box_h=0.5,
                               landmarks=[], depth_estimate=0.5, confidence=0.9,
                               enroll_progress='')

    def test_other_person_alert_is_visible_without_mediapipe_detection(self):
        import cv2
        hud, audio = FaceScannerHUD(), self.Audio()
        texts = []
        original = cv2.putText
        def draw(frame, text, *args, **kwargs):
            texts.append(text)
            return original(frame, text, *args, **kwargs)
        with patch('hud.face_scanner.cv2.putText', side_effect=draw):
            hud.render(FRAME, self.face('DANGER_OTHER', detected=False), 0.1, audio)
            hud.render(FRAME, self.face('DANGER_OTHER', detected=False), 0.1, audio)
        self.assertIn('DANGER - UNKNOWN PERSON AHEAD', texts)
        self.assertNotIn('NO FACE | E ENROLL | D DELETE', texts)
        self.assertEqual(audio.sounds.count('warning'), 1)
        self.assertEqual(audio.words, ['Danger, Mister. Unknown person detected ahead.'])
        self.assertFalse(hud.is_locked)

    def test_welcome_does_not_rearm_after_owner_absence(self):
        hud, audio = FaceScannerHUD(), self.Audio()
        active = self.face()
        inactive = self.face('NO_FACE', detected=False)
        active.session_unlocked = True
        inactive.session_unlocked = True
        hud.render(FRAME, active, 3.0, audio)
        hud.render(FRAME, inactive, 3.0, audio)
        self.assertTrue(hud.is_locked)
        hud.render(FRAME, active, 3.0, audio)
        self.assertEqual(len(audio.words), 1)
        self.assertEqual(audio.sounds.count('lock'), 1)


if __name__ == '__main__':
    unittest.main()

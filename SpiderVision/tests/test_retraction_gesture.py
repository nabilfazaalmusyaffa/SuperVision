"""Regression cases for retracting a war web with a real-world imperfect fist.

These landmarks are constructed fixtures, not camera captures from the user.
"""
import unittest
from unittest.mock import Mock

from main import SpiderVisionApp
from tracking.gesture import Gesture, GestureLatch, HandState, detect_gesture


def hand_landmarks():
    """Folded four fingers, exposed thumb; plausible normalized hand shape."""
    lm = [(0.5, 0.65, 0.0)] * 21
    points = {
        0: (.50, .85), 2: (.37, .68), 4: (.24, .59),
        5: (.43, .55), 6: (.43, .46), 8: (.46, .61),
        9: (.49, .54), 10: (.49, .43), 12: (.51, .62),
        13: (.55, .56), 14: (.56, .46), 16: (.55, .63),
        17: (.62, .60), 18: (.64, .51), 20: (.61, .65),
    }
    for idx, (x, y) in points.items():
        lm[idx] = (x, y, 0.0)
    return lm


def app_for_gestures():
    app = SpiderVisionApp.__new__(SpiderVisionApp)
    app.anatomy = Mock(mode='CLOSED')
    app._war_mode = True
    app._prev_gestures = {}
    app._last_web_shot = {}
    app._web_latch = GestureLatch(hold_seconds=0.08)
    app.web_system = Mock()
    app.web_system.retract.return_value = True
    app.particles = Mock()
    app.audio = Mock()
    return app


class RetractionGestureTests(unittest.TestCase):
    def test_folded_fingers_retract_even_if_thumb_looks_extended(self):
        from tracking.gesture import is_retraction_pose
        fist = hand_landmarks()
        self.assertEqual(detect_gesture(fist), Gesture.THUMBS_UP)
        self.assertTrue(is_retraction_pose(fist))
        for extended in (8, 20):
            other = list(fist)
            other[extended] = ((.40, .32, 0.0) if extended == 8
                               else (.69, .34, 0.0))
            if extended == 8:
                self.assertFalse(is_retraction_pose(other),
                                 'An extended index finger is pointing, not a fist')
            else:
                self.assertTrue(is_retraction_pose(other),
                                'One occluded or open pinky should not block a fist')
        web_shot = list(fist)
        web_shot[8] = (.40, .32, 0.0)
        web_shot[20] = (.69, .34, 0.0)
        self.assertEqual(detect_gesture(web_shot), Gesture.WEB_SHOT)
        self.assertFalse(is_retraction_pose(web_shot))
        self.assertFalse(is_retraction_pose([]), 'Missing landmarks are not a fist')

    def test_raw_fist_retracts_before_smoothed_gesture_catches_up_once(self):
        app = app_for_gestures()
        fist = hand_landmarks()
        web_shot = list(fist)
        web_shot[8] = (.40, .32, 0.0)
        web_shot[20] = (.69, .34, 0.0)
        app._process_gestures([HandState(handedness='Right', gesture=Gesture.WEB_SHOT,
                                         aim_landmarks=web_shot)], .016)
        app.web_system.retract.assert_not_called()
        app._process_gestures([HandState(handedness='Right', gesture=Gesture.WEB_SHOT,
                                         aim_landmarks=fist)], .016)
        app.web_system.retract.assert_called_once_with('Right')
        app.audio.play.assert_called_once_with('retract')
        # Slower smoothed classifier eventually changes to FIST: no second pull.
        app._process_gestures([HandState(handedness='Right', gesture=Gesture.FIST,
                                         aim_landmarks=fist)], .016)
        app.web_system.retract.assert_called_once()
        app.particles.emit_burst.assert_called_once()

    def test_stale_smoothed_web_shot_cannot_fire_again_during_raw_fist(self):
        app = app_for_gestures()
        app._prev_gestures['Right'] = {'gesture': Gesture.WEB_SHOT,
                                       'since': 0.0, 'fired': False,
                                       'retract_pose': False}
        app._web_latch = Mock()
        app._web_latch.update.return_value = True
        app._process_gestures([HandState(handedness='Right', gesture=Gesture.WEB_SHOT,
                                         aim_landmarks=hand_landmarks())], .016)
        app.web_system.launch.assert_not_called()
        app.audio.play.assert_called_once_with('retract')

    def test_stale_smoothed_fist_does_not_retract_after_raw_hand_opens(self):
        app = app_for_gestures()
        open_hand = list(hand_landmarks())
        for tip, xy in {8:(.40,.32), 12:(.50,.31), 16:(.56,.32),
                        20:(.69,.34)}.items():
            open_hand[tip] = (*xy, 0.0)
        app._process_gestures([HandState(handedness='Right', gesture=Gesture.FIST,
                                         aim_landmarks=open_hand)], .016)
        app.web_system.retract.assert_not_called()
        app.audio.play.assert_not_called()

    def test_raw_fist_never_retracts_outside_war_mode(self):
        app = app_for_gestures()
        app._war_mode = False
        app._process_gestures([HandState(handedness='Right', gesture=Gesture.THUMBS_UP,
                                         aim_landmarks=hand_landmarks())], .016)
        app.web_system.retract.assert_not_called()
        app.audio.play.assert_not_called()


if __name__ == '__main__':
    unittest.main()

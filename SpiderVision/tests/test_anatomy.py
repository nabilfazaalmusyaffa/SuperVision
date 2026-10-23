"""Standalone, educational anatomy overlay on a mirrored BGR camera frame."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from hud.anatomy import AnatomyHUD


def hand(x, y, gesture='POINT'):
    landmarks = [(0.0, 0.0, 0.0)] * 21
    landmarks[8] = (x, y, 0.0)
    return SimpleNamespace(landmarks=landmarks, gesture=gesture, handedness='Right')


class AnatomyTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.hud = AnatomyHUD()

    def test_closed_does_not_change_input(self):
        self.assertEqual(self.hud.mode, 'CLOSED')
        self.assertEqual(self.hud.zoom, 1.0)
        self.assertIsNone(self.hud.selected)
        self.assertIs(self.hud.render(self.frame, [], now=0), self.frame)
        self.assertEqual(np.count_nonzero(self.frame), 0)
        self.hud.open('BODY')
        self.hud.close()
        self.assertEqual(self.hud.mode, 'CLOSED')
        self.assertIs(self.hud.render(self.frame, [hand(.5, .5)], now=1), self.frame)

    def test_modes_draw_distinct_diagrams_and_safety_legend(self):
        expected = {
            'BODY': ('HEART', 'LUNGS', 'LIVER', 'STOMACH', 'KIDNEYS',
                     'SPINAL CORD', 'PERIPHERAL NERVES'),
            'BRAIN': ('FRONTAL LOBE', 'PARIETAL LOBE', 'TEMPORAL LOBE',
                      'OCCIPITAL LOBE', 'CEREBELLUM', 'BRAINSTEM', 'SPINAL CORD'),
        }
        images = []
        for mode, labels in expected.items():
            self.hud.open(mode)
            with patch('hud.anatomy.cv2.putText', wraps=cv2.putText) as text:
                image = self.hud.render(self.frame, [], now=0)
            drawn = [call.args[1] for call in text.call_args_list]
            for label in labels:
                self.assertIn(label, drawn)
            self.assertIn('EDUCATIONAL ILLUSTRATION - NOT A MEDICAL SCAN', drawn)
            self.assertGreater(np.count_nonzero(image[100:630, 380:815]), 500)
            self.assertEqual(np.count_nonzero(self.frame), 0)
            images.append(image)
        self.assertFalse(np.array_equal(*images))

    def test_selected_part_exposes_short_educational_description(self):
        self.hud.open('BODY')
        self.assertIsNone(self.hud.describe_selected())
        self.hud.render(self.frame, [hand(610 / 1280, 260 / 720, 'PINCH')], now=0)
        self.assertEqual(self.hud.selected, 'HEART')
        self.assertIn('blood', self.hud.describe_selected().lower())
        self.hud.open('BRAIN')
        self.assertIsNone(self.hud.describe_selected())

    def test_atlas_shows_how_to_get_and_hear_descriptions(self):
        self.hud.open('BRAIN')
        with patch('hud.anatomy.cv2.putText', wraps=cv2.putText) as text:
            self.hud.render(self.frame, [], now=0)
        labels = [call.args[1] for call in text.call_args_list]
        self.assertTrue(any('JARVIS DESCRIBE' in label for label in labels))

    def test_body_art_shows_detailed_head_and_both_feet_at_default_zoom(self):
        self.hud.open('BODY')
        image = self.hud.render(self.frame, [], now=0)
        # Full head-to-toe silhouette, including two separate feet, in the central crop.
        for x, y in ((600, 145), (573, 567), (650, 567)):
            self.assertGreater(int(image[y-4:y+5, x-4:x+5, 0].max()), 60,
                               f'Anatomy artwork missing near {(x, y)}')

    def test_heart_hotspot_follows_illustrated_chest_not_old_schematic(self):
        self.hud.open('BODY')
        self.hud.render(self.frame, [hand(610 / 1280, 260 / 720, 'PINCH')], now=0)
        self.assertEqual(self.hud.selected, 'HEART')

    def test_selected_heart_has_short_educational_explanation(self):
        self.hud.open('BODY')
        with patch('hud.anatomy.cv2.putText', wraps=cv2.putText) as text:
            self.hud.render(self.frame, [hand(610 / 1280, 260 / 720, 'PINCH')], now=0)
        captions = [call.args[1] for call in text.call_args_list]
        self.assertTrue(any('pumps blood' in caption.lower() for caption in captions))

    def test_peripheral_nerve_callout_uses_separate_arm_tracer(self):
        self.hud.open('BODY')
        image = self.hud.render(self.frame, [], now=0)
        # The arm artwork depicts bone/vessels, not an explicit peripheral nerve.
        self.assertGreater(int(image[339:344, 536:541, 0].max()), 180)

    def test_brain_art_has_cyan_not_yellow_tint(self):
        self.hud.open('BRAIN')
        image = self.hud.render(self.frame, [], now=0)
        # BGR: cyan needs more blue/green than red in the filled temporal lobe.
        blue, green, red = (int(v) for v in image[345, 550])
        self.assertGreater(blue, red + 40)
        self.assertGreater(green, red + 40)

    def test_scan_grid_uses_cool_holographic_tint(self):
        self.hud.open('BODY')
        image = self.hud.render(self.frame, [], now=0)
        blue, _, red = (int(v) for v in image[180, 380])
        self.assertGreater(blue, red + 10)

    def test_body_silhouette_has_subtle_glow_outside_head(self):
        self.hud.open('BODY')
        image = self.hud.render(self.frame, [], now=0)
        self.assertGreater(int(image[136, 600, 0]), 25)

    def test_unchanged_zoom_reuses_scaled_art_per_frame(self):
        self.hud.open('BRAIN')
        self.hud.zoom = 1.8
        self.hud.render(self.frame, [], now=0)
        with patch('hud.anatomy.cv2.resize', wraps=cv2.resize) as resize:
            self.hud.render(self.frame, [], now=1)
        self.assertEqual(resize.call_count, 0)

    def test_panel_blend_never_processes_entire_camera_frame(self):
        self.hud.open('BODY')
        with patch('hud.anatomy.cv2.addWeighted', wraps=cv2.addWeighted) as blend:
            self.hud.render(self.frame, [], now=0)
        self.assertTrue(blend.called)
        self.assertTrue(all(call.args[0].shape[1] <= 465 for call in blend.call_args_list))

    def test_brain_callouts_and_safety_notice_remain_readable(self):
        self.hud.open('BRAIN')
        with patch('hud.anatomy.cv2.putText', wraps=cv2.putText) as text:
            self.hud.render(self.frame, [], now=0)
        calls = {call.args[1]: call for call in text.call_args_list}
        for label in ('PARIETAL LOBE', 'OCCIPITAL LOBE', 'BRAINSTEM'):
            self.assertGreaterEqual(calls[label].args[2][0], 730)
            self.assertGreaterEqual(calls[label].args[4], .38)
        warning = calls['EDUCATIONAL ILLUSTRATION - NOT A MEDICAL SCAN']
        self.assertGreaterEqual(warning.args[4], .4)

    def test_invalid_mode_is_rejected_without_changing_state(self):
        with self.assertRaises(ValueError):
            self.hud.open('XRAY')
        self.assertEqual(self.hud.mode, 'CLOSED')

    def test_point_dwell_selects_heart_and_breaking_hover_resets_timer(self):
        self.hud.open('BODY')
        heart = hand(610 / 1280, 260 / 720)
        self.hud.render(self.frame, [heart], now=0)
        self.hud.render(self.frame, [], now=.5)
        self.hud.render(self.frame, [heart], now=.6)
        self.assertIsNone(self.hud.selected)
        self.hud.render(self.frame, [heart], now=1.3)
        self.assertEqual(self.hud.selected, 'HEART')

    def test_fingertip_pointer_is_visible_in_atlas(self):
        self.hud.open('BODY')
        without = self.hud.render(self.frame, [], now=0)
        with_hand = self.hud.render(self.frame, [hand(400 / 1280, 550 / 720)], now=.1)
        self.assertGreater(np.count_nonzero(with_hand[540:560, 390:410] !=
                                            without[540:560, 390:410]), 25)

    def test_pinch_selects_lobe_but_not_empty_space(self):
        self.hud.open('BRAIN')
        self.hud.render(self.frame, [hand(.9, .8, 'PINCH')], now=0)
        self.assertIsNone(self.hud.selected)
        self.hud.render(self.frame, [hand(541 / 1280, 279 / 720)], now=.1)
        self.hud.render(self.frame, [hand(541 / 1280, 279 / 720, 'PINCH')], now=.2)
        self.assertEqual(self.hud.selected, 'FRONTAL LOBE')
        self.hud.open('BODY')
        self.assertIsNone(self.hud.selected)

    def test_two_hand_spread_and_pinch_zoom_clamps_and_hit_regions_follow(self):
        self.hud.open('BODY')
        def pair(distance):
            return [hand((600 - distance / 2) / 1280, 355 / 720),
                    hand((600 + distance / 2) / 1280, 355 / 720)]
        self.hud.render(self.frame, pair(100), now=0)
        self.hud.render(self.frame, pair(400), now=.1)
        self.assertEqual(self.hud.zoom, 2.5)
        # Liver anchor follows art from (585, 300) to (562.5, 217.5).
        self.hud.render(self.frame, [hand(562.5 / 1280, 217.5 / 720, 'PINCH')], now=.2)
        self.assertEqual(self.hud.selected, 'LIVER')
        self.hud.render(self.frame, pair(400), now=.3)
        self.hud.render(self.frame, pair(10), now=.4)
        self.assertEqual(self.hud.zoom, 1.0)
        self.hud.close()
        self.assertEqual(self.hud.zoom, 1.0)

    def test_missing_landmarks_and_no_hands_are_safe(self):
        self.hud.open('BRAIN')
        for hands in (None, [SimpleNamespace(landmarks=[], gesture='POINT')],
                      [hand(float('nan'), .5)]):
            self.assertEqual(self.hud.render(self.frame, hands).shape, self.frame.shape)
        self.assertIsNone(self.hud.selected)

    def test_invalid_first_hand_does_not_block_valid_pinch(self):
        self.hud.open('BODY')
        invalid = SimpleNamespace(landmarks=[], gesture='POINT')
        self.hud.render(self.frame, [invalid, hand(610 / 1280, 260 / 720, 'PINCH')], now=1)
        self.assertEqual(self.hud.selected, 'HEART')

    def test_edge_lobes_remain_selectable_at_max_zoom(self):
        self.hud.open('BRAIN')
        self.hud.zoom = 2.5
        self.hud.render(self.frame, [hand(787.5 / 1280, 187.5 / 720, 'PINCH')], now=0)
        self.assertEqual(self.hud.selected, 'OCCIPITAL LOBE')

    def test_offscreen_brain_hotspot_is_not_pinned_to_crop_border(self):
        self.hud.open('BRAIN')
        self.hud.zoom = 2.5
        # Parietal anchor at y=234 moves above the illustration crop.
        self.hud.render(self.frame, [hand(635 / 1280, 140 / 720, 'PINCH')], now=0)
        self.assertIsNone(self.hud.selected)

    def test_brain_spinal_cord_anchor_touches_stem_continuation(self):
        self.hud.open('BRAIN')
        anchor = next(pos for name, pos, _ in self.hud._BRAIN if name == 'SPINAL CORD')
        self.assertEqual(anchor, (653, 463))
        self.hud.render(self.frame, [hand(653 / 1280, 463 / 720, 'PINCH')], now=0)
        self.assertEqual(self.hud.selected, 'SPINAL CORD')

    def test_zoom_property_itself_is_bounded(self):
        self.hud.zoom = 100
        self.assertEqual(self.hud.zoom, 2.5)
        self.hud.zoom = -100
        self.assertEqual(self.hud.zoom, 1.0)

    def test_zoomed_diagram_does_not_cover_side_panels(self):
        self.hud.open('BRAIN')
        self.hud.zoom = 2.5
        result = self.hud.render(self.frame, [], now=1)
        self.assertEqual(np.count_nonzero(result[:, :350]), 0)
        self.assertEqual(np.count_nonzero(result[:, 830:]), 0)

"""Regression checks for borderless laptop windows and camera-space pulse HUD."""
import unittest
from unittest.mock import patch

import glfw
import cv2
import numpy as np

from hud.vitals import VitalsHUD
from config.settings import WINDOW_H, WINDOW_W
from core.renderer import Renderer


class LaptopWindowTests(unittest.TestCase):
    def test_borderless_window_fits_offset_workarea_without_changing_render_resolution(self):
        renderer = Renderer.__new__(Renderer)
        window = object()
        monitor = object()
        with patch.object(glfw, 'init', return_value=True), \
             patch.object(glfw, 'get_primary_monitor', return_value=monitor), \
             patch.object(glfw, 'get_monitor_workarea', return_value=(0, 25, 1440, 802)) as workarea, \
             patch.object(glfw, 'get_video_mode', side_effect=AssertionError('unsafe native call')) as video_mode, \
             patch.object(glfw, 'window_hint'), \
             patch.object(glfw, 'create_window', return_value=window) as create, \
             patch.object(glfw, 'set_window_pos') as position, \
             patch.object(glfw, 'get_framebuffer_size', return_value=(1425, 802)), \
             patch.object(glfw, 'make_context_current'), \
             patch.object(glfw, 'swap_interval'), \
             patch.object(glfw, 'set_key_callback'):
            renderer._init_glfw()
        workarea.assert_called_once_with(monitor)
        video_mode.assert_not_called()
        width, height, _, _, _ = create.call_args.args
        x, y = position.call_args.args[1:]
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)
        self.assertLessEqual(x + width, 1440)
        self.assertLessEqual(y + height, 827)
        self.assertEqual((x, y), (0, 25))
        self.assertAlmostEqual(width / height, WINDOW_W / WINDOW_H, delta=0.002)
        self.assertEqual((renderer.width, renderer.height), (WINDOW_W, WINDOW_H))
        self.assertEqual((renderer.fb_width, renderer.fb_height), (1425, 802))

    def test_workarea_probe_failure_falls_back_without_video_mode(self):
        renderer = Renderer.__new__(Renderer)
        window = object()
        with patch.object(glfw, 'init', return_value=True), \
             patch.object(glfw, 'get_primary_monitor', return_value=object()), \
             patch.object(glfw, 'get_monitor_workarea', side_effect=RuntimeError('no workarea')), \
             patch.object(glfw, 'get_video_mode', side_effect=AssertionError('unsafe native call')) as video_mode, \
             patch.object(glfw, 'window_hint'), \
             patch.object(glfw, 'create_window', return_value=window) as create, \
             patch.object(glfw, 'set_window_pos') as position, \
             patch.object(glfw, 'get_framebuffer_size', return_value=(1920, 1080)), \
             patch.object(glfw, 'make_context_current'), \
             patch.object(glfw, 'swap_interval'), \
             patch.object(glfw, 'set_key_callback'):
            renderer._init_glfw()
        self.assertEqual(create.call_args.args[:2], (WINDOW_W, WINDOW_H))
        position.assert_called_once_with(window, 0, 0)
        video_mode.assert_not_called()


class VitalsLayoutTests(unittest.TestCase):
    def test_panel_moves_below_performance_area_on_standard_frame(self):
        self._assert_visible_panel(1280, 720, min_y=190, max_y=235)

    def test_small_frame_keeps_panel_and_all_text_inside(self):
        self._assert_visible_panel(640, 360, min_y=160, max_y=190)

    def _assert_visible_panel(self, width, height, min_y, max_y):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        rectangles = []
        texts = []
        original_rectangle = cv2.rectangle
        original_text = cv2.putText

        def record_rectangle(image, start, end, *args, **kwargs):
            rectangles.append((start, end))
            return original_rectangle(image, start, end, *args, **kwargs)

        def record_text(image, text, origin, font, scale, *args, **kwargs):
            text_width, text_height = cv2.getTextSize(text, font, scale, args[1] if len(args) > 1 else 1)[0]
            texts.append((text, origin, text_width, text_height))
            return original_text(image, text, origin, font, scale, *args, **kwargs)

        with patch.object(cv2, 'rectangle', side_effect=record_rectangle), \
             patch.object(cv2, 'putText', side_effect=record_text):
            output = VitalsHUD().render(frame, 0.0)
        self.assertTrue(rectangles, 'panel border should be visible')
        (left, top), (right, bottom) = rectangles[0]
        self.assertLess(left, width // 3)
        self.assertGreaterEqual(top, min_y)
        self.assertLessEqual(top, max_y)
        self.assertGreaterEqual(left, 0)
        self.assertLess(right, width)
        self.assertLess(bottom, height)
        self.assertIn(VitalsHUD.LABEL, [text for text, *_ in texts])
        self.assertEqual(len(texts), 3)
        for text, (x, y), text_width, text_height in texts:
            with self.subTest(text=text, frame=(width, height)):
                self.assertGreaterEqual(x, left)
                self.assertLessEqual(x + text_width, right)
                self.assertGreaterEqual(y - text_height, top)
                self.assertLessEqual(y, bottom)
        self.assertTrue(np.any(output[top:bottom, left:right]))
        self.assertFalse(np.any(frame), 'render must not mutate input')


if __name__ == '__main__':
    unittest.main()

import unittest
import time
import glfw
import glm
import numpy as np
from unittest.mock import Mock
from unittest.mock import patch

from main import SpiderVisionApp
from tracking.gesture import Gesture, HandState
from hud.web_aim import WebAimHUD
from config.settings import MENU_ITEMS, AppState
from tracking.face_tracker import FaceData


class VoiceAppIntegrationTests(unittest.TestCase):
    def test_runtime_imports_real_anatomy_overlay(self):
        import main
        from hud.anatomy import AnatomyHUD
        self.assertIs(main.AnatomyHUD, AnatomyHUD)

    def new_app(self):
        app = SpiderVisionApp.__new__(SpiderVisionApp)
        app.menu = Mock()
        app.menu.is_open = False
        app.audio = Mock(muted=False)
        app.screenshot = Mock(is_active=False)
        app.identity = Mock()
        app.voice = Mock()
        app.show_performance = True
        app.show_particles = True
        app.web_aim_hud = WebAimHUD()
        app.anatomy = Mock(mode='CLOSED', zoom=1.0)
        app.web_system = Mock()
        app._war_mode = False
        app._exit_after_speech = None
        app._pending_audio_mute = False
        return app

    def test_menu_responds_to_voice_after_authorization_only(self):
        app = self.new_app()
        app.voice.pop_command.side_effect = ['MENU', None, 'MENU', None]
        app._process_voice_commands(False)
        app.menu.show.assert_not_called()
        app._process_voice_commands(True)
        app.menu.show.assert_called_once()
        self.assertTrue(app.audio.speak.called)

    def test_voice_selects_menu_option_and_replies(self):
        app = self.new_app()
        app.voice.pop_command.side_effect = ['PERFORMANCE', None]
        app._process_voice_commands(True)
        self.assertFalse(app.show_performance)
        app.audio.speak.assert_called_once()

    def test_jarvis_aim_toggles_mode_only_after_face_verification(self):
        app = self.new_app()
        app.voice.pop_command.side_effect = ['WEB AIM', None, 'WEB AIM', None, 'WEB AIM', None]
        app._process_voice_commands(False)
        self.assertFalse(app.web_aim_hud.enabled)
        app._process_voice_commands(True)
        self.assertTrue(app.web_aim_hud.enabled)
        app.audio.speak.assert_called_with('Web aim on, Mister.')
        app._process_voice_commands(True)
        self.assertFalse(app.web_aim_hud.enabled)
        app.audio.speak.assert_called_with('Web aim off, Mister.')

    def test_web_aim_is_selectable_by_sixth_keyboard_menu_option(self):
        self.assertEqual(MENU_ITEMS[5], 'WEB AIM')
        app = self.new_app()
        app._face_data = FaceData()
        app._face_data.session_unlocked = True
        app.menu.is_open = True
        app._on_key(glfw.KEY_6, glfw.PRESS, 0)
        self.assertTrue(app.web_aim_hud.enabled)

    def test_voice_body_and_brain_open_educational_atlas_after_verification(self):
        app = self.new_app()
        app.voice.pop_command.side_effect = ['BODY', None, 'BODY', None, 'BRAIN', None]
        app._process_voice_commands(False)
        app.anatomy.open.assert_not_called()
        app._process_voice_commands(True)
        app.anatomy.open.assert_called_once_with('BODY')
        app._process_voice_commands(True)
        app.anatomy.open.assert_called_with('BRAIN')
        self.assertIn('Educational', app.audio.speak.call_args.args[0])

    def test_voice_zoom_and_close_anatomy(self):
        app = self.new_app()
        app.anatomy.mode = 'BRAIN'
        app.voice.pop_command.side_effect = ['ZOOM IN', None, 'ZOOM OUT', None, 'CLOSE ANATOMY', None]
        app._process_voice_commands(True)
        self.assertGreater(app.anatomy.zoom, 1.0)
        app._process_voice_commands(True)
        self.assertAlmostEqual(app.anatomy.zoom, 1.0)
        app._process_voice_commands(True)
        app.anatomy.close.assert_called_once()

    def test_spoken_close_dismisses_either_open_atlas_not_just_menu(self):
        from hud.anatomy import AnatomyHUD
        from sound.voice_commands import VoiceCommandListener
        for mode in ('BODY', 'BRAIN'):
            with self.subTest(mode=mode):
                app = self.new_app()
                app.anatomy = AnatomyHUD()
                app.anatomy.open(mode)
                app.voice = VoiceCommandListener()
                app.voice.accept_text('close menu')
                app._process_voice_commands(True)
                self.assertEqual(app.anatomy.mode, 'CLOSED')
                app.audio.speak.assert_called_with('Anatomy view closed, Mister.')

    def test_jarvis_describe_explains_selected_part_and_handles_no_selection(self):
        from hud.anatomy import AnatomyHUD
        from sound.voice_commands import VoiceCommandListener
        app = self.new_app()
        app.anatomy = AnatomyHUD()
        app.anatomy.open('BODY')
        app.voice = VoiceCommandListener()
        app.voice.accept_text('Jarvis describe')
        app._process_voice_commands(True)
        self.assertIn('Point to', app.audio.speak.call_args.args[0])
        app.anatomy.selected = 'HEART'
        app.voice.accept_text('Jarvis describe')
        app._process_voice_commands(True)
        self.assertIn('blood', app.audio.speak.call_args.args[0].lower())

    def test_jarvis_status_reports_open_atlas_without_fabricating_health_data(self):
        app = self.new_app()
        app.anatomy.mode = 'BRAIN'
        app.voice.pop_command.side_effect = ['STATUS', None]
        app._process_voice_commands(True)
        self.assertEqual(app.audio.speak.call_args.args[0],
                         'Spider Vision online. brain atlas open.')

    def test_jarvis_help_explains_map_command_without_claiming_gps(self):
        app = self.new_app()
        app.voice.pop_command.side_effect = ['HELP', None]
        app._process_voice_commands(True)
        self.assertIn('map', app.audio.speak.call_args.args[0].lower())

    def test_war_mode_requires_verified_face_and_has_distinct_replies(self):
        from sound.voice_commands import VoiceCommandListener
        app = self.new_app()
        app.voice = VoiceCommandListener()
        app.voice.accept_text('Jarvis war mode on')
        app._process_voice_commands(False)
        self.assertFalse(app._war_mode)
        app.voice.accept_text('Jarvis war mode on')
        app._process_voice_commands(True)
        self.assertTrue(app._war_mode)
        self.assertIn('visual', app.audio.speak.call_args.args[0].lower())
        app.voice.accept_text('Jarvis war mode off')
        app._process_voice_commands(True)
        self.assertFalse(app._war_mode)
        app.web_system.clear_war_webs.assert_called_once()
        self.assertIn('off', app.audio.speak.call_args.args[0].lower())

    def test_war_mode_clears_when_face_session_locks(self):
        app = self.new_app()
        app._war_mode = True
        app.voice.pop_command.return_value = None
        app._process_voice_commands(False)
        self.assertFalse(app._war_mode)
        app.web_system.clear_war_webs.assert_called_once()

    def test_war_mode_does_not_start_over_open_anatomy(self):
        app = self.new_app()
        app.anatomy.mode = 'BODY'
        app.voice.pop_command.side_effect = ['WAR MODE ON', None]
        app._process_voice_commands(True)
        self.assertFalse(app._war_mode)
        self.assertIn('close', app.audio.speak.call_args.args[0].lower())

    def test_war_hud_only_renders_for_verified_unobstructed_session(self):
        from hud.war_mode import WarModeHUD
        app = self.new_app()
        app.war_hud = WarModeHUD()
        app.state = AppState.IDLE
        frame = np.zeros((720, 1280, 3), np.uint8)
        self.assertIs(app._render_war_mode(frame, True, 0), frame)
        app._war_mode = True
        self.assertIs(app._render_war_mode(frame, False, 0), frame)
        app.anatomy.mode = 'BRAIN'
        self.assertIs(app._render_war_mode(frame, True, 0), frame)
        app.anatomy.mode = 'CLOSED'
        result = app._render_war_mode(frame, True, 0)
        self.assertGreater(result[15, 15, 2], 100)
        self.assertEqual(np.count_nonzero(frame), 0)

    def test_fired_web_inherits_war_visual_mode_without_changing_normal_mode(self):
        app = self.new_app()
        app._last_web_shot = {}
        app.hand_tracker = Mock()
        app.cam_effect = Mock()
        app.particles = Mock()
        app.web_aim_hud = WebAimHUD()
        hand = HandState(handedness='Right', aim_landmarks=[(0.5, 0.5, 0)] * 21)
        app._fire_web('Right', hand)
        self.assertFalse(app.web_system.launch.call_args.kwargs['war_mode'])
        app._last_web_shot.clear()
        app._war_mode = True
        app._fire_web('Right', hand)
        self.assertTrue(app.web_system.launch.call_args.kwargs['war_mode'])

    def test_new_fist_retracts_same_hand_web_once_only_in_war_mode(self):
        from tracking.gesture import GestureLatch
        app = self.new_app()
        app._war_mode = True
        app._prev_gestures = {}
        app._last_web_shot = {}
        app._web_latch = GestureLatch(hold_seconds=0.08)
        app.particles = Mock()
        hand = HandState(handedness='Left', gesture=Gesture.FIST)
        app._process_gestures([hand], 0.016)
        app._process_gestures([hand], 0.016)
        app.web_system.retract.assert_called_once_with('Left')
        app._war_mode = False
        app._process_gestures([], 0.016)
        app._process_gestures([hand], 0.016)
        app.web_system.retract.assert_called_once()

    def test_successful_fist_retraction_has_immediate_audio_and_burst_once(self):
        from effects.web_projectile import WebProjectile, WebProjectileSystem
        from tracking.gesture import GestureLatch
        app = self.new_app()
        app._war_mode = True
        app._prev_gestures = {}
        app._last_web_shot = {}
        app._web_latch = GestureLatch(hold_seconds=0.08)
        app.particles = Mock()
        system = WebProjectileSystem.__new__(WebProjectileSystem)
        web = WebProjectile(glm.vec2(480, 540), glm.vec2(1, 0), 'Right', war_mode=True)
        system._webs = [web]
        system.update(0.5)
        system.update(3.0)
        app.web_system = system
        hand = HandState(handedness='Right', gesture=Gesture.FIST)
        app._process_gestures([hand], 0.016)
        self.assertTrue(web.retracting)
        app.audio.play.assert_called_once_with('retract')
        app.particles.emit_burst.assert_called_once()
        app._process_gestures([hand], 0.016)
        app.audio.play.assert_called_once()
        app.particles.emit_burst.assert_called_once()

    def test_unsuccessful_fist_does_not_play_false_retraction_sound(self):
        from tracking.gesture import GestureLatch
        app = self.new_app()
        app._war_mode = True
        app._prev_gestures = {}
        app._last_web_shot = {}
        app._web_latch = GestureLatch(hold_seconds=0.08)
        app.particles = Mock()
        app.web_system.retract.return_value = False
        app._process_gestures([HandState(handedness='Left', gesture=Gesture.FIST)], 0.016)
        app.audio.play.assert_not_called()
        app.particles.emit_burst.assert_not_called()

    def test_map_location_request_requires_a_second_explicit_confirmation(self):
        from sound.voice_commands import VoiceCommandListener
        from hud.location_map import LocationMapHUD
        app = self.new_app()
        app.voice = VoiceCommandListener()
        app.location_map = LocationMapHUD(fetch=Mock())
        app._map_visible = True
        app._map_consent_pending = False
        app._map_generation = 0
        app._start_map_lookup = Mock()
        app.voice.accept_text('Jarvis allow location')
        app._process_voice_commands(True)
        app._start_map_lookup.assert_not_called()
        app.voice.accept_text('Jarvis locate me')
        app._process_voice_commands(True)
        self.assertTrue(app._map_consent_pending)
        self.assertIn('public IP', app.audio.speak.call_args.args[0])
        app._start_map_lookup.assert_not_called()
        app.voice.accept_text('Jarvis allow location')
        app._process_voice_commands(True)
        app._start_map_lookup.assert_called_once()
        self.assertFalse(app._map_consent_pending)

    def test_manual_area_override_shows_map_without_ip_lookup(self):
        from sound.voice_commands import VoiceCommandListener
        from hud.location_map import LocationMapHUD
        app = self.new_app()
        app.voice = VoiceCommandListener()
        app.location_map = LocationMapHUD(fetch=Mock())
        app._map_visible = False
        app._map_consent_pending = False
        app._start_map_lookup = Mock()
        with patch.dict('os.environ', {'SPIDERVISION_MAP_COORDS': '-6.2,106.8'}):
            app.voice.accept_text('Jarvis map')
            app._process_voice_commands(True)
        self.assertTrue(app._map_visible)
        self.assertEqual(app.location_map._location[:2], (-6.2, 106.8))
        self.assertEqual(app.location_map._location[2], 'USER CHOSEN')
        app._start_map_lookup.assert_not_called()
        self.assertIn('not GPS', app.audio.speak.call_args.args[0])

    def test_hidden_or_unverified_map_never_paints_location(self):
        from hud.location_map import LocationMapHUD
        app = self.new_app()
        app.state = AppState.IDLE
        app.location_map = LocationMapHUD(fetch=Mock())
        app._map_visible = True
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.assertIs(app._render_location_map(frame, False), frame)
        app.menu.is_open = True
        self.assertIs(app._render_location_map(frame, True), frame)
        app.menu.is_open = False
        self.assertGreater(np.count_nonzero(app._render_location_map(frame.copy(), True)), 100)

    def test_approved_lookup_runs_off_render_thread_and_uses_coarse_result(self):
        from hud.location_map import LocationMapHUD
        app = self.new_app()
        app.location_map = LocationMapHUD(fetch=Mock())
        app._map_generation = 0
        app._map_lookup_running = False
        app._map_reply = None
        app._map_visible = True
        with patch('main.threading.Thread') as thread:
            app._start_map_lookup()
        thread.assert_called_once()
        self.assertTrue(thread.call_args.kwargs['daemon'])
        self.assertTrue(app._map_lookup_running)
        with patch('main.IPAreaProvider') as provider:
            provider.return_value.request_location.return_value = {
                'lat': 37.8, 'lon': -122.4, 'accuracy_m': None, 'error': None}
            app._fetch_map_area(app._map_generation)
            provider.return_value.request_location.assert_called_once_with(opt_in=True)
        self.assertEqual(app.location_map._location[:2], (37.8, -122.4))
        self.assertEqual(app.location_map._location[2], 'IP ESTIMATE')
        self.assertFalse(app._map_lookup_running)
        app._finish_map_lookup(True)
        self.assertIn('not GPS', app.audio.speak.call_args.args[0])

    def test_location_result_is_discarded_after_map_hidden_or_face_locked(self):
        from hud.location_map import LocationMapHUD
        app = self.new_app()
        app.location_map = LocationMapHUD(fetch=Mock())
        app._map_generation = 1
        app._map_visible = True
        app._map_was_unlocked = True
        app._map_consent_pending = True
        app.state = AppState.IDLE
        app._render_location_map(np.zeros((720, 1280, 3), np.uint8), False)
        self.assertFalse(app._map_consent_pending)
        self.assertEqual(app.location_map.state, 'inactive')
        with patch('main.IPAreaProvider') as provider:
            provider.return_value.request_location.return_value = {
                'lat': 37.8, 'lon': -122.4, 'accuracy_m': None, 'error': None}
            app._fetch_map_area(1)
        self.assertEqual(app.location_map.state, 'inactive')

    def test_hide_map_clears_active_area_and_discloses_cached_tiles(self):
        from hud.location_map import LocationMapHUD
        from sound.voice_commands import VoiceCommandListener
        app = self.new_app()
        app.location_map = LocationMapHUD(fetch=Mock())
        app.location_map.set_location(41.9, 12.5, 'USER CHOSEN')
        app._map_visible = True
        app._map_generation = 1
        app._map_consent_pending = True
        app._map_lookup_running = True
        app._map_reply = (1, 'Old reply')
        app.voice = VoiceCommandListener()
        app.voice.accept_text('Jarvis hide map')
        app._process_voice_commands(True)
        self.assertFalse(app._map_visible)
        self.assertEqual(app.location_map.state, 'inactive')
        self.assertIsNone(app._map_reply)
        self.assertIn('cached', app.audio.speak.call_args.args[0].lower())

    def test_menu_exposes_body_and_brain_before_exit(self):
        self.assertEqual(MENU_ITEMS[-3:], ['BODY ATLAS', 'BRAIN ATLAS', 'EXIT'])
        app = self.new_app()
        app._face_data = FaceData()
        app._face_data.session_unlocked = True
        app.menu.is_open = True
        app._on_key(glfw.KEY_7, glfw.PRESS, 0)
        app.anatomy.open.assert_called_with('BODY')

    def test_backspace_closes_atlas_and_space_cannot_fire_during_atlas(self):
        app = self.new_app()
        app._face_data = FaceData()
        app._face_data.session_unlocked = True
        app.anatomy.mode = 'BODY'
        app._fire_web = Mock()
        app._on_key(glfw.KEY_SPACE, glfw.PRESS, 0)
        app._fire_web.assert_not_called()
        app._on_key(glfw.KEY_BACKSPACE, glfw.PRESS, 0)
        app.anatomy.close.assert_called_once()

    def test_anatomy_render_is_face_gated_and_closes_when_session_locks(self):
        app = self.new_app()
        app.state = AppState.IDLE
        app.anatomy.mode = 'BODY'
        app.anatomy.render.side_effect = lambda frame, hands: np.full_like(frame, 255)
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertTrue(np.any(app._render_anatomy(frame, [], True)))
        self.assertTrue(np.array_equal(app._render_anatomy(frame, [], False), frame))
        app.anatomy.close.assert_called_once()

    def test_anatomy_hides_combat_hud_so_diagrams_remain_visible(self):
        app = self.new_app()
        self.assertTrue(app._combat_hud_visible(True))
        app.anatomy.mode = 'BODY'
        self.assertFalse(app._combat_hud_visible(True))
        app.anatomy.mode = 'BRAIN'
        self.assertFalse(app._combat_hud_visible(True))

    def test_open_palm_never_opens_hand_menu(self):
        app = self.new_app()
        app._prev_gestures = {'Right': {'gesture': Gesture.PALM_OPEN,
                                        'since': time.monotonic() - 1.0, 'fired': False}}
        app._web_latch = Mock()
        app._web_latch.update.return_value = False
        app._last_web_shot = {}
        app.particles = Mock()
        hand = Mock(handedness='Right', gesture=Gesture.PALM_OPEN)
        app._process_gestures([hand], 1.0)
        app.menu.show.assert_not_called()

    def test_anatomy_hand_navigation_does_not_accidentally_fire_webs(self):
        app = self.new_app()
        app.anatomy.mode = 'BODY'
        app._prev_gestures = {}
        app._web_latch = Mock()
        app._web_latch.update.return_value = True
        app._last_web_shot = {}
        app.particles = Mock()
        app.web_system = Mock()
        app.cam_effect = Mock()
        hand = HandState(gesture=Gesture.WEB_SHOT,
                         landmarks=[(0.5, 0.5, 0.0)] * 21)
        app._process_gestures([hand], 0.016)
        app.web_system.launch.assert_not_called()

    def test_voice_exit_waits_for_reply_instead_of_closing_immediately(self):
        app = self.new_app()
        app.renderer = Mock()
        app.voice.pop_command.side_effect = ['EXIT', None]
        app._process_voice_commands(True)
        self.assertIsNotNone(app._exit_after_speech)
        app.audio.speak.assert_called_with('Closing Spider Vision. Goodbye, Mister.')

    def test_voice_audio_off_answers_before_muting(self):
        app = self.new_app()
        app.voice.pop_command.side_effect = ['AUDIO', None]
        app._process_voice_commands(True)
        self.assertFalse(app.audio.muted)
        self.assertTrue(app._pending_audio_mute)
        app.audio.speak.assert_called_with('Audio off, Mister.')

    def test_pending_audio_mutes_only_after_reply_ends(self):
        app = self.new_app()
        app._pending_audio_mute = True
        app.audio.is_speaking = True
        app._finish_voice_replies()
        self.assertFalse(app.audio.muted)
        app.audio.is_speaking = False
        app._finish_voice_replies()
        self.assertTrue(app.audio.muted)

    def test_exit_waits_for_speech_then_closes(self):
        app = self.new_app()
        app.renderer = Mock(window=object())
        app._exit_after_speech = (time.monotonic() - 1, time.monotonic() + 10)
        app.audio.is_speaking = True
        with patch('main.glfw.set_window_should_close') as close:
            app._finish_voice_replies()
            close.assert_not_called()
            app.audio.is_speaking = False
            app._finish_voice_replies()
            close.assert_called_once_with(app.renderer.window, True)

    def test_unverified_voice_command_explains_why_menu_did_not_open(self):
        app = self.new_app()
        app.voice.pop_command.side_effect = ['MENU', None]
        app._process_voice_commands(False)
        app.menu.show.assert_not_called()
        self.assertEqual(app._voice_notice, 'VERIFY FACE FIRST')

    def test_voice_status_distinguishes_model_loading_from_microphone_failure(self):
        app = self.new_app()
        app._voice_notice = ''
        app.audio.is_speaking = False
        app.voice.available = False
        app.voice.error = None
        self.assertEqual(app._voice_status_text(False), 'MIC LOADING')
        app.voice.error = 'permission denied'
        self.assertEqual(app._voice_status_text(False), 'MIC OFF - PRESS M')
        app.voice.available = True
        app.voice.last_feedback = 'MIC READY'
        self.assertEqual(app._voice_status_text(True), 'MIC READY')


if __name__ == '__main__':
    unittest.main()

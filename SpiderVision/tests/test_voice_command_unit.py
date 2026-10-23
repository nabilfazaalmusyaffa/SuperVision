"""Offline command and response behavior; no microphone or audio device required."""
import threading
import time
import unittest
import sys
import types
import asyncio
from unittest.mock import patch, Mock

from sound.voice_commands import VoiceCommandListener, parse_command, _GRAMMAR
from sound.audio_manager import AudioManager


class ParseCommandTests(unittest.TestCase):
    def test_wake_word_and_supported_commands(self):
        cases = {
            'Jarvis menu': 'MENU', 'Hey Jarvis show the menu': 'MENU',
            'Jarvis close menu': 'CLOSE', 'Jarvis audio': 'AUDIO',
            'Jarvis performance': 'PERFORMANCE', 'Jarvis effects': 'EFFECTS',
            'Jarvis screenshot': 'SCREENSHOT', 'Jarvis enroll face': 'ENROLL FACE',
            'Jarvis exit': 'EXIT', 'Jarvis help': 'HELP',
            'Jarvis aim': 'WEB AIM', 'Jarvis web aim': 'WEB AIM',
            'Jarvis body': 'BODY', 'Jarvis anatomy': 'BODY',
            'Jarvis brain': 'BRAIN', 'Jarvis close anatomy': 'CLOSE ANATOMY',
            'Jarvis zoom in': 'ZOOM IN', 'Jarvis zoom out': 'ZOOM OUT',
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_command(text), expected)

    def test_no_wake_word_or_unknown_command(self):
        for text in ('audio', 'aim', 'brain', 'body', 'exit', 'jarvis nonsense', 'notjarvis menu', 'jarvis', ''):
            with self.subTest(text=text):
                self.assertIsNone(parse_command(text))

    def test_aim_phrases_reach_constrained_microphone_recognizer(self):
        self.assertIn('jarvis aim', _GRAMMAR)
        self.assertIn('jarvis web aim', _GRAMMAR)

    def test_anatomy_phrases_reach_microphone_recognizer(self):
        for phrase in ('jarvis body', 'jarvis brain', 'jarvis anatomy',
                       'jarvis close anatomy', 'jarvis zoom in', 'jarvis zoom out'):
            self.assertIn(phrase, _GRAMMAR)

    def test_natural_close_phrases_are_recognizable_while_anatomy_is_open(self):
        for phrase, expected in (('close', 'CLOSE'), ('Jarvis close', 'CLOSE'),
                                 ('Jarvis close body', 'CLOSE ANATOMY'),
                                 ('Jarvis close brain', 'CLOSE ANATOMY'),
                                 ('Jarvis close atlas', 'CLOSE ANATOMY')):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase.lower(), _GRAMMAR)
                self.assertEqual(parse_command(phrase), expected)

    def test_assistant_description_phrases_reach_microphone_and_parser(self):
        for phrase in ('jarvis describe', 'jarvis tell me about this',
                       'jarvis what can you do', 'jarvis status'):
            self.assertIn(phrase, _GRAMMAR)
        self.assertEqual(parse_command('Jarvis describe'), 'DESCRIBE')
        self.assertEqual(parse_command('Jarvis tell me about this'), 'DESCRIBE')
        self.assertEqual(parse_command('Jarvis what can you do'), 'HELP')
        self.assertEqual(parse_command('Jarvis status'), 'STATUS')

    def test_map_controls_are_in_real_speech_grammar(self):
        for phrase, command in (('jarvis map', 'SHOW MAP'),
                                ('jarvis show map', 'SHOW MAP'),
                                ('jarvis hide map', 'HIDE MAP'),
                                ('jarvis locate me', 'LOCATE'),
                                ('jarvis allow location', 'ALLOW LOCATION'),
                                ('jarvis clear location', 'HIDE MAP')):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, _GRAMMAR)
                self.assertEqual(parse_command(phrase), command)

    def test_war_mode_phrases_reach_microphone_and_parser(self):
        for phrase, expected in (('jarvis war mode on', 'WAR MODE ON'),
                                 ('jarvis war mode off', 'WAR MODE OFF')):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, _GRAMMAR)
                self.assertEqual(parse_command(phrase), expected)
        self.assertIsNone(parse_command('war mode on'))

    def test_listener_accept_text_queue(self):
        listener = VoiceCommandListener(audio=None)
        listener.accept_text('Jarvis menu')
        listener.accept_text('menu')
        self.assertEqual(listener.pop_command(), 'MENU')
        self.assertEqual(listener.pop_command(), 'MENU')
        self.assertIsNone(listener.pop_command())

    def test_listener_ignores_self_heard_voice(self):
        audio = Mock(is_speaking=True)
        listener = VoiceCommandListener(audio)
        listener.accept_text('Jarvis exit')
        self.assertIsNone(listener.pop_command())
        audio.is_speaking = False
        listener.accept_text('Jarvis help')
        self.assertEqual(listener.pop_command(), 'HELP')


class MicrophoneTests(unittest.TestCase):
    def test_missing_model_does_not_block_start_and_reports_error(self):
        listener = VoiceCommandListener(model_path='/no/such/model')
        listener.start()
        listener._thread.join(timeout=2)
        self.assertFalse(listener._thread.is_alive())
        self.assertFalse(listener.available)
        self.assertIn('model', listener.error.lower())
        listener.stop()

    def test_microphone_failure_reports_error(self):
        class FailedStream:
            def __init__(self, **kwargs):
                raise RuntimeError('microphone denied')
        fake_sd = types.SimpleNamespace(RawInputStream=FailedStream)
        fake_vosk = types.SimpleNamespace(Model=lambda path: object(),
                                          KaldiRecognizer=lambda model, rate: object())
        with patch.dict(sys.modules, {'sounddevice': fake_sd, 'vosk': fake_vosk}):
            listener = VoiceCommandListener()
            listener.start()
            listener._thread.join(timeout=2)
        self.assertEqual(listener.error, 'microphone denied')
        self.assertFalse(listener.available)

    def test_recognized_utterance_and_speech_suppression(self):
        audio = Mock(is_speaking=False)
        ready = threading.Event()
        class Stream:
            def __init__(self, **kwargs):
                self.callback = kwargs['callback']
            def __enter__(self):
                ready.set()
                return self
            def __exit__(self, *args):
                pass
        class Recognizer:
            def AcceptWaveform(self, samples):
                return True
            def Result(self):
                return '{"text":"Jarvis menu"}'
            def Reset(self):
                pass
        with patch.dict(sys.modules, {'sounddevice': types.SimpleNamespace(RawInputStream=Stream),
                                      'vosk': types.SimpleNamespace(Model=lambda p: object(),
                                                                    KaldiRecognizer=lambda m, r: Recognizer())}):
            listener = VoiceCommandListener(audio)
            listener.start()
            self.assertTrue(ready.wait(2))
            self.assertTrue(listener.available)
            audio.is_speaking = True
            listener._capture(b'noise', 1, None, None)
            self.assertIsNone(listener.pop_command())
            audio.is_speaking = False
            listener._capture(b'command', 1, None, None)
            deadline = time.monotonic() + 2
            command = None
            while time.monotonic() < deadline and command is None:
                command = listener.pop_command()
                time.sleep(0.01)
            listener.stop()
            self.assertEqual(command, 'MENU')
            self.assertFalse(listener._thread.is_alive())


class AudioResponseTests(unittest.TestCase):
    def test_macos_speech_uses_offline_engine_first(self):
        with patch('sound.audio_manager.sys.platform', 'darwin'), \
             patch('sound.audio_manager.subprocess.run') as run, \
             patch('edge_tts.Communicate') as edge:
            audio = AudioManager.__new__(AudioManager)
            asyncio.run(audio._tts_async('Ready'))
            run.assert_called_once_with(['say', '-v', 'Daniel', 'Ready'], check=True, timeout=25)
            edge.assert_not_called()

    def test_speech_is_fifo_and_busy_includes_pending_replies(self):
        entered = threading.Event()
        release = threading.Event()
        spoken = []
        async def fake_speak(self, text):
            spoken.append(text)
            entered.set()
            if text == 'first':
                await asyncio.to_thread(release.wait)

        with patch('sound.audio_manager.pygame.mixer.pre_init'), \
             patch('sound.audio_manager.pygame.mixer.init'), \
             patch('sound.audio_manager.pygame.mixer.set_num_channels'), \
             patch.object(AudioManager, '_load_sounds'), \
             patch.object(AudioManager, '_tts_async', fake_speak):
            audio = AudioManager()
            audio.speak('first')
            self.assertTrue(entered.wait(2))
            audio.speak('second')
            audio.speak('third')
            self.assertTrue(audio.is_speaking)
            release.set()
            audio._tts_thread.join(timeout=2)
            self.assertEqual(spoken, ['first', 'second', 'third'])
            self.assertFalse(audio.is_speaking)


if __name__ == '__main__':
    unittest.main()

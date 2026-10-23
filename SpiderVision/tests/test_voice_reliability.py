"""Reliability and safety contract for the offline menu listener."""
import unittest
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import types
from pathlib import Path
from unittest.mock import patch

from sound.voice_commands import VoiceCommandListener, parse_command


class ParserReliabilityTests(unittest.TestCase):
    def test_wake_variant_opens_menu(self):
        for phrase in ('Jarvis menu', 'Jervis open menu', 'hey jervis show the menu'):
            with self.subTest(phrase=phrase):
                self.assertEqual(parse_command(phrase), 'MENU')

    def test_bare_menu_phrases_are_precise(self):
        for phrase in ('menu', 'open menu', 'show the menu'):
            with self.subTest(phrase=phrase):
                self.assertEqual(parse_command(phrase), 'MENU')
        for phrase in ('close menu', 'hide menu'):
            with self.subTest(phrase=phrase):
                self.assertEqual(parse_command(phrase), 'CLOSE')

    def test_no_bare_privileged_or_incidental_commands(self):
        for phrase in ('exit', 'quit', 'enroll face', 'the menu was great',
                       'a restaurant menu', 'close the app', 'notjarvis menu',
                       'please exit menu', '[unk] menu'):
            with self.subTest(phrase=phrase):
                self.assertIsNone(parse_command(phrase))

    def test_unknown_token_blocks_mixed_recognizer_result(self):
        for phrase in ('[unk] menu', 'jarvis [unk] menu',
                       'jarvis [unk] exit', 'jervis menu [unk]'):
            with self.subTest(phrase=phrase):
                self.assertIsNone(parse_command(phrase))

    def test_bare_menu_stays_nonblocking_and_diagnostic(self):
        listener = VoiceCommandListener()
        listener.accept_text('open menu')
        self.assertEqual(listener.last_feedback, 'HEARD MENU')
        self.assertEqual(listener.pop_command(), 'MENU')
        self.assertIsNone(listener.pop_command())
        listener.accept_text('unrecognized sentence')
        self.assertEqual(listener.last_feedback, 'COMMAND NOT RECOGNIZED')
        self.assertIsNone(listener.pop_command())


class ListenerReliabilityTests(unittest.TestCase):
    def test_ready_status_and_constrained_grammar_include_unknown(self):
        ready = threading.Event()
        captured = []

        class Stream:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                ready.set()
                return self
            def __exit__(self, *args):
                pass

        class Recognizer:
            def Reset(self):
                pass
            def SetGrammar(self, grammar):
                raise RuntimeError('native vosk_recognizer_set_grm symbol missing')

        def make_recognizer(model, rate, grammar):
            captured.append(json.loads(grammar))
            return Recognizer()

        with patch.dict(sys.modules, {
            'sounddevice': types.SimpleNamespace(RawInputStream=Stream),
            'vosk': types.SimpleNamespace(Model=lambda path: object(),
                                          KaldiRecognizer=make_recognizer),
        }):
            listener = VoiceCommandListener()
            listener.start()
            try:
                self.assertTrue(ready.wait(3))
                self.assertTrue(listener.available)
                self.assertEqual(listener.last_feedback, 'MIC READY')
                self.assertIsNone(listener.error)
                self.assertEqual(len(captured), 1)
                self.assertIn('jarvis menu', captured[0])
                self.assertIn('jervis menu', captured[0])
                self.assertIn('open menu', captured[0])
                self.assertIn('close menu', captured[0])
                self.assertIn('[unk]', captured[0])
            finally:
                listener.stop()

    @unittest.skipUnless(shutil.which('say') and shutil.which('ffmpeg'),
                         'macOS say and ffmpeg required for generated fixture')
    def test_real_model_recognizes_generated_menu_voice(self):
        from vosk import Model, KaldiRecognizer
        from sound.voice_commands import MODEL_PATH, _GRAMMAR

        with tempfile.TemporaryDirectory() as folder:
            model = Model(str(MODEL_PATH))
            for phrase, expected in (('Jarvis menu', 'MENU'), ('open menu', 'MENU'),
                                     ('close menu', 'CLOSE')):
                with self.subTest(phrase=phrase):
                    audio = Path(folder) / 'voice.aiff'
                    pcm = Path(folder) / 'voice.raw'
                    subprocess.run(['say', '-v', 'Daniel', '-o', str(audio), phrase],
                                   check=True, stdout=subprocess.DEVNULL)
                    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', str(audio),
                                    '-ac', '1', '-ar', '16000', '-f', 's16le', str(pcm)],
                                   check=True, stdout=subprocess.DEVNULL)
                    recognizer = KaldiRecognizer(model, 16000, json.dumps(_GRAMMAR))
                    results = []
                    data = pcm.read_bytes()
                    for offset in range(0, len(data), 8000):
                        if recognizer.AcceptWaveform(data[offset:offset + 8000]):
                            results.append(json.loads(recognizer.Result())['text'])
                    results.append(json.loads(recognizer.FinalResult())['text'])
                    self.assertIn(expected, [parse_command(text) for text in results])

    @unittest.skipUnless(shutil.which('say') and shutil.which('ffmpeg'),
                         'macOS say and ffmpeg required for generated fixture')
    def test_real_model_does_not_promote_bare_dangerous_speech(self):
        from vosk import Model, KaldiRecognizer
        from sound.voice_commands import MODEL_PATH, _GRAMMAR

        model = Model(str(MODEL_PATH))
        with tempfile.TemporaryDirectory() as folder:
            for phrase in ('exit', 'enroll face', 'the weather is sunny'):
                with self.subTest(phrase=phrase):
                    audio = Path(folder) / 'voice.aiff'
                    pcm = Path(folder) / 'voice.raw'
                    subprocess.run(['say', '-v', 'Daniel', '-o', str(audio), phrase],
                                   check=True, stdout=subprocess.DEVNULL)
                    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', str(audio),
                                    '-ac', '1', '-ar', '16000', '-f', 's16le', str(pcm)],
                                   check=True, stdout=subprocess.DEVNULL)
                    recognizer = KaldiRecognizer(model, 16000, json.dumps(_GRAMMAR))
                    results = []
                    data = pcm.read_bytes()
                    for offset in range(0, len(data), 8000):
                        if recognizer.AcceptWaveform(data[offset:offset + 8000]):
                            results.append(json.loads(recognizer.Result())['text'])
                    results.append(json.loads(recognizer.FinalResult())['text'])
                    self.assertTrue(all(parse_command(text) is None for text in results),
                                    'unwoken speech must not trigger a command')


if __name__ == '__main__':
    unittest.main()

"""Guided real-microphone verification is a dry run, not app dispatch."""
import io
import unittest
from unittest.mock import Mock

from scripts.verify_jarvis_voice import PHRASES, run_check, wait_until_ready


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class VoiceVerificationTests(unittest.TestCase):
    def test_phrases_cover_assistant_and_map_without_performing_actions(self):
        self.assertIn(('Jarvis status', 'STATUS'), PHRASES)
        self.assertIn(('Jarvis describe', 'DESCRIBE'), PHRASES)
        self.assertIn(('Jarvis allow location', 'ALLOW LOCATION'), PHRASES)
        self.assertIn(('Jarvis war mode on', 'WAR MODE ON'), PHRASES)
        self.assertIn(('Jarvis war mode off', 'WAR MODE OFF'), PHRASES)

    def test_ready_reports_microphone_failure_or_timeout(self):
        clock = FakeClock()
        listener = Mock(available=False, error='microphone denied')
        self.assertFalse(wait_until_ready(listener, 1, clock.time, clock.sleep))
        listener.error = None
        self.assertFalse(wait_until_ready(listener, 1, clock.time, clock.sleep))
        self.assertGreaterEqual(clock.now, 1)
        listener.available = True
        self.assertTrue(wait_until_ready(listener, 1, clock.time, clock.sleep))

    def test_guided_check_reports_hits_mismatches_and_timeouts_without_dispatch(self):
        class Listener:
            available = True
            error = None
            last_feedback = None
            def __init__(self):
                self.commands = []
            def pop_command(self):
                return self.commands.pop(0) if self.commands else None
        listener = Listener()
        output = io.StringIO()
        clock = FakeClock()
        target = [('Jarvis status', 'STATUS'), ('Jarvis brain', 'BRAIN'),
                  ('Jarvis map', 'SHOW MAP')]
        sent = iter(('STATUS', 'BODY', None))
        def confirm(_prompt):
            command = next(sent)
            if command:
                listener.commands.append(command)
        results = run_check(listener, target, timeout=0.2, confirm=confirm,
                            announce=lambda message: print(message, file=output),
                            clock=clock.time, sleep=clock.sleep)
        self.assertEqual(results, [('Jarvis status', True, 'STATUS'),
                                   ('Jarvis brain', False, 'BODY'),
                                   ('Jarvis map', False, None)])
        self.assertIn('TIMEOUT', output.getvalue())
        self.assertIn('BODY', output.getvalue())

    def test_pending_command_is_drained_before_new_phrase(self):
        class Listener:
            available = True
            error = None
            last_feedback = None
            def __init__(self):
                self.commands = ['STATUS']
            def pop_command(self):
                return self.commands.pop(0) if self.commands else None
        clock = FakeClock()
        listener = Listener()
        results = run_check(listener, [('Jarvis status', 'STATUS')], timeout=0.1,
                            confirm=lambda _: None, announce=lambda _: None,
                            clock=clock.time, sleep=clock.sleep)
        self.assertEqual(results, [('Jarvis status', False, None)])


if __name__ == '__main__':
    unittest.main()

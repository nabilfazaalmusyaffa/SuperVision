"""Guide a real-microphone dry run of Jarvis' offline speech recognizer.

This script does NOT dispatch app actions, perform a location lookup, record audio
files, or claim to verify a particular speaker's identity. Run the app separately
for an end-to-end visual check after these phrases are recognized.
"""
import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sound.voice_commands import VoiceCommandListener

PHRASES = (
    ('Jarvis status', 'STATUS'),
    ('Jarvis what can you do', 'HELP'),
    ('Jarvis brain', 'BRAIN'),
    ('Close', 'CLOSE'),
    ('Jarvis describe', 'DESCRIBE'),
    ('Jarvis war mode on', 'WAR MODE ON'),
    ('Jarvis war mode off', 'WAR MODE OFF'),
    ('Jarvis map', 'SHOW MAP'),
    ('Jarvis locate me', 'LOCATE'),
    ('Jarvis allow location', 'ALLOW LOCATION'),
)


def wait_until_ready(listener, timeout, clock=time.monotonic, sleep=time.sleep):
    """Wait for actual model/microphone acquisition, not just thread startup."""
    deadline = clock() + timeout
    while clock() < deadline:
        if listener.available:
            return True
        if listener.error:
            return False
        sleep(0.05)
    return bool(listener.available)


def _confirm(prompt):
    response = input(prompt)
    if response.strip().lower() in ('q', 'quit'):
        raise KeyboardInterrupt


def run_check(listener, steps=PHRASES, timeout=12.0, *, confirm=_confirm,
              announce=print, clock=time.monotonic, sleep=time.sleep):
    """Return (spoken phrase, pass, recognized command) for each live attempt.

    Only the recognizer's command queue is inspected. Commands are NEVER sent to
    SpiderVisionApp, so even 'allow location' cannot initiate an IP lookup here.
    """
    results = []
    for index, (phrase, expected) in enumerate(steps, start=1):
        # Discard leftover output from the preceding utterance before prompting.
        while listener.pop_command() is not None:
            pass
        listener.last_feedback = None
        confirm(f'[{index}/{len(steps)}] Say "{phrase}" — press Enter to start (q to quit): ')
        deadline = clock() + timeout
        heard = None
        while clock() < deadline:
            heard = listener.pop_command()
            if heard is not None:
                break
            if listener.error or not listener.available:
                break
            sleep(0.05)
        passed = heard == expected
        if passed:
            announce(f'PASS: {phrase} -> {expected}')
        elif heard is not None:
            announce(f'FAIL: {phrase} -> {heard} (expected {expected})')
        elif listener.error or not listener.available:
            announce(f'FAIL: MIC UNAVAILABLE: {listener.error or "device closed"}')
        else:
            feedback = listener.last_feedback or 'no command received'
            announce(f'FAIL: TIMEOUT for {phrase} ({feedback})')
        results.append((phrase, passed, heard))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description='Verify Jarvis speech commands using YOUR microphone.')
    parser.add_argument('--probe', action='store_true', help='check mic/model readiness only; no speech test')
    parser.add_argument('--timeout', type=float, default=12.0, help='seconds allowed per phrase (default: 12)')
    parser.add_argument('--ready-timeout', type=float, default=20.0, help='seconds to initialize microphone')
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.ready_timeout <= 0:
        parser.error('timeouts must be positive')
    print('Close SpiderVision first so this check can use the microphone.')
    print('Microphone processing is local. No audio files, location lookup, or app actions.')
    try:
        from vosk import SetLogLevel
        SetLogLevel(-1)  # Keep the guided result readable, not model-initialization logs.
    except ImportError:
        pass  # The listener will report the missing dependency as MIC UNAVAILABLE.
    listener = VoiceCommandListener()
    listener.start()
    try:
        if not wait_until_ready(listener, args.ready_timeout):
            print(f'MIC UNAVAILABLE: {listener.error or "model/microphone startup timed out"}')
            return 2
        if args.probe:
            print('MIC READY. This does NOT verify your spoken commands; run without --probe and speak.')
            return 0
        print('Speak in English, clearly, after pressing Enter. Pause after each phrase.')
        results = run_check(listener, timeout=args.timeout)
        passed = sum(ok for _, ok, _ in results)
        print(f'Result: {passed}/{len(results)} phrases recognized by microphone + parser.')
        print('This is not a speaker-identity test or free-form AI conversation test.')
        return 0 if passed == len(results) else 1
    except (KeyboardInterrupt, EOFError):
        print('\nVerification stopped; spoken commands not fully verified.')
        return 130
    finally:
        listener.stop()


if __name__ == '__main__':
    raise SystemExit(main())

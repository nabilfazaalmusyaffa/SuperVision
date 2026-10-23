"""Offline Vosk microphone listener with a limited wake-free menu fallback."""
import json

import queue
import re
import threading
from pathlib import Path

from utils.logger import log

MODEL_PATH = Path(__file__).resolve().parent.parent / 'assets/models/vosk-model-small-en-us-0.15'
# Keep [unk] so unrelated speech can be rejected instead of forced into a command.
_GRAMMAR = [
    'menu', 'open menu', 'show menu', 'show the menu',
    'close', 'close menu', 'close the menu', 'hide menu', 'hide the menu',
    *[f'{wake} {command}' for wake in ('jarvis', 'jervis', 'hey jarvis', 'hey jervis')
      for command in ('menu', 'open menu', 'show the menu', 'close', 'close menu',
                      'hide menu', 'audio', 'performance', 'effects',
                      'screenshot', 'exit', 'quit', 'help', 'what can you do',
                      'status', 'describe', 'tell me about this', 'enroll face',
                      'map', 'show map', 'hide map', 'locate me',
                      'allow location', 'clear location',
                      'aim', 'web aim', 'war mode on', 'war mode off',
                      'body', 'brain', 'anatomy',
                      'close anatomy', 'close body', 'close brain', 'close atlas',
                      'zoom in', 'zoom out')],
    '[unk]',
]


def parse_command(text: str) -> str | None:
    """Return a canonical command; only exact menu phrases work without wake."""
    if '[unk]' in text.lower():
        return None
    words = re.findall(r"[a-z]+", text.lower())
    bare = ' '.join(words)
    if bare in ('menu', 'open menu', 'show menu', 'show the menu'):
        return 'MENU'
    if bare in ('close', 'close menu', 'close the menu', 'hide menu', 'hide the menu'):
        return 'CLOSE'
    if not any(word in ('jarvis', 'jervis') for word in words):
        return None
    words = words[next(i for i, word in enumerate(words) if word in ('jarvis', 'jervis')) + 1:]
    phrase = ' '.join(words)
    if phrase in ('close anatomy', 'close body', 'close brain', 'close atlas'):
        return 'CLOSE ANATOMY'
    if phrase == 'zoom in':
        return 'ZOOM IN'
    if phrase == 'zoom out':
        return 'ZOOM OUT'
    if phrase in ('describe', 'tell me about this'):
        return 'DESCRIBE'
    if phrase == 'what can you do':
        return 'HELP'
    if phrase == 'status':
        return 'STATUS'
    if phrase in ('map', 'show map'):
        return 'SHOW MAP'
    if phrase in ('hide map', 'clear location'):
        return 'HIDE MAP'
    if phrase == 'war mode on':
        return 'WAR MODE ON'
    if phrase == 'war mode off':
        return 'WAR MODE OFF'
    if phrase == 'locate me':
        return 'LOCATE'
    if phrase == 'allow location':
        return 'ALLOW LOCATION'
    if phrase == 'brain':
        return 'BRAIN'
    if phrase in ('body', 'anatomy'):
        return 'BODY'

    if 'close' in words or 'hide' in words:
        return 'CLOSE'
    if 'enroll' in words and 'face' in words:
        return 'ENROLL FACE'
    if 'aim' in words:
        return 'WEB AIM'
    for word, command in (
        ('screenshot', 'SCREENSHOT'), ('performance', 'PERFORMANCE'),
        ('effects', 'EFFECTS'), ('audio', 'AUDIO'), ('menu', 'MENU'),
        ('exit', 'EXIT'), ('quit', 'EXIT'), ('help', 'HELP'),
    ):
        if word in words:
            return command
    return None


class VoiceCommandListener:
    """Nonblocking microphone capture, recognition and FIFO command retrieval.

    start() returns immediately. `available` becomes true only after microphone
    acquisition succeeds; `error` describes failures. Authorization is the caller's
    responsibility when dispatching commands (not part of speech recognition).
    """

    def __init__(self, audio=None, model_path=None):
        self.audio = audio
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.available = False
        self.error = None
        self.last_feedback = None
        self._commands = queue.Queue()
        self._samples = queue.Queue(maxsize=32)
        self._stop = threading.Event()
        self._thread = None

    def accept_text(self, text: str):
        """Feed a completed utterance; also useful for deterministic tests."""
        if self.audio is not None and self.audio.is_speaking:
            return
        command = parse_command(text)
        if command:
            self._commands.put(command)
            self.last_feedback = f'HEARD {command}'
        elif text.strip():
            self.last_feedback = 'COMMAND NOT RECOGNIZED'

    def pop_command(self):
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self.error = None
        self.available = False
        self.last_feedback = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._listen, name='SpiderVisionVoice', daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self.available = False
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)

    def _capture(self, data, frames, time_info, status):
        if self._stop.is_set() or (self.audio is not None and self.audio.is_speaking):
            return
        try:
            self._samples.put_nowait(bytes(data))
        except queue.Full:
            pass  # Never block PortAudio's callback.

    def _listen(self):
        try:
            if not self.model_path.is_dir():
                raise FileNotFoundError(f'Vosk model missing: {self.model_path}')
            import sounddevice as sd
            from vosk import Model, KaldiRecognizer
            model = Model(str(self.model_path))
            rate = 16000
            try:
                # Vosk macOS wheels can lack native SetGrammar even when the Python
                # wrapper advertises it. Constructor grammar uses a different symbol.
                recognizer = KaldiRecognizer(model, rate, json.dumps(_GRAMMAR))
            except (TypeError, AttributeError):
                recognizer = KaldiRecognizer(model, rate)
            with sd.RawInputStream(samplerate=rate, blocksize=4000, dtype='int16',
                                   channels=1, callback=self._capture):
                self.available = True
                self.last_feedback = 'MIC READY'
                log.info('Voice microphone ready')
                suppressed = False
                while not self._stop.is_set():
                    speaking = self.audio is not None and self.audio.is_speaking
                    if speaking:
                        suppressed = True
                        recognizer.Reset()
                        self._drain_samples()
                        self._stop.wait(0.05)
                        continue
                    if suppressed:
                        self._drain_samples()
                        recognizer.Reset()
                        suppressed = False
                    try:
                        samples = self._samples.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if self.audio is not None and self.audio.is_speaking:
                        continue
                    if recognizer.AcceptWaveform(samples):
                        text = json.loads(recognizer.Result()).get('text', '')
                        self.accept_text(text)
        except Exception as exc:
            if not self._stop.is_set():
                self.error = str(exc)
                self.last_feedback = 'MIC UNAVAILABLE'
                log.warning(f'Voice microphone unavailable: {exc}')
        finally:
            self.available = False

    def _drain_samples(self):
        while True:
            try:
                self._samples.get_nowait()
            except queue.Empty:
                return

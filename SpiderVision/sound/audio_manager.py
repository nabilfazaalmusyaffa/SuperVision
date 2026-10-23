"""
SpiderVision — Audio Manager
pygame.mixer wrapper for sound effects + edge-tts for TTS.
"""
import os
import asyncio
import threading
import tempfile
import subprocess
import sys
from collections import deque
import pygame
import pygame.mixer
from utils.logger import log
from config.settings import (
    SOUNDS_DIR, AUDIO_FREQ, AUDIO_CHANNELS, AUDIO_BUFFER,
    MASTER_VOLUME, TTS_VOICE, TTS_RATE, TTS_VOLUME
)


class AudioManager:
    """
    Manages sound effects via pygame.mixer.
    TTS via edge-tts (async, runs in dedicated thread).
    """

    def __init__(self):
        pygame.mixer.pre_init(AUDIO_FREQ, -16, AUDIO_CHANNELS, AUDIO_BUFFER)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(16)
        self._sounds: dict[str, pygame.mixer.Sound] = {}
        self._tts_thread: threading.Thread | None = None
        self._tts_lock = threading.Lock()
        self._tts_pending = deque()
        self._speaking = threading.Event()
        self._shutting_down = False
        self.muted = False
        self._load_sounds()
        log.info("AudioManager ready")

    def _load_sounds(self):
        names = ["startup","scanning","lock","warning","menu",
                 "web","retract","explosion","capture","shutdown"]
        for name in names:
            path = os.path.join(SOUNDS_DIR, f"{name}.wav")
            if os.path.exists(path):
                try:
                    snd = pygame.mixer.Sound(path)
                    snd.set_volume(MASTER_VOLUME)
                    self._sounds[name] = snd
                except Exception as e:
                    log.warning(f"Could not load sound {name}: {e}")
            else:
                log.warning(f"Sound missing: {path}")

    # ── SFX ───────────────────────────────────────────────────────────────────

    def play(self, name: str, volume: float | None = None, loops: int = 0):
        if self.muted:
            return
        snd = self._sounds.get(name)
        if snd:
            if volume is not None:
                snd.set_volume(volume * MASTER_VOLUME)
            snd.play(loops=loops)

    def stop(self, name: str):
        snd = self._sounds.get(name)
        if snd:
            snd.stop()

    def stop_all(self):
        pygame.mixer.stop()

    # ── TTS ───────────────────────────────────────────────────────────────────

    def speak(self, text: str):
        """Queue a response without blocking the render thread."""
        if not text or self.muted:
            return
        with self._tts_lock:
            if self._shutting_down:
                return
            self._tts_pending.append(text)
            self._speaking.set()
            if self._tts_thread is None or not self._tts_thread.is_alive():
                self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
                self._tts_thread.start()

    @property
    def is_speaking(self):
        """True throughout playback and while queued responses await playback."""
        return self._speaking.is_set()

    def _tts_worker(self):
        while True:
            with self._tts_lock:
                if not self._tts_pending or self._shutting_down:
                    self._tts_pending.clear()
                    self._speaking.clear()
                    return
                text = self._tts_pending.popleft()
            try:
                if not self.muted:
                    asyncio.run(self._tts_async(text))
            except Exception as e:
                log.warning(f"TTS failed: {e}")

    async def _tts_async(self, text: str):
        tmp_path = None
        try:
            # Prefer the local voice on macOS: speech must work without internet.
            # Run the blocking CLI in a worker so the asyncio loop stays responsive.
            if sys.platform == 'darwin':
                await asyncio.to_thread(self._speak_offline, text)
                return
            import edge_tts
            communicate = edge_tts.Communicate(
                text=text, voice=TTS_VOICE, rate=TTS_RATE, volume=TTS_VOLUME
            )
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            await asyncio.wait_for(communicate.save(tmp_path), timeout=12)

            # Play via pygame (load mp3)
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.set_volume(MASTER_VOLUME)
            pygame.mixer.music.play()

            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)

        except Exception as e:
            log.warning(f"edge-tts error: {e}")
            # Offline fallback: local macOS voice, no network account needed.
            try:
                self._speak_offline(text)
            except Exception as fallback_error:
                log.warning(f"Offline voice failed: {fallback_error}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _speak_offline(self, text):
        if sys.platform == "darwin":
            subprocess.run(['say', '-v', 'Daniel', text], check=True, timeout=25)
        else:
            raise RuntimeError("Offline TTS only configured for macOS")

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def shutdown(self):
        with self._tts_lock:
            self._shutting_down = True
            self._tts_pending.clear()
        if self._tts_thread and self._tts_thread.is_alive():
            self._tts_thread.join(timeout=3)
        self.play("shutdown")
        pygame.time.wait(2000)
        pygame.mixer.quit()
        log.info("AudioManager shutdown")

"""Retraction audio is a real local WAV effect, not just a queued voice line."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile


class RetractAudioTests(unittest.TestCase):
    def test_retract_sound_is_non_silent_local_wav(self):
        from utils.sound_generator import gen_retract
        scratch = '/Users/user/.hermes/cache/scratch'
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            with patch('utils.sound_generator.SOUNDS_DIR', folder):
                gen_retract()
            path = Path(folder) / 'retract.wav'
            self.assertTrue(path.exists())
            rate, signal = wavfile.read(path)
            self.assertEqual(rate, 48000)
            self.assertGreater(len(signal), rate * 0.35)
            self.assertLess(len(signal), rate * 1.5)
            self.assertGreater(np.max(np.abs(signal.astype(np.int32))), 1000)

    def test_audio_manager_loads_and_plays_retract_sound(self):
        from sound.audio_manager import AudioManager
        from utils.sound_generator import gen_retract
        scratch = '/Users/user/.hermes/cache/scratch'
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            with patch('utils.sound_generator.SOUNDS_DIR', folder):
                gen_retract()
            manager = AudioManager.__new__(AudioManager)
            manager.muted = False
            manager._sounds = {}
            with patch('sound.audio_manager.SOUNDS_DIR', folder), \
                 patch('sound.audio_manager.pygame.mixer.Sound') as sound, \
                 patch('sound.audio_manager.log.warning'):
                manager._load_sounds()
            self.assertIn('retract', manager._sounds)
            manager.play('retract')
            sound.return_value.play.assert_called_once_with(loops=0)


if __name__ == '__main__':
    unittest.main()

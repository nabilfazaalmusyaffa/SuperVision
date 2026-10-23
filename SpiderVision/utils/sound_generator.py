"""
SpiderVision — Sound Generator
Procedural WAV synthesis using NumPy + SciPy.
Generates local sound effects at 48kHz stereo without external samples.
"""
import os
import math
import numpy as np
import scipy.io.wavfile as wavfile
from utils.logger import log
from config.settings import SOUNDS_DIR, AUDIO_FREQ

SR   = AUDIO_FREQ
MONO = True   # generate mono, pygame will play stereo


def _save(name: str, signal: np.ndarray):
    path = os.path.join(SOUNDS_DIR, name)
    # Normalize to int16
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak
    out = (np.clip(signal, -1, 1) * 32000).astype(np.int16)
    # Convert mono to stereo
    stereo = np.column_stack([out, out])
    wavfile.write(path, SR, stereo)
    log.info(f"Sound generated: {name}")


def _t(duration: float) -> np.ndarray:
    return np.linspace(0, duration, int(SR * duration), endpoint=False)


def _sine(freq, t, amp=1.0):
    return np.sin(2 * math.pi * freq * t) * amp


def _env_adsr(t, a, d, s_level, r, total):
    env = np.ones_like(t)
    N   = len(t)
    na  = int(a * SR)
    nd  = int(d * SR)
    nr  = int(r * SR)
    ns  = N - na - nd - nr
    if na > 0:
        env[:na]          = np.linspace(0, 1, na)
    if nd > 0:
        env[na:na+nd]     = np.linspace(1, s_level, nd)
    if ns > 0:
        env[na+nd:na+nd+ns] = s_level
    if nr > 0:
        env[-nr:]         = np.linspace(s_level, 0, nr)
    return env


def gen_startup():
    """Rising arpeggio synth — cinematic power-on."""
    t   = _t(2.5)
    sig = np.zeros(len(t))
    notes = [220, 330, 440, 660, 880, 1320]
    for i, freq in enumerate(notes):
        start = int(i * 0.3 * SR)
        end   = min(len(t), start + int(0.6 * SR))
        tt    = np.linspace(0, 0.6, end - start)
        wave  = _sine(freq, tt) * 0.4
        wave += _sine(freq * 2, tt) * 0.15
        env   = _env_adsr(tt, 0.02, 0.1, 0.7, 0.3, 0.6)
        sig[start:end] += wave * env
    # Pad reverb
    decay = np.exp(-np.linspace(0, 8, len(t)))
    sig  *= decay * 0.8 + 0.2
    _save("startup.wav", sig)


def gen_scanning():
    """Digital scanning beep loop."""
    t   = _t(1.2)
    sig = np.zeros(len(t))
    for i in range(6):
        start = int(i * 0.18 * SR)
        end   = min(len(t), start + int(0.08 * SR))
        tt    = np.linspace(0, 0.08, end - start)
        wave  = _sine(1800 + i * 200, tt) * 0.5
        env   = _env_adsr(tt, 0.005, 0.02, 0.6, 0.05, 0.08)
        sig[start:end] += wave * env
    _save("scanning.wav", sig)


def gen_lock():
    """Confirmation / target lock tone."""
    t   = _t(0.5)
    s1  = _sine(880, t) * 0.5
    s2  = _sine(1320, t) * 0.3
    sig = s1 + s2
    env = _env_adsr(t, 0.01, 0.05, 0.7, 0.3, 0.5)
    sig *= env
    _save("lock.wav", sig)


def gen_warning():
    """Warning buzz — alternating pitches."""
    t   = _t(1.0)
    # Alternating 440/660 Hz with 10 Hz modulation
    mod = (np.sin(2 * math.pi * 10 * t) > 0).astype(float)
    sig = (_sine(440, t) * (1 - mod) + _sine(660, t) * mod) * 0.5
    env = _env_adsr(t, 0.02, 0.05, 0.8, 0.2, 1.0)
    sig *= env
    _save("warning.wav", sig)


def gen_menu():
    """Interface click — short crisp UI tone."""
    t   = _t(0.15)
    sig = _sine(1200, t) * 0.4 + _sine(1800, t) * 0.2
    env = _env_adsr(t, 0.002, 0.03, 0.4, 0.1, 0.15)
    sig *= env
    _save("menu.wav", sig)


def gen_web():
    """Web thwip — whoosh + tension snap."""
    t    = _t(0.6)
    # White noise burst (whoosh)
    noise = np.random.normal(0, 1, len(t))
    # Descending frequency sweep (thwip)
    freq  = np.linspace(2000, 400, len(t))
    sweep = np.sin(2 * math.pi * np.cumsum(freq) / SR) * 0.4
    sig   = noise * 0.3 + sweep
    env   = np.exp(-t * 4.0)
    sig  *= env
    _save("web.wav", sig)


def gen_retract():
    """Quick elastic pull: ascending tension sweep with a soft release."""
    t = _t(0.72)
    freq = np.linspace(330, 1350, len(t))
    sweep = np.sin(2 * math.pi * np.cumsum(freq) / SR) * 0.55
    shimmer = np.sin(2 * math.pi * np.cumsum(freq * 1.6) / SR) * 0.12
    noise = np.random.default_rng(734).normal(0, 1, len(t)) * 0.09
    env = np.sin(np.pi * t / 0.72) ** 1.4
    _save("retract.wav", (sweep + shimmer + noise) * env)


def gen_explosion():
    """Impact thud — sub-bass boom + decay."""
    t     = _t(0.8)
    # Sub bass
    sub   = _sine(60, t) * 0.7
    # Noise
    noise = np.random.normal(0, 1, len(t)) * 0.3
    sig   = sub + noise
    env   = np.exp(-t * 3.5)
    sig  *= env
    _save("explosion.wav", sig)


def gen_capture():
    """Camera shutter — mechanical click."""
    t   = _t(0.2)
    # White noise with very fast attack and decay
    noise = np.random.normal(0, 1, len(t))
    env   = np.exp(-t * 30.0)
    click = _sine(4000, t) * 0.3
    sig   = noise * 0.5 * env + click * env
    _save("capture.wav", sig)


def gen_shutdown():
    """Power down — descending synth."""
    t   = _t(2.0)
    freq_arr = np.linspace(880, 55, len(t))
    phase    = np.cumsum(freq_arr) / SR * 2 * math.pi
    sig      = np.sin(phase) * 0.5 + np.sin(phase * 2) * 0.2
    env      = np.linspace(1.0, 0.0, len(t)) ** 2
    sig     *= env
    _save("shutdown.wav", sig)


def generate_all():
    """Generate all sound files if they don't exist yet."""
    files = {
        "startup.wav":   gen_startup,
        "scanning.wav":  gen_scanning,
        "lock.wav":      gen_lock,
        "warning.wav":   gen_warning,
        "menu.wav":      gen_menu,
        "web.wav":       gen_web,
        "retract.wav":   gen_retract,
        "explosion.wav": gen_explosion,
        "capture.wav":   gen_capture,
        "shutdown.wav":  gen_shutdown,
    }
    for fname, fn in files.items():
        path = os.path.join(SOUNDS_DIR, fname)
        if not os.path.exists(path):
            try:
                fn()
            except Exception as e:
                log.error(f"Failed to generate {fname}: {e}")
        else:
            log.debug(f"Sound exists, skipping: {fname}")


if __name__ == "__main__":
    generate_all()

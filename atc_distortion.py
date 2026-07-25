"""ATC (air traffic control) VHF radio distortion for converted audio.

Re-imposes a synthetic VHF-radio / ATC channel on clean voice-conversion
output: band-limit -> in-band noise -> soft-clip distortion.

This is a scipy/numpy port of the torch/torchaudio `atc_radio_effect` used in
the sibling `atc-anonym` (kNN-VC) repo, kept structurally and parametrically
identical so the two pipelines apply the same channel. The noise is band-limited
to the same passband before mixing, so the static sits "inside" the radio band
rather than as broadband hiss.

The effect is deterministic (fixed noise seed) so runs are reproducible.
"""

import numpy as np
from scipy.signal import butter, sosfilt

# --- ATC effect parameters (matched to atc-anonym/knn/effects.py) ---
LOW_HZ = 300.0        # comms passband low edge
HIGH_HZ = 3400.0      # comms passband high edge
NOISE_LEVEL = 0.01    # RMS-ish level of in-band static, relative to full scale
DRIVE = 3.0           # soft-clip gain; higher = more overmodulation grit
TARGET_PEAK = 0.98    # peak the output is renormalized to
NOISE_SEED = 0        # fixed seed -> reproducible noise bed

# Butterworth order per edge. atc-anonym chains single biquads (2nd-order each);
# we use a modest order to approximate that gentle radio roll-off.
FILTER_ORDER = 2


def _highpass_sos(sr, cutoff):
    return butter(FILTER_ORDER, cutoff / (sr / 2.0), btype="high", output="sos")


def _lowpass_sos(sr, cutoff):
    # Clamp below Nyquist to avoid a filter-design error at low sample rates.
    nyquist = sr / 2.0
    cutoff = min(cutoff, nyquist * 0.99)
    return butter(FILTER_ORDER, cutoff / nyquist, btype="low", output="sos")


def _bandlimit(x, sr):
    """High-pass at LOW_HZ then low-pass at HIGH_HZ (the radio band)."""
    x = sosfilt(_highpass_sos(sr, LOW_HZ), x)
    x = sosfilt(_lowpass_sos(sr, HIGH_HZ), x)
    return x


def atc_radio_effect(wav, sample_rate=16000):
    """Re-impose a VHF-radio / ATC channel on a clean float waveform.

    Args:
        wav: 1-D float numpy array, nominally in [-1, 1].
        sample_rate: sample rate of wav (FreeVC outputs 16 kHz).

    Returns:
        A float32 numpy array of the same length, in [-1, 1].
    """
    x = np.asarray(wav, dtype=np.float64)

    # 1. Band-pass to the radio band (kills lows and highs -> "radio" timbre).
    x = _bandlimit(x, sample_rate)

    # 2. Additive static, band-limited to the same channel so it sits "in" the
    #    radio rather than as broadband hiss.
    if NOISE_LEVEL > 0:
        rng = np.random.default_rng(NOISE_SEED)
        noise = rng.standard_normal(x.shape)
        noise = _bandlimit(noise, sample_rate)
        noise = noise / (np.max(np.abs(noise)) + 1e-8) * NOISE_LEVEL
        x = x + noise

    # 3. Soft-clip distortion (tanh) to mimic overmodulation / analog grit.
    x = np.tanh(x * DRIVE)

    # Normalize to avoid harsh full-scale clipping on save.
    x = x / (np.max(np.abs(x)) + 1e-8) * TARGET_PEAK
    return x.astype(np.float32)


# Backwards-compatible alias: earlier code in this repo called
# apply_atc_distortion(wav, sr). Keep it pointing at the matched effect.
def apply_atc_distortion(wav, sr):
    return atc_radio_effect(wav, sample_rate=sr)

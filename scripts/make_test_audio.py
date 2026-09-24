"""Generate synthetic test music (no real tracks needed for MVP verification).

Each 'family' has a distinct musical character (tempo / pitch / energy / timbre),
and clips within a family vary slightly. A correct similarity model should rank
same-family clips nearest to each other -> this gives us a control assertion.

Usage:
    python scripts/make_test_audio.py data/test_music
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100

# name, duration, bpm, base_freq, harmonic_brightness(1..8), rms_target, noise_ratio, vibrato
FAMILIES = {
    "calm_piano":    dict(dur=4.0, bpm=72,  freq=220.0, harm=2, rms=0.06, noise=0.00, vib=1.5),
    "deep_bass":     dict(dur=4.0, bpm=84,  freq=82.0,  harm=3, rms=0.10, noise=0.01, vib=1.0),
    "bright_pop":    dict(dur=4.0, bpm=120, freq=440.0, harm=6, rms=0.20, noise=0.02, vib=3.0),
    "energetic_drum":dict(dur=4.0, bpm=140, freq=180.0, harm=4, rms=0.28, noise=0.06, vib=2.0),
}


def _t(dur):
    return np.linspace(0, dur, int(SR * dur), endpoint=False)


def synth(p: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = _t(p["dur"])
    # slight per-clip detune + tempo jitter within a family
    base = p["freq"] * float(rng.uniform(0.97, 1.03))
    signal = np.zeros_like(t)
    for h in range(1, p["harm"] + 1):
        amp = 1.0 / h
        vib = 1.0 + 0.002 * np.sin(2 * np.pi * p["vib"] * t)
        signal += amp * np.sin(2 * np.pi * base * h * vib * t)
    # percussive envelope pulses at the tempo -> drives onset/bpm features
    beat = 60.0 / p["bpm"]
    env = np.ones_like(t)
    for b in np.arange(0, p["dur"], beat):
        idx = int(b * SR)
        width = int(0.05 * SR)
        env[idx:idx + width] *= np.linspace(1.0, 0.3, min(width, max(0, len(t) - idx)))[:min(width, max(0, len(t) - idx))]
    signal *= env
    if p["noise"] > 0:
        n = rng.normal(0, 1.0, len(t))
        # low-pass-ish noise by moving average to sound less hiss-like
        k = max(1, int(SR / 4000))
        n = np.convolve(n, np.ones(k) / k, mode="same")
        signal = (1 - p["noise"]) * signal + p["noise"] * n * np.abs(signal).mean() * 6
    # normalize to target rms
    cur = np.sqrt(np.mean(signal ** 2)) + 1e-9
    signal = signal / cur * p["rms"]
    return signal.astype("float32")


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/test_music")
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for fam, params in FAMILIES.items():
        for i in range(3):
            audio = synth(params, seed=hash((fam, i)) & 0xFFFF)
            path = out / f"{fam}_{i+1}.wav"
            sf.write(str(path), audio, SR)
            count += 1
    print(f"wrote {count} synthetic tracks to {out}")


if __name__ == "__main__":
    main()

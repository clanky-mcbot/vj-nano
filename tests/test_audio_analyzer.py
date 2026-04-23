"""Tests for the audio analyzer.

Uses synthetic signals so we don't need any real audio files.
"""

import numpy as np

from vj.audio.analyzer import AudioAnalyzer


def _click_track(bpm: float, sr: int = 44100, dur: float = 8.0) -> np.ndarray:
    """Generate a simple click track at a given BPM (short noise bursts)."""
    n = int(sr * dur)
    x = np.zeros(n, dtype=np.float32)
    period = 60.0 / bpm
    rng = np.random.default_rng(0)
    i = 0.0
    while i < dur:
        start = int(i * sr)
        burst = min(int(0.01 * sr), n - start)
        if burst > 0:
            x[start : start + burst] = 0.8 * rng.standard_normal(burst).astype(np.float32)
        i += period
    return x


def _run(analyzer: AudioAnalyzer, audio: np.ndarray):
    feats = []
    for i in range(0, len(audio) - analyzer.hop + 1, analyzer.hop):
        feats.append(analyzer.process(audio[i : i + analyzer.hop]))
    return feats


def test_silence_has_no_onsets():
    sr = 22050
    a = AudioAnalyzer(sr=sr, hop=512)
    feats = _run(a, np.zeros(sr * 2, dtype=np.float32))
    assert not any(f.onset for f in feats)
    assert all(f.rms < 1e-3 for f in feats)


def test_click_track_detects_roughly_correct_bpm():
    for true_bpm in (100.0, 120.0, 140.0):
        sr = 22050
        a = AudioAnalyzer(sr=sr, hop=512)
        audio = _click_track(true_bpm, sr=sr, dur=8.0)
        feats = _run(a, audio)
        final_bpm = feats[-1].bpm
        # Allow 5% error from a 6-second window.
        err = abs(final_bpm - true_bpm) / true_bpm
        assert err < 0.05, f"true={true_bpm}, got={final_bpm}"


def test_click_track_produces_onsets():
    sr = 22050
    a = AudioAnalyzer(sr=sr, hop=512)
    audio = _click_track(120.0, sr=sr, dur=4.0)
    feats = _run(a, audio)
    n_onsets = sum(1 for f in feats if f.onset)
    # 4s at 120bpm -> 8 beats; allow wide tolerance for warmup.
    assert 5 <= n_onsets <= 12, f"got {n_onsets} onsets"


def test_beat_phase_advances_monotonically_between_onsets():
    sr = 22050
    a = AudioAnalyzer(sr=sr, hop=512)
    audio = _click_track(120.0, sr=sr, dur=6.0)
    feats = _run(a, audio)
    # After BPM locks, phase should be in [0,1].
    for f in feats:
        assert 0.0 <= f.beat_phase <= 1.0


def test_band_rms_reacts_to_frequency_content():
    sr = 22050
    a = AudioAnalyzer(sr=sr, hop=1024)
    t = np.arange(sr * 2) / sr
    bass_tone = 0.5 * np.sin(2 * np.pi * 80 * t).astype(np.float32)
    treble_tone = 0.5 * np.sin(2 * np.pi * 5000 * t).astype(np.float32)
    f_bass = _run(a, bass_tone)[-1]
    a2 = AudioAnalyzer(sr=sr, hop=1024)
    f_treb = _run(a2, treble_tone)[-1]
    assert f_bass.bass > f_bass.treble * 3
    assert f_treb.treble > f_treb.bass * 3

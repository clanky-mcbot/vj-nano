"""Real-time audio analysis for VJ rig.

Extracts per-frame features useful for driving visuals:
    - rms       : overall loudness envelope
    - bass/mid/treble : band-limited RMS (energy in 3 frequency ranges)
    - flux      : spectral flux (positive-only diff of magnitude spectra)
    - onset     : boolean pulse when a beat/transient is detected
    - bpm       : rolling BPM estimate from onset autocorrelation
    - beat_phase: 0..1 position within the current beat (locks animations)

The analyzer is designed to be fed fixed-size hops of audio samples,
regardless of source (file, sounddevice, loopback).  See `sources.py`.

No librosa dependency — pure numpy / scipy for speed on the Nano.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import get_window


# ---------------------------------------------------------------------------
# Feature container
# ---------------------------------------------------------------------------

@dataclass
class AudioFeatures:
    """Per-hop snapshot of audio features, updated in place by the analyzer."""

    t: float = 0.0                  # time in seconds since analyzer start
    rms: float = 0.0                # 0..1-ish, overall loudness
    bass: float = 0.0               # RMS in bass band
    mid: float = 0.0                # RMS in mid band
    treble: float = 0.0             # RMS in treble band
    flux: float = 0.0               # spectral flux this hop
    onset: bool = False             # True on the hop an onset was detected
    bpm: float = 0.0                # rolling BPM, 0 until enough data
    beat_phase: float = 0.0         # 0..1, fractional position in current beat

    def as_dict(self) -> dict:
        return {
            "t": self.t,
            "rms": self.rms,
            "bass": self.bass,
            "mid": self.mid,
            "treble": self.treble,
            "flux": self.flux,
            "onset": self.onset,
            "bpm": self.bpm,
            "beat_phase": self.beat_phase,
        }


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

@dataclass
class AudioAnalyzer:
    """Streaming audio feature extractor.

    Usage:
        a = AudioAnalyzer(sr=44100, hop=1024)
        for chunk in source:          # chunk is float32 mono, shape (hop,)
            feat = a.process(chunk)
            print(feat.bpm, feat.onset)
    """

    sr: int = 44100                 # sample rate
    hop: int = 1024                 # samples per hop (analysis frame)
    fft_size: int = 2048            # zero-padded FFT length (>= hop)

    # Band edges in Hz. Tweaked for electronic music: kick/bass, body, hats.
    bass_hz: tuple = (30.0, 250.0)
    mid_hz: tuple = (250.0, 2000.0)
    treble_hz: tuple = (2000.0, 10000.0)

    # Onset detector: threshold = median(flux history) * thresh_mul + eps.
    onset_history_sec: float = 1.5
    onset_thresh_mul: float = 1.6
    onset_min_interval_sec: float = 0.12  # debounce (~500 BPM ceiling)

    # BPM tracker: autocorr on onset envelope over this window.
    bpm_window_sec: float = 6.0
    bpm_min: float = 70.0
    bpm_max: float = 180.0

    # --- internal state ---
    _window: np.ndarray = field(init=False, repr=False)
    _prev_mag: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _flux_history: list = field(default_factory=list, init=False, repr=False)
    _onset_env: list = field(default_factory=list, init=False, repr=False)
    _last_onset_t: float = field(default=-1e9, init=False, repr=False)
    _hops_processed: int = field(default=0, init=False, repr=False)
    _bpm: float = field(default=0.0, init=False, repr=False)
    _last_beat_t: float = field(default=0.0, init=False, repr=False)
    # Band indices (filled in __post_init__)
    _bass_bins: tuple = field(default=(0, 0), init=False, repr=False)
    _mid_bins: tuple = field(default=(0, 0), init=False, repr=False)
    _treble_bins: tuple = field(default=(0, 0), init=False, repr=False)

    def __post_init__(self) -> None:
        if self.fft_size < self.hop:
            self.fft_size = self.hop
        self._window = get_window("hann", self.hop, fftbins=True).astype(np.float32)
        bin_hz = self.sr / self.fft_size
        self._bass_bins = self._hz_to_bins(self.bass_hz, bin_hz)
        self._mid_bins = self._hz_to_bins(self.mid_hz, bin_hz)
        self._treble_bins = self._hz_to_bins(self.treble_hz, bin_hz)

    @staticmethod
    def _hz_to_bins(rng: tuple, bin_hz: float) -> tuple:
        lo = max(1, int(round(rng[0] / bin_hz)))
        hi = max(lo + 1, int(round(rng[1] / bin_hz)))
        return (lo, hi)

    # ------------------------------------------------------------------
    def process(self, chunk: np.ndarray) -> AudioFeatures:
        """Process one hop of mono float32 audio, return current features."""
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)
        if chunk.shape[0] != self.hop:
            # Pad / truncate so the window math stays stable.
            c = np.zeros(self.hop, dtype=np.float32)
            n = min(self.hop, chunk.shape[0])
            c[:n] = chunk[:n]
            chunk = c
        chunk = chunk.astype(np.float32, copy=False)

        # --- windowed FFT magnitude ---
        windowed = chunk * self._window
        if self.fft_size > self.hop:
            padded = np.zeros(self.fft_size, dtype=np.float32)
            padded[: self.hop] = windowed
            windowed = padded
        spec = np.fft.rfft(windowed)
        mag = np.abs(spec).astype(np.float32)

        # --- features ---
        t = self._hops_processed * self.hop / self.sr
        rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
        bass = self._band_rms(mag, self._bass_bins)
        mid = self._band_rms(mag, self._mid_bins)
        treble = self._band_rms(mag, self._treble_bins)

        # spectral flux = sum of positive magnitude diffs (onset-novelty)
        if self._prev_mag is None:
            flux = 0.0
        else:
            diff = mag - self._prev_mag
            flux = float(np.sum(np.maximum(diff, 0.0)))
        self._prev_mag = mag

        # --- onset detection with adaptive median threshold ---
        self._flux_history.append(flux)
        hist_max = int(self.onset_history_sec * self.sr / self.hop)
        if len(self._flux_history) > hist_max:
            self._flux_history.pop(0)
        med = float(np.median(self._flux_history)) if self._flux_history else 0.0
        thresh = med * self.onset_thresh_mul + 1e-6
        onset = bool(
            flux > thresh
            and (t - self._last_onset_t) >= self.onset_min_interval_sec
        )
        if onset:
            self._last_onset_t = t
            self._last_beat_t = t

        # --- onset envelope for BPM autocorr ---
        # Store a smoothed value so autocorr is less noisy than raw flux.
        env_val = max(0.0, flux - med)
        self._onset_env.append(env_val)
        env_max = int(self.bpm_window_sec * self.sr / self.hop)
        if len(self._onset_env) > env_max:
            self._onset_env.pop(0)

        # Update BPM every ~0.5s to save CPU.
        hops_per_half_sec = max(1, int(0.5 * self.sr / self.hop))
        if (
            self._hops_processed % hops_per_half_sec == 0
            and len(self._onset_env) > env_max // 2
        ):
            self._bpm = self._estimate_bpm(np.asarray(self._onset_env, dtype=np.float32))

        # --- beat phase ---
        if self._bpm > 0.0:
            period = 60.0 / self._bpm
            phase = ((t - self._last_beat_t) % period) / period
        else:
            phase = 0.0

        self._hops_processed += 1

        return AudioFeatures(
            t=t,
            rms=rms,
            bass=bass,
            mid=mid,
            treble=treble,
            flux=flux,
            onset=onset,
            bpm=self._bpm,
            beat_phase=phase,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _band_rms(mag: np.ndarray, bins: tuple) -> float:
        lo, hi = bins
        if hi <= lo or lo >= mag.shape[0]:
            return 0.0
        band = mag[lo : min(hi, mag.shape[0])]
        return float(np.sqrt(np.mean(band * band) + 1e-12))

    def _estimate_bpm(self, env: np.ndarray) -> float:
        """Autocorrelation-based BPM estimate on the onset envelope.

        Returns 0.0 if no confident peak found.
        """
        if env.size < 8:
            return 0.0
        env = env - env.mean()
        if np.allclose(env, 0.0):
            return 0.0

        # Autocorrelation via FFT (faster than np.correlate for our sizes).
        n = 1 << (env.size * 2 - 1).bit_length()
        f = np.fft.rfft(env, n=n)
        ac = np.fft.irfft(f * np.conj(f), n=n)[: env.size]
        ac[0] = 0.0  # kill DC / zero-lag peak

        hops_per_sec = self.sr / self.hop
        min_lag = int(hops_per_sec * 60.0 / self.bpm_max)
        max_lag = int(hops_per_sec * 60.0 / self.bpm_min)
        max_lag = min(max_lag, ac.size - 1)
        if max_lag <= min_lag:
            return 0.0

        region = ac[min_lag : max_lag + 1]
        if region.size == 0 or region.max() <= 0:
            return 0.0
        lag = min_lag + int(np.argmax(region))
        bpm = 60.0 * hops_per_sec / lag

        # Nudge obvious half/double-time mistakes back into a DJ range (90-140).
        while bpm < 85.0:
            bpm *= 2.0
        while bpm > 175.0:
            bpm *= 0.5
        return float(bpm)


# ---------------------------------------------------------------------------
# Smoke test / CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    """Run the analyzer over a .wav/.flac/.ogg file and print features.

    Usage:
        python -m vj.audio.analyzer path/to/song.wav [--plot]
    """
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Analyze an audio file.")
    ap.add_argument("path", help="Audio file to analyze")
    ap.add_argument("--hop", type=int, default=1024)
    ap.add_argument("--plot", action="store_true", help="Save a feature-plot PNG")
    args = ap.parse_args()

    try:
        import soundfile as sf
    except ImportError:
        print("error: soundfile not installed; pip install soundfile", file=sys.stderr)
        sys.exit(1)

    data, sr = sf.read(args.path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    print(f"loaded: {args.path}  sr={sr}  dur={len(data)/sr:.1f}s  samples={len(data)}")

    analyzer = AudioAnalyzer(sr=sr, hop=args.hop)
    feats = []
    for i in range(0, len(data) - args.hop + 1, args.hop):
        feats.append(analyzer.process(data[i : i + args.hop]))

    # Print periodic summary
    step = max(1, len(feats) // 20)
    print(f"{'t(s)':>6} {'rms':>6} {'bass':>6} {'mid':>6} {'treb':>6} "
          f"{'bpm':>6} {'phase':>6} onset")
    onsets = 0
    for i, f in enumerate(feats):
        if f.onset:
            onsets += 1
        if i % step == 0:
            print(f"{f.t:6.2f} {f.rms:6.3f} {f.bass:6.3f} {f.mid:6.3f} "
                  f"{f.treble:6.3f} {f.bpm:6.1f} {f.beat_phase:6.2f} "
                  f"{'X' if f.onset else '.'}")
    bpm_final = feats[-1].bpm if feats else 0.0
    print(f"\ndetected onsets: {onsets}  final BPM: {bpm_final:.1f}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping plot", file=sys.stderr)
            return
        t = np.array([f.t for f in feats])
        rms = np.array([f.rms for f in feats])
        bass = np.array([f.bass for f in feats])
        mid = np.array([f.mid for f in feats])
        treb = np.array([f.treble for f in feats])
        onset_t = np.array([f.t for f in feats if f.onset])
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].plot(t, rms, label="rms", color="k", lw=0.8)
        axes[0].plot(t, bass, label="bass", color="#003f5c", lw=0.8)
        axes[0].plot(t, mid, label="mid", color="#0071A9", lw=0.8)
        axes[0].plot(t, treb, label="treble", color="#7fb8d5", lw=0.8)
        axes[0].vlines(onset_t, 0, rms.max(), color="red", alpha=0.25, lw=0.5, label="onset")
        axes[0].legend(loc="upper right")
        axes[0].set_ylabel("energy")
        axes[1].plot(t, [f.bpm for f in feats], color="k")
        axes[1].set_ylabel("bpm")
        axes[1].set_xlabel("t (s)")
        axes[1].set_ylim(60, 200)
        out = args.path.rsplit(".", 1)[0] + "_features.png"
        plt.tight_layout()
        plt.savefig(out, dpi=110)
        print(f"saved plot: {out}")


if __name__ == "__main__":
    _cli()

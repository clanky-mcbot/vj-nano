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

# NOTE: We deliberately do NOT use `from __future__ import annotations`
# or PEP 604 | union syntax — the Jetson Nano runs Python 3.6.9 which
# predates both. Stick to Python 3.6-compatible typing (typing.Optional, etc.).

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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

    # Onset detector: threshold = median + mul * MAD  (MAD = median abs dev)
    onset_history_sec: float = 1.5
    onset_thresh_mul: float = 2.5   # multiplier for MAD above median
    onset_min_interval_sec: float = 0.12  # debounce (~500 BPM ceiling)

    # BPM tracker
    bpm_window_sec: float = 2.5     # seconds of onset history for tempo estimation
    bpm_min: float = 70.0
    bpm_max: float = 180.0
    tempo_smooth_alpha: float = 0.45  # EMA factor for tempo updates (higher = faster tracking)
    min_confidence_for_lock: float = 0.45
    beat_lookahead_sec: float = 0.04  # accept onsets within +/- 40ms of predicted beat

    # --- internal state ---
    _window: np.ndarray = field(init=False, repr=False)
    _prev_mag: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _flux_history: List[float] = field(default_factory=list, init=False, repr=False)
    _odf_history: List[float] = field(default_factory=list, init=False, repr=False)
    _onset_env: List[float] = field(default_factory=list, init=False, repr=False)
    _last_onset_t: float = field(default=-1e9, init=False, repr=False)
    _hops_processed: int = field(default=0, init=False, repr=False)
    _bpm: float = field(default=0.0, init=False, repr=False)
    _bpm_raw: float = field(default=0.0, init=False, repr=False)
    _tempo_confidence: float = field(default=0.0, init=False, repr=False)
    _last_beat_t: float = field(default=0.0, init=False, repr=False)
    _next_predicted_beat: float = field(default=0.0, init=False, repr=False)
    _beat_predictions: List[Tuple[float, bool]] = field(default_factory=list, init=False, repr=False)
    # Band indices (filled in __post_init__)
    _bass_bins: Tuple[int, int] = field(default=(0, 0), init=False, repr=False)
    _mid_bins: Tuple[int, int] = field(default=(0, 0), init=False, repr=False)
    _treble_bins: Tuple[int, int] = field(default=(0, 0), init=False, repr=False)

    def __post_init__(self):
        # type: () -> None
        if self.fft_size < self.hop:
            self.fft_size = self.hop
        self._window = get_window("hann", self.hop, fftbins=True).astype(np.float32)
        bin_hz = self.sr / self.fft_size
        self._bass_bins = self._hz_to_bins(self.bass_hz, bin_hz)
        self._mid_bins = self._hz_to_bins(self.mid_hz, bin_hz)
        self._treble_bins = self._hz_to_bins(self.treble_hz, bin_hz)

    @staticmethod
    def _hz_to_bins(rng, bin_hz):
        # type: (Tuple[float, float], float) -> Tuple[int, int]
        lo = max(1, int(round(rng[0] / bin_hz)))
        hi = max(lo + 1, int(round(rng[1] / bin_hz)))
        return (lo, hi)

    # ------------------------------------------------------------------
    def process(self, chunk):
        # type: (np.ndarray) -> AudioFeatures
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
        prev_mag = self._prev_mag
        if prev_mag is None:
            flux = 0.0
        else:
            diff = mag - prev_mag
            flux = float(np.sum(np.maximum(diff, 0.0)))
        self._prev_mag = mag

        # --- bass/mid specific flux for better beat detection ---
        bass_flux = 0.0
        mid_flux = 0.0
        if prev_mag is not None:
            b_lo, b_hi = self._bass_bins
            m_lo, m_hi = self._mid_bins
            bass_diff = mag[b_lo:b_hi] - prev_mag[b_lo:b_hi]
            mid_diff = mag[m_lo:m_hi] - prev_mag[m_lo:m_hi]
            bass_flux = float(np.sum(np.maximum(bass_diff, 0.0)))
            mid_flux = float(np.sum(np.maximum(mid_diff, 0.0)))

        # Combined ODF: bass dominates, mid contributes (snares)
        odf = max(bass_flux, 0.5 * mid_flux)
        # Log compression: emphasize quieter onsets, cap loud ones
        odf = float(np.log1p(odf))

        # --- onset detection with adaptive MAD threshold ---
        self._odf_history.append(odf)
        hist_max = int(self.onset_history_sec * self.sr / self.hop)
        if len(self._odf_history) > hist_max:
            self._odf_history.pop(0)

        med = float(np.median(self._odf_history)) if self._odf_history else 0.0
        mad = float(np.median(np.abs(np.array(self._odf_history) - med))) if self._odf_history else 0.0
        thresh = med + self.onset_thresh_mul * mad + 1e-6

        onset = bool(
            odf > thresh
            and (t - self._last_onset_t) >= self.onset_min_interval_sec
        )
        if onset:
            self._last_onset_t = t

        # --- onset envelope for BPM autocorr ---
        env_val = max(0.0, odf - med)
        self._onset_env.append(env_val)
        env_max = int(self.bpm_window_sec * self.sr / self.hop)
        if len(self._onset_env) > env_max:
            self._onset_env.pop(0)

        # Update BPM every ~0.5s to save CPU.
        hops_per_quarter_sec = max(1, int(0.25 * self.sr / self.hop))
        if (
            self._hops_processed % hops_per_quarter_sec == 0
            and len(self._onset_env) > env_max // 2
        ):
            raw_bpm = self._estimate_bpm(np.asarray(self._onset_env, dtype=np.float32))
            self._update_tempo(raw_bpm, t)

        # --- beat phase with prediction & locking ---
        beat_phase = self._update_beat_phase(t, onset)

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
            beat_phase=beat_phase,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _band_rms(mag, bins):
        # type: (np.ndarray, Tuple[int, int]) -> float
        lo, hi = bins
        if hi <= lo or lo >= mag.shape[0]:
            return 0.0
        band = mag[lo : min(hi, mag.shape[0])]
        return float(np.sqrt(np.mean(band * band) + 1e-12))

    def _estimate_bpm(self, env):
        # type: (np.ndarray) -> float
        """Autocorrelation-based BPM estimate on the onset envelope.

        Uses parabolic interpolation for sub-sample lag accuracy.
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

        peak_idx = int(np.argmax(region))
        lag = min_lag + peak_idx

        # Parabolic interpolation around peak for sub-sample accuracy.
        lag_f = self._parabolic_interp(ac, lag)
        lag_f = max(float(min_lag), min(float(max_lag), lag_f))

        bpm = 60.0 * hops_per_sec / lag_f

        # Nudge obvious half/double-time mistakes back into a DJ range (90-140).
        while bpm < 85.0:
            bpm *= 2.0
        while bpm > 175.0:
            bpm *= 0.5
        return float(bpm)

    @staticmethod
    def _parabolic_interp(arr, idx):
        # type: (np.ndarray, int) -> float
        """Parabolic interpolation of peak at idx."""
        if idx <= 0 or idx >= len(arr) - 1:
            return float(idx)
        a = arr[idx - 1]
        b = arr[idx]
        c = arr[idx + 1]
        denom = a - 2.0 * b + c
        if abs(denom) < 1e-12:
            return float(idx)
        p = 0.5 * (a - c) / denom
        return idx + p

    def _update_tempo(self, new_bpm, t):
        # type: (float, float) -> None
        """Smooth tempo updates with adaptive confidence tracking."""
        if new_bpm <= 0.0:
            return

        self._bpm_raw = new_bpm

        if self._bpm <= 0.0:
            self._bpm = new_bpm
            self._tempo_confidence = 0.25
            self._last_beat_t = t
            self._next_predicted_beat = t
            self._tempo_divergence_count = 0
            return

        ratio = max(new_bpm, self._bpm) / max(min(new_bpm, self._bpm), 1e-6)
        consistent = 0.82 < ratio < 1.22  # within ~20%

        if consistent:
            self._tempo_confidence = min(1.0, self._tempo_confidence + 0.10)
            self._tempo_divergence_count = 0
            # Adaptive alpha: faster tracking when confidence is low, very smooth when locked
            alpha = self.tempo_smooth_alpha + 0.6 * self._tempo_confidence * (1.0 - self.tempo_smooth_alpha)
            alpha = min(0.85, alpha)
            self._bpm = alpha * new_bpm + (1.0 - alpha) * self._bpm
        else:
            self._tempo_divergence_count = getattr(self, '_tempo_divergence_count', 0) + 1
            self._tempo_confidence = max(0.0, self._tempo_confidence - 0.15)
            # If divergent for 3+ consecutive estimates, jump to new tempo (fast re-lock)
            if self._tempo_divergence_count >= 2 or self._tempo_confidence < 0.20:
                self._bpm = new_bpm
                self._last_beat_t = t
                self._next_predicted_beat = t
                self._tempo_divergence_count = 0
                self._tempo_confidence = 0.30

    def _update_beat_phase(self, t, onset):
        # type: (float, bool) -> float
        """Predict-and-correct beat phase tracking.

        Once tempo is locked, we predict beat times. Onsets near predictions
        confirm the beat; missed beats don't drift the phase.
        """
        if self._bpm <= 0.0 or self._tempo_confidence < self.min_confidence_for_lock:
            # Not locked: reset phase on each onset, free-run otherwise
            if onset:
                self._last_beat_t = t
                self._beat_predictions = []
                return 0.0
            if self._last_beat_t > 0.0:
                # Rough phase using last raw onset
                period = 60.0 / max(self._bpm, 120.0)
                return ((t - self._last_beat_t) % period) / period
            return 0.0

        period = 60.0 / self._bpm

        # Initialize prediction anchor if needed
        if self._next_predicted_beat <= 0.0:
            self._next_predicted_beat = self._last_beat_t + period

        # Advance predictions that we've passed
        lookahead = self.beat_lookahead_sec
        while self._next_predicted_beat < t + period * 0.5:
            predicted = self._next_predicted_beat
            dist = abs(t - predicted)
            confirmed = dist < lookahead

            self._beat_predictions.append((predicted, confirmed))

            # Trim old predictions
            cutoff = t - 4.0
            self._beat_predictions = [(bt, c) for bt, c in self._beat_predictions if bt > cutoff]

            # Update confidence from recent confirmation rate
            if len(self._beat_predictions) > 2:
                n_conf = sum(1 for _, c in self._beat_predictions if c)
                measured_conf = float(n_conf) / len(self._beat_predictions)
                self._tempo_confidence = 0.7 * self._tempo_confidence + 0.3 * measured_conf

            # If this prediction was confirmed by a nearby onset, snap anchor
            if confirmed and onset:
                self._last_beat_t = predicted

            self._next_predicted_beat += period

        phase = ((t - self._last_beat_t) % period) / period
        return float(phase)


# ---------------------------------------------------------------------------
# Smoke test / CLI entry point
# ---------------------------------------------------------------------------

def _cli():
    # type: () -> None
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
    print("loaded: {}  sr={}  dur={:.1f}s  samples={}".format(
        args.path, sr, len(data)/sr, len(data)))

    analyzer = AudioAnalyzer(sr=sr, hop=args.hop)
    feats = []
    for i in range(0, len(data) - args.hop + 1, args.hop):
        feats.append(analyzer.process(data[i : i + args.hop]))

    # Print periodic summary
    step = max(1, len(feats) // 20)
    print("{:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>8} {:>6}".format(
        "t(s)", "rms", "bass", "mid", "treb", "bpm", "conf", "phase", "onset"))
    onsets = 0
    for i, f in enumerate(feats):
        if f.onset:
            onsets += 1
        if i % step == 0:
            print("{:6.2f} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.1f} {:6.2f} {:8.3f} {:>6}".format(
                f.t, f.rms, f.bass, f.mid, f.treble, f.bpm,
                analyzer._tempo_confidence, f.beat_phase, "X" if f.onset else "."))
    bpm_final = feats[-1].bpm if feats else 0.0
    print("\ndetected onsets: {}  final BPM: {:.1f}  confidence: {:.2f}".format(
        onsets, bpm_final, analyzer._tempo_confidence))

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
        axes[1].plot(t, [f.bpm for f in feats], color="k", label="bpm")
        axes[1].set_ylabel("bpm")
        axes[1].set_xlabel("t (s)")
        axes[1].set_ylim(60, 200)
        out = args.path.rsplit(".", 1)[0] + "_features.png"
        plt.tight_layout()
        plt.savefig(out, dpi=110)
        print("saved plot: {}".format(out))


if __name__ == "__main__":
    _cli()

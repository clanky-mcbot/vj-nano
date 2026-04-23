"""Audio input sources.

Two sources share the same interface: an iterable yielding float32 mono
chunks of shape (hop,).

    FileSource(path, sr, hop)        — reads a .wav/.flac/.ogg, dev mode
    LineInSource(sr, hop, device)    — live sounddevice input, gig mode
"""

# Python 3.6 compatible (Jetson Nano JetPack 4.6.4) — no `from __future__
# import annotations`, no PEP 604 unions.

from typing import Iterator, Optional

import numpy as np


class FileSource:
    """Iterate hops from an audio file, resampling if needed.

    Uses soundfile for decoding and scipy.signal.resample_poly for rate
    conversion. Blocks in real time if `realtime=True` (useful for dev).
    """

    def __init__(
        self,
        path: str,
        sr: int = 44100,
        hop: int = 1024,
        mono: bool = True,
        realtime: bool = False,
        loop: bool = False,
    ):
        self.path = path
        self.sr = sr
        self.hop = hop
        self.mono = mono
        self.realtime = realtime
        self.loop = loop

    def __iter__(self) -> Iterator[np.ndarray]:
        import soundfile as sf
        import time

        while True:
            data, file_sr = sf.read(self.path, dtype="float32", always_2d=False)
            if data.ndim > 1 and self.mono:
                data = data.mean(axis=1).astype(np.float32)
            if file_sr != self.sr:
                from scipy.signal import resample_poly
                # Use rational resampling. gcd trick keeps numerator small.
                from math import gcd
                g = gcd(self.sr, file_sr)
                data = resample_poly(data, self.sr // g, file_sr // g).astype(np.float32)

            start_wall = time.monotonic()
            for i in range(0, len(data) - self.hop + 1, self.hop):
                chunk = data[i : i + self.hop]
                if self.realtime:
                    target = start_wall + (i + self.hop) / self.sr
                    now = time.monotonic()
                    if target > now:
                        time.sleep(target - now)
                yield chunk
            if not self.loop:
                return


class LineInSource:
    """Iterate hops from the default (or selected) sounddevice input.

    Requires `sounddevice` (and PortAudio system package).
    """

    def __init__(
        self,
        sr: int = 44100,
        hop: int = 1024,
        device: Optional[int] = None,
        channels: int = 1,
        latency: Optional[float] = None,
    ):
        self.sr = sr
        self.hop = hop
        self.device = device
        self.channels = channels
        self.latency = latency

    def __iter__(self) -> Iterator[np.ndarray]:
        import queue
        import sounddevice as sd

        q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=32)

        def cb(indata, frames, time_info, status):
            if status:
                # Drop-prints should happen rarely; stderr to avoid visual spam.
                import sys
                print(f"[sounddevice] {status}", file=sys.stderr)
            mono = indata.mean(axis=1) if indata.ndim > 1 else indata[:, 0]
            q.put(mono.astype(np.float32).copy())

        with sd.InputStream(
            samplerate=self.sr,
            blocksize=self.hop,
            channels=self.channels,
            dtype="float32",
            device=self.device,
            callback=cb,
        ):
            while True:
                yield q.get()

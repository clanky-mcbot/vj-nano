# vj-nano

**Real-time music-reactive VJ rig for Jetson Nano 4GB** — renders a PS1-style 3D character that dances to the beat, with its palette continuously re-tinted from a live webcam pointed at the crowd.

Intended use: visual illustrations during DJ sets. HDMI out to projector/screen.

## Target

- **Hardware:** Jetson Nano 4GB devkit (Maxwell, 128 CUDA cores)
- **Software:** JetPack 4.6.4 (Ubuntu 18.04, Python 3.6, CUDA 10.2, TensorRT 8.2)
- **Target framerate:** 15–20 fps sustained
- **All on one box** — inference + rendering on the Nano, HDMI out directly

## Architecture

```
┌─ sounddevice (file in v1, line-in later) ────────────────┐
│   ├─ FFT → bass / mid / treble RMS                       │
│   ├─ spectral flux → onset pulses                        │
│   └─ autocorrelation → rolling BPM                       │
│                                                          │
├─ gstreamer (webcam 640×480) ─────────────────────────────┤
│   ├─ downscaled 64×48 → k-means → 5-color palette        │
│   └─ full frame → background layer texture               │
│                                                          │
├─ animation state machine ────────────────────────────────┤
│   ├─ library: idle / sway / dance_A / dance_B / ...      │
│   ├─ clip choice ← energy envelope                       │
│   └─ playback time ← musical beat phase (bar-locked)     │
│                                                          │
└─ Panda3D renderer ───────────────────────────────────────┘
    ├─ background quad (webcam texture)
    ├─ character (PS1-style rigged GLB, center)
    │   ├─ vertex-snap + affine-UV shader
    │   └─ palette-remap frag shader (tinted by webcam)
    └─ HDMI out
```

## Status

**🚧 Early development.** See [todo list below](#roadmap).

## Quick start

### On a dev laptop (audio analyzer only, no Nano bits)

```bash
git clone https://github.com/clanky-mcbot/vj-nano.git
cd vj-nano
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m vj.audio.analyzer assets/audio/test.wav  # prints live band energies + BPM
```

### On the Jetson Nano

```bash
git clone https://github.com/clanky-mcbot/vj-nano.git
cd vj-nano
./setup_jetson.sh        # installs sys deps, creates venv, pulls PyTorch wheel
source .venv/bin/activate
python -m vj             # launches the full pipeline (HDMI out)
```

## Roadmap

- [x] Repo scaffold + audio analyzer (file source, dev-testable anywhere)
- [ ] Jetson bootstrap script
- [ ] Webcam capture + palette extraction
- [ ] Panda3D renderer skeleton
- [ ] PS1 shader pack (vertex snap, affine UV, dither, palette remap)
- [ ] Character model + Mixamo dance retargeting
- [ ] Beat-phase-locked animation state machine
- [ ] End-to-end integration
- [ ] Line-in audio source (for the actual gig)

## License

MIT

"""Stream audio from your PC to vj-nano over the network.

Usage:
    1. Start vj-nano on the Jetson with --audio net:
         python3 -m vj.main --audio net --win-size 640x480 --debug

    2. On your Windows PC (or WSL), run this script:
         python audio_sender.py --host 192.168.1.7 --port 5000

    3. Play music on your PC. The audio is captured and streamed live.

Capturing SYSTEM AUDIO on Windows:
    - Best: Install VB-Cable (free virtual audio cable) and set your
      music player output to "CABLE Input", then select "CABLE Output"
      as the recording device below.
    - Alternative: If your sound card has "Stereo Mix", select that.
    - Fallback: Just select your microphone — place it near speakers.

Requirements: Python 3, numpy, sounddevice
    pip install numpy sounddevice
"""

from __future__ import print_function

import argparse
import socket
import struct
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Stream PC audio to vj-nano")
    ap.add_argument("--host", default="192.168.1.7", help="Jetson Nano IP")
    ap.add_argument("--port", type=int, default=5000, help="TCP port")
    ap.add_argument("--device", type=int, default=None, help="sounddevice input index")
    ap.add_argument("--list-devices", action="store_true", help="Show audio devices and exit")
    ap.add_argument("--sr", type=int, default=44100, help="Sample rate")
    ap.add_argument("--hop", type=int, default=512, help="Frames per chunk")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print("Audio capture devices (choose one with input channels):")
        for i, d in enumerate(sd.query_devices()):
            print("  {}: {}  (in={}, out={})".format(
                i, d["name"], d.get("max_input_channels", 0), d.get("max_output_channels", 0)))
        sys.exit(0)

    # Connect to Nano
    print("[sender] connecting to {}:{} ...".format(args.host, args.port))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            sock.connect((args.host, args.port))
            break
        except socket.error:
            print("[sender] retrying in 2s ...")
            time.sleep(2)
    print("[sender] connected!")

    import sounddevice as sd
    import queue

    q = queue.Queue(maxsize=64)

    def cb(indata, frames, t, status):
        if status:
            print("[sender]", status, file=sys.stderr)
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata[:, 0]
        try:
            q.put_nowait(mono.astype(np.float32).copy())
        except queue.Full:
            pass

    stream = sd.InputStream(
        samplerate=args.sr,
        blocksize=args.hop,
        channels=1,
        dtype="float32",
        device=args.device,
        callback=cb,
    )

    print("[sender] streaming ... press Ctrl+C to stop")
    try:
        with stream:
            while True:
                chunk = q.get()
                data = chunk.tobytes()
                header = struct.pack("<I", len(data))
                sock.sendall(header + data)
    except KeyboardInterrupt:
        print("\n[sender] stopping")
    finally:
        sock.close()
        print("[sender] disconnected")


if __name__ == "__main__":
    main()

"""Network audio source — receives float32 audio over TCP.

Usage on the Nano:
    source = NetworkAudioSource(port=5000, sr=44100, hop=512)
    for chunk in source:
        analyzer.process(chunk)

Usage on the PC (see scripts/audio_sender.py):
    python audio_sender.py --host 192.168.1.7 --port 5000
"""

from __future__ import print_function

import queue
import socket
import struct
import sys
import threading
from typing import Iterator, Optional

import numpy as np


class NetworkAudioSource(object):
    """TCP server that yields audio chunks streamed from a remote client.

    Mimics the LineInSource interface (iterable of float32 mono chunks).
    Runs a listener thread; blocks on __iter__ until a client connects.
    """

    def __init__(
        self,
        port=5000,
        sr=44100,
        hop=512,
        channels=1,
        bind_host="0.0.0.0",
    ):
        self.port = port
        self.sr = sr
        self.hop = hop
        self.channels = channels
        self.bind_host = bind_host

        self._q = queue.Queue(maxsize=128)
        self._running = True
        self._listener = threading.Thread(target=self._listen, daemon=True)
        self._listener.start()

    def _listen(self):
        """Accept connections, decode chunks, push to queue."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.bind_host, self.port))
        sock.listen(1)
        print("[net-audio] listening on {}:{}".format(self.bind_host, self.port), file=sys.stderr)

        while self._running:
            try:
                sock.settimeout(1.0)
                conn, addr = sock.accept()
            except socket.timeout:
                continue
            except Exception as exc:
                print("[net-audio] accept error:", exc, file=sys.stderr)
                continue

            print("[net-audio] client connected from", addr, file=sys.stderr)
            self._handle_client(conn)
            print("[net-audio] client disconnected", file=sys.stderr)

        sock.close()

    def _handle_client(self, conn):
        """Read length-prefixed float32 chunks from a connected client."""
        conn.settimeout(None)
        while self._running:
            # Read 4-byte length header
            header = self._recv_all(conn, 4)
            if header is None:
                break
            nbytes = struct.unpack("<I", header)[0]
            if nbytes == 0:
                break

            # Read payload
            payload = self._recv_all(conn, nbytes)
            if payload is None:
                break

            chunk = np.frombuffer(payload, dtype=np.float32)
            if self.channels > 1 and chunk.ndim == 1:
                # Client sends interleaved stereo — average to mono.
                chunk = chunk.reshape(-1, self.channels).mean(axis=1)
            elif self.channels == 1 and chunk.ndim > 1:
                chunk = chunk.mean(axis=1)

            # Drop oldest if queue full to maintain low latency.
            try:
                self._q.put_nowait(chunk.astype(np.float32))
            except queue.Full:
                try:
                    self._q.get_nowait()
                    self._q.put_nowait(chunk.astype(np.float32))
                except queue.Empty:
                    pass

    @staticmethod
    def _recv_all(conn, n):
        """Receive exactly n bytes or return None on disconnect."""
        buf = b""
        while len(buf) < n:
            try:
                data = conn.recv(n - len(buf))
            except (OSError, socket.error):
                return None
            if not data:
                return None
            buf += data
        return buf

    def __iter__(self):
        # type: () -> Iterator[np.ndarray]
        while self._running:
            try:
                yield self._q.get(timeout=0.1)
            except queue.Empty:
                continue

    def close(self):
        self._running = False

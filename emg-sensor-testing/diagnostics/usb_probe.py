#!/usr/bin/env python3
"""Probe the board's USB stream directly, without the logger.

Answers the two questions from hardware bring-up:
  1. Is the board sending packets at all? (No bytes at all = firmware built without USB CDC on boot,
     so its output goes to the COM/UART pins instead of the USB port.)
  2. Does opening the port reboot the board? (ROM boot text in the stream, or the board clock jumping
     back.) Presetting DTR/RTS off before opening -- meant to *avoid* a reset -- is what reboots the
     ESP32-S3 over its own USB port.

Usage: python diagnostics/usb_probe.py [--port /dev/cu.usbmodem101]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from log_emg import BAUD, PacketParser, find_port  # noqa: E402


def probe(port, label, preset_lines_off, send_rate=None, seconds=4):
    import serial

    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, BAUD, 0.05
    if preset_lines_off:
        s.dtr = s.rts = False
    s.open()
    raw, t0, sent = b"", time.time(), False
    while time.time() - t0 < seconds:
        raw += s.read(8192)
        if send_rate and not sent and time.time() - t0 > 1.5:
            s.write(f"RATE {send_rate}\n".encode())
            sent = True
    s.close()

    packets = PacketParser().feed(raw)
    t = np.array([p[0] for p in packets], dtype=np.int64)
    n = np.array([len(p[2]) for p in packets])
    back = np.flatnonzero(np.diff(t) < 0)
    if len(back):  # measure the rate only after the last reboot
        t, n = t[back[-1] + 1:], n[back[-1] + 1:]
    rate = n[1:].sum() / ((t[-1] - t[0]) / 1e6) if len(t) > 2 else 0
    print(f"{label}: {len(raw):,} bytes, {len(packets)} packets, ROM boot text: {b'ESP-ROM' in raw}, "
          f"board clock jumped back {len(back)}x, rate after last reboot {rate:,.0f} readings/s")
    i = raw.find(b"ESP-ROM")
    if i >= 0:
        print("   ", raw[i:i + 120])
    time.sleep(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port (auto-detected if only one)")
    args = ap.parse_args()
    port = args.port or find_port()
    probe(port, "1) plain open            ", False)
    probe(port, "2) DTR/RTS preset off    ", True)
    probe(port, "3) plain open + RATE cmd ", False, send_rate=10000)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Records one set from the fatigue sensor in capture mode.

Flash fatigue_sensor with CAPTURE_MODE 1, plug it in over USB, then:

    python tools/record.py --subject S01 --session 1 --set control
    python tools/record.py --subject S01 --session 1 --set fatigue

It shows a live status line (EMG level, clipping, reps, stream health)
while recording. Press Ctrl-C when the set is done; it then asks for the
failure rep and RPE and saves:

    recordings/S01_s1_fatigue_20261027-153012.bin   the raw stream, as received
    recordings/S01_s1_fatigue_20261027-153012.json  who, which set, labels

SAFETY: keep the laptop on battery, charger unplugged, whenever the
electrodes are on and USB is connected.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import serial
import serial.tools.list_ports

import capture_format as cf

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(HERE, "..", "recordings")


def find_port() -> str:
    ports = list(serial.tools.list_ports.comports())
    # ESP32-S3 native USB (303a) or the usual USB-UART bridges on dev kits.
    likely = [p for p in ports if p.vid in (0x303A, 0x10C4, 0x1A86, 0x0403)]
    if len(likely) == 1:
        return likely[0].device
    names = "\n  ".join(f"{p.device}  {p.description}" for p in ports) or "(none)"
    sys.exit(f"Couldn't pick a serial port; pass --port. Ports:\n  {names}")


def ask(prompt: str, cast=str, allow_blank=True):
    while True:
        raw = input(prompt).strip()
        if not raw and allow_blank:
            return None
        try:
            return cast(raw)
        except ValueError:
            print("  try again")


class LiveStatus:
    """Decodes the stream as it arrives, for the status line."""

    def __init__(self) -> None:
        self.reader = cf.FrameReader()
        self.window: list[np.ndarray] = []
        self.clipped = 0
        self.reps = 0
        self.rep_count_at_start: int | None = None
        self.overflows = 0
        self.dropped = 0
        self.emg_per_mv = 10
        self.got_status = False

    def feed(self, data: bytes) -> None:
        for ftype, p in self.reader.feed(data):
            if ftype == cf.EMG:
                n, clipped = p[4], p[5]
                self.window.append(np.frombuffer(p, dtype="<u2", count=n, offset=6) / self.emg_per_mv)
                self.window = self.window[-31:]  # ~0.5 s
                self.clipped += clipped
            elif ftype == cf.REP:
                count = p[4] | p[5] << 8
                if self.rep_count_at_start is None:
                    self.rep_count_at_start = count - 1
                self.reps = count - self.rep_count_at_start
            elif ftype == cf.STATUS and len(p) >= 25:
                self.got_status = True
                self.emg_per_mv = p[7] | p[8] << 8
                self.overflows = int.from_bytes(p[17:21], "little")
                self.dropped = int.from_bytes(p[21:25], "little")

    def line(self, elapsed: float, nbytes: int) -> str:
        if not self.got_status:
            return f"{elapsed:5.0f}s  waiting for capture frames… (is CAPTURE_MODE 1 flashed?)"
        rms = float(np.std(np.concatenate(self.window))) if self.window else 0.0
        health = []
        if self.clipped:
            health.append(f"CLIPPED {self.clipped}")
        if self.overflows:
            health.append(f"adc overflows {self.overflows}")
        if self.dropped:
            health.append(f"dropped frames {self.dropped}")
        if self.reader.crc_errors:
            health.append(f"crc errors {self.reader.crc_errors}")
        bar = "#" * min(40, int(rms / 5))
        return (
            f"{elapsed:5.0f}s  reps {self.reps:2d}  emg rms {rms:6.1f} mV {bar:<40}  "
            f"{nbytes / max(elapsed, 1e-9) / 1000:4.1f} KB/s  {' · '.join(health) or 'ok'}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True, help="anonymous ID, e.g. S01")
    ap.add_argument("--session", required=True, type=int, help="session number for this person")
    ap.add_argument("--set", required=True, choices=["control", "fatigue", "rest"], dest="set_type",
                    help="control (light, 12 reps), fatigue (to failure), or rest (noise baseline)")
    ap.add_argument("--weight-kg", type=float)
    ap.add_argument("--port", help="serial port (default: auto-detect)")
    ap.add_argument("--baud", type=int, default=921600, help="must match CAPTURE_BAUD (ignored on native USB)")
    ap.add_argument("--out-dir", default=DEFAULT_DIR)
    args = ap.parse_args()

    port = args.port or find_port()
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = os.path.join(args.out_dir, f"{args.subject}_s{args.session}_{args.set_type}_{stamp}")

    print(f"Recording from {port}. Ctrl-C when the set is done.")
    print("Reminder: laptop on battery, charger unplugged.\n")
    status = LiveStatus()
    nbytes = 0
    started = time.time()
    with serial.Serial(port, args.baud, timeout=0.1) as ser, open(base + ".bin", "wb") as out:
        ser.reset_input_buffer()
        last_print = 0.0
        try:
            while True:
                data = ser.read(4096)
                if data:
                    out.write(data)
                    nbytes += len(data)
                    status.feed(data)
                now = time.time()
                if now - last_print > 0.25:
                    print("\r" + status.line(now - started, nbytes), end="", flush=True)
                    last_print = now
        except KeyboardInterrupt:
            pass
    duration = time.time() - started
    print(f"\n\nSaved {nbytes / 1000:.0f} KB, {duration:.0f} s, to {base}.bin")

    print("Labels for this set (Enter to skip):")
    failure_rep = ask("  Failure rep (last rep completed with good form): ", int) if args.set_type == "fatigue" else None
    rpe = ask("  RPE (1-10): ", float)
    notes = ask("  Notes: ") or ""
    meta = {
        "subject": args.subject,
        "session": args.session,
        "set_type": args.set_type,
        "weight_kg": args.weight_kg,
        "failure_rep": failure_rep,
        "rpe": rpe,
        "notes": notes,
        "recorded_at": dt.datetime.now().isoformat(timespec="seconds"),
        "duration_s": round(duration, 1),
        "port": port,
        "reps_seen_live": status.reps,
    }
    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved labels to {base}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

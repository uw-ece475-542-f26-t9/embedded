#!/usr/bin/env python3
"""Record MyoWare RAW + ENV from the ESP32-S3 into data/<name>.csv plus data/<name>.json.

Usage:
  python log_emg.py <name> [--seconds N] [--rate 10000] [--port /dev/cu.usbmodemXXXX]
  python log_emg.py <name> --hold          # timed hold: 5 s countdown, 60 s hold, spoken cues

With --countdown/--hold the "start" and "end" markers are placed automatically and the
recording ends on its own. You can still type a label + Enter to drop a marker. Ctrl+C stops.
"""
import argparse
import json
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

SYNC = b"\xa5\x5a"
HEADER = struct.Struct("<HIHI")  # sync, board time (us), pair count, overflow count
MAX_PAIRS = 128
BAUD = 921600
DATA_DIR = Path(__file__).resolve().parent / "data"


class PacketParser:
    """Turns the board's byte stream into packets, resyncing if bytes go missing."""

    def __init__(self):
        self.buf = bytearray()
        self.resyncs = 0
        self.synced = False

    def feed(self, data):
        self.buf += data
        packets = []
        while True:
            start = self.buf.find(SYNC)
            if start < 0:
                del self.buf[:-1]  # keep a possible first half of a sync marker
                return packets
            if start > 0:
                if self.synced:
                    self.resyncs += 1
                del self.buf[:start]
            if len(self.buf) < HEADER.size:
                return packets
            _, t_us, n, overflows = HEADER.unpack_from(self.buf)
            if n == 0 or n > MAX_PAIRS:
                del self.buf[:1]  # sync bytes appeared inside data, not a real header
                continue
            end = HEADER.size + 4 * n
            if len(self.buf) < end + len(SYNC):
                return packets  # wait for the payload and the next packet's sync
            if self.buf[end:end + len(SYNC)] != SYNC:
                del self.buf[:1]  # payload was cut short
                continue
            pairs = np.frombuffer(bytes(self.buf[HEADER.size:end]), dtype="<u2").reshape(-1, 2)
            packets.append((t_us, overflows, pairs))
            self.synced = True
            del self.buf[:end]


def drain(ser, seconds):
    """Read and throw away everything for a while (reset_input_buffer misses data still in flight)."""
    end = time.time() + seconds
    while time.time() < end:
        ser.read(8192)


def cue(text):
    """Print a cue and speak it, so whoever is holding the weight doesn't need to watch the screen."""
    print(f"  >> {text}")
    if shutil.which("say"):
        subprocess.Popen(["say", text])


def timed_events(countdown, seconds):
    """(seconds after recording starts, spoken cue, marker label or None), in time order."""
    events = []
    if countdown:
        events.append((0, "Get in position", None))
        events += [(countdown - k, str(k), None) for k in (3, 2, 1) if k < countdown]
        events.append((countdown, "Go", "start"))
    if seconds:
        end = countdown + seconds
        events += [(end - k, f"{k} seconds left", None) for k in (30, 10) if seconds > k + 5]
        events.append((end, "Stop", "end" if countdown else None))
    return sorted(events, key=lambda e: e[0])


def find_port():
    from serial.tools import list_ports

    ports = [p.device for p in list_ports.comports()
             if p.device.startswith("/dev/cu.") and "usb" in p.device.lower()]
    if len(ports) != 1:
        sys.exit(f"Found {len(ports)} USB serial ports {ports}; pass one with --port")
    return ports[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="trial name, e.g. A_heavy1")
    ap.add_argument("--seconds", type=float, help="stop automatically after this many seconds (after the countdown)")
    ap.add_argument("--countdown", type=float, default=0, help="seconds to get in position; marks 'start' at the end")
    ap.add_argument("--hold", action="store_true", help="timed hold: same as --countdown 5 --seconds 60")
    ap.add_argument("--squeeze-cues", action="store_true",
                    help="Test 1: spoken 'Squeeze'/'Relax' cues, 3 s squeezes at 5, 35 and 65 s; records 75 s")
    ap.add_argument("--rate", type=int, default=10000, help="readings per second per pin")
    ap.add_argument("--port", help="serial port (auto-detected if only one)")
    args = ap.parse_args()
    if args.hold:
        args.countdown = args.countdown or 5
        args.seconds = args.seconds or 60
    if args.squeeze_cues:
        args.seconds = args.seconds or 75

    import serial

    DATA_DIR.mkdir(exist_ok=True)
    csv_path, json_path = DATA_DIR / f"{args.name}.csv", DATA_DIR / f"{args.name}.json"
    if csv_path.exists():
        sys.exit(f"{csv_path} already exists; pick another name")

    # Plain open: on the ESP32-S3's USB port, presetting DTR/RTS off actually resets the chip.
    ser = serial.Serial(args.port or find_port(), BAUD, timeout=0.05)
    drain(ser, 1.0)  # old readings queued while nobody was listening
    ser.write(f"RATE {args.rate}\n".encode())
    drain(ser, 0.5)  # readings taken at the old rate

    parser = PacketParser()
    chunks, times, overflows, markers = [], [], [], []
    count = 0  # RAW/ENV pairs recorded so far

    def read_markers():
        while True:
            try:
                label = input().strip() or "mark"
            except EOFError:
                return
            markers.append([count, label])
            print(f"  marker '{label}' at {count / args.rate:.1f} s")

    threading.Thread(target=read_markers, daemon=True).start()
    print(f"Recording {args.name} at {args.rate}/s. Type a label + Enter to mark, Ctrl+C to stop.")

    events = timed_events(args.countdown, args.seconds)
    if args.squeeze_cues:
        events = sorted(events + [e for t in (5, 35, 65) for e in ((t, "Squeeze", None), (t + 3, "Relax", None))],
                        key=lambda e: e[0])
    started = time.time()
    stop_at = started + args.countdown + args.seconds if args.seconds else None
    next_status = started + 5
    try:
        while True:
            while events and time.time() - started >= events[0][0]:
                _, text, label = events.pop(0)
                cue(text)
                if label:
                    markers.append([count, label])
            if stop_at is not None and time.time() >= stop_at:
                break
            for t_us, ovf, pairs in parser.feed(ser.read(8192)):
                chunks.append(pairs)
                times.append(t_us)
                overflows.append(ovf)
                count += len(pairs)
            if time.time() >= next_status:
                print(f"  {count / args.rate:.0f} s recorded")
                next_status += 5
    except KeyboardInterrupt:
        if markers and markers[-1][1] == "start":
            markers.append([count, "end"])  # stopped early: the hold ends here
    ser.close()

    if len(chunks) < 2:
        sys.exit("No data received. Check the port, the cable, and that the firmware is flashed.")

    data = np.concatenate(chunks)
    gaps_us = np.diff(np.array(times, dtype=np.int64))
    reboots = int(np.sum(gaps_us < 0))  # board clock jumped back (trials are far shorter than its 71 min wrap)
    measured_rate = (count - len(chunks[0])) / (gaps_us[gaps_us >= 0].sum() / 1e6)
    lost = overflows[-1] - overflows[0]
    ok = lost == 0 and parser.resyncs == 0 and reboots == 0

    np.savetxt(csv_path, data, fmt="%d", delimiter=",", header="raw,env", comments="")
    json_path.write_text(json.dumps({
        "name": args.name,
        "requested_rate_hz": args.rate,
        "rate_hz": measured_rate,
        "readings": int(count),
        "seconds": count / measured_rate,
        "board_overflows": int(lost),
        "resyncs": parser.resyncs,
        "board_reboots": reboots,
        "markers": markers,
        "ok": bool(ok),
    }, indent=2))

    print(f"\nSaved {csv_path.name}: {count:,} readings, {count / measured_rate:.1f} s, "
          f"{measured_rate:,.0f} readings/s, {len(markers)} markers")
    if ok:
        print("OK: no readings lost")
    else:
        print(f"LOST DATA ({lost} board overflows, {parser.resyncs} resyncs, {reboots} board reboots). "
              "Redo this trial.")
    if abs(measured_rate / args.rate - 1) > 0.05:
        print(f"Note: rate is {measured_rate / args.rate - 1:+.0%} off the setting; analysis uses the measured rate.")
    raw, env = data[:, 0], data[:, 1]
    print(f"RAW {raw.min()}-{raw.max()} (quiet ~1900), ENV max {env.max()}")
    if raw.min() <= 5 or raw.max() >= 4090:
        print("RAW hit the top/bottom of the range (flat tops). Check sticker contact, then redo.")
    if env.max() >= 4000:
        print("ENV maxed out. Turn the MyoWare gain knob down a little.")


if __name__ == "__main__":
    main()

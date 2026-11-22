"""Live EMG loudness meter: is muscle signal reaching the board?

    python -m pipeline.meter [--port /dev/cu.usbmodem101]

Prints 4 times a second: 20-450 Hz loudness (mV RMS) with a bar, the raw level (mV at the pin),
its spread, and clipped readings. Relax, flex, relax: a working chain jumps from ~3 mV to tens of mV.
Needs the capture build on the board. Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from .capture import EMG, EMG_HEAD, STATUS, STATUS_FMT, FrameReader
from .config import load_config
from .lab import open_port


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--every", type=float, default=0.25, help="seconds per line")
    ap.add_argument("--full-scale", type=float, default=60.0, help="mV RMS that fills the bar")
    args = ap.parse_args()
    cfg = load_config()
    port = open_port(cfg, args.port)

    hz = cfg["input"]["emg_hz"]
    sos = butter(4, [20, 450], "band", fs=hz, output="sos")
    zi, per_mv = None, 10
    reader, buf, clip, last = FrameReader(), [], 0, time.monotonic()
    floor, early = None, []
    print("Relax for 3 s (sets the floor), then flex. Ctrl-C to stop.\n")
    try:
        while True:
            for ftype, p in reader.feed(port.read(4096)):
                if ftype == STATUS and len(p) >= STATUS_FMT.size:
                    per_mv = STATUS_FMT.unpack_from(p)[4] or 10
                elif ftype == EMG and len(p) >= EMG_HEAD.size:
                    _, n, c = EMG_HEAD.unpack_from(p)
                    if len(p) >= EMG_HEAD.size + 2 * n:
                        buf.append(np.frombuffer(p, "<u2", n, EMG_HEAD.size) / per_mv)
                        clip += c
            now = time.monotonic()
            if now - last < args.every:
                continue
            last = now
            if not buf:
                print("  no EMG frames (is the capture build on the board?)")
                continue
            raw = np.concatenate(buf)
            buf = []
            if zi is None:
                zi = sosfilt_zi(sos) * raw[0]
            x, zi = sosfilt(sos, raw, zi=zi)
            rms = float(np.sqrt(np.mean(x ** 2)))
            if floor is None:   # the floor: median of the first 3 s (keep the arm relaxed)
                early.append(rms)
                if len(early) * args.every >= 3:
                    floor = float(np.median(early[2:]))
                    print(f"  floor {floor:.1f} mV. Now flex.")
                continue
            bar = "#" * int(min(rms / args.full_scale, 1) * 40)
            print(f"{rms:6.1f} mV  x{rms / floor:4.1f}  |{bar:<40}|  level {raw.mean():5.0f} mV  "
                  f"spread {raw.std():5.1f}  clipped {clip}", flush=True)
            clip = 0
    except KeyboardInterrupt:
        print()
    finally:
        port.close()


if __name__ == "__main__":
    main()

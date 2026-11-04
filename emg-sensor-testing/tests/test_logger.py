#!/usr/bin/env python3
"""Tests for the logger's packet decoding and spoken-cue timing (no board needed).

Usage: python tests/test_logger.py        (or: pytest tests/)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from log_emg import HEADER, SYNC, PacketParser, timed_events  # noqa: E402

rng = np.random.default_rng(1)


def make_stream(n_packets=300, pairs_per_packet=10):
    """Fake board output, including data bytes that look like a sync marker."""
    packets, truth = [], []
    for k in range(n_packets):
        pairs = rng.integers(0, 4096, size=(pairs_per_packet, 2), dtype=np.uint16)
        pairs[0] = [0x5AA5 & 0xFFF, 0xA5]
        truth.append(pairs)
        packets.append(HEADER.pack(0x5AA5, 1000 * k, pairs_per_packet, 0) + pairs.astype("<u2").tobytes())
    return b"\x00\xff garbage" + b"".join(packets), truth


def feed_in_chunks(parser, data):
    out, i = [], 0
    while i < len(data):
        n = int(rng.integers(1, 3000))
        out += parser.feed(data[i:i + n])
        i += n
    return out


def test_clean_stream_decodes_exactly():
    stream, truth = make_stream()
    parser = PacketParser()
    out = feed_in_chunks(parser, stream)
    got = np.concatenate([p[2] for p in out])
    # the last packet waits for the next sync marker, so one fewer comes out
    assert len(out) == len(truth) - 1
    assert np.array_equal(got, np.concatenate(truth[:len(out)]))
    assert parser.resyncs == 0


def test_lost_bytes_are_flagged():
    stream, _ = make_stream()
    damaged = stream[:len(stream) // 2] + stream[len(stream) // 2 + 37:]
    parser = PacketParser()
    feed_in_chunks(parser, damaged)
    assert parser.resyncs > 0  # the logger reports LOST DATA when this is non-zero


def test_old_128_reading_packets_still_decode():
    stream, truth = make_stream(n_packets=20, pairs_per_packet=128)
    out = feed_in_chunks(PacketParser(), stream)
    assert np.array_equal(np.concatenate([p[2] for p in out]), np.concatenate(truth[:len(out)]))


def test_hold_cues():
    events = timed_events(5, 60)
    assert [(t, text) for t, text, _ in events] == [
        (0, "Get in position"), (2, "3"), (3, "2"), (4, "1"), (5, "Go"),
        (35, "30 seconds left"), (55, "10 seconds left"), (65, "Stop")]
    assert [label for *_, label in events if label] == ["start", "end"]


def test_sync_marker_bytes():
    assert SYNC == b"\xa5\x5a"


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError:
            failed += 1
            print(f"FAIL  {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

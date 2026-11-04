#!/usr/bin/env bash
# Regenerate every figure and report in results/ from the recordings in data/.
# Usage: ./reproduce.sh          (set PY=/path/to/python to use a specific interpreter)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
R=results/reports
mkdir -p "$R"
run() { local out=$1; shift; echo "\$ python $*" > "$R/$out.txt"; "$PY" "$@" | tee -a "$R/$out.txt"; }

# Test 1: signal check
run test1_signal_A analyze.py signal --rest A_rest --squeeze A_squeeze
run test1_signal_B analyze.py signal --rest B_rest2 --squeeze B_squeeze2

# Test 2: aliasing check (B_rest/B_squeeze were recorded on the old 128-reading-packet firmware and fail)
run test2_alias_A analyze.py alias --rest A_rest --rest-alt A_rest_10500 --squeeze A_squeeze
run test2_alias_B_old_firmware analyze.py alias --rest B_rest --rest-alt B_rest_10500 --squeeze B_squeeze
run test2_alias_B analyze.py alias --rest B_rest2 --rest-alt B_rest2_10500 --squeeze B_squeeze2

# Squeeze and rest comparisons
run squeezes_A analyze.py squeezes --squeeze A_squeeze
run rest_A analyze.py squeezes --squeeze A_squeeze --rest
run squeezes_B analyze.py squeezes --squeeze B_squeeze2
run rest_B analyze.py squeezes --squeeze B_squeeze2 --rest

# Test 3: static holds (Person B's loaded holds are the three 10 s weight-finding trials)
run test3_hold_A analyze.py hold --light A_light --heavy A_heavy1 A_heavy2 A_heavy3 --squeeze A_squeeze
run test3_hold_B analyze.py hold --light B_light2 --heavy B_try1 B_try2 B_try3 --squeeze B_squeeze2

# Diagnostics
run diag_hold_summary diagnostics/hold_summary.py A_squeeze:A_light,A_heavy1,A_heavy2,A_heavy3 \
    B_squeeze2:B_light2,B_try1,B_try2,B_try3
run diag_slice_noise diagnostics/slice_noise.py A_light B_light2 A_heavy1 A_heavy2 A_heavy3
run diag_packet_noise_old_firmware diagnostics/packet_noise.py A_rest B_rest --period 128
run diag_packet_noise_new_firmware diagnostics/packet_noise.py B_rest2 --period 10
run diag_noise_sources diagnostics/noise_sources.py A_rest B_rest B_rest2
run diag_check_B_check diagnostics/check_recording.py B_check --compare A_squeeze
run diag_check_B_light diagnostics/check_recording.py B_light --chunk 5

echo "All figures in results/, all reports in results/reports/"

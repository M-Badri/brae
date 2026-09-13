#!/usr/bin/env bash
# The truncated Neumann series survives EVERY path through deviceJacobiBiCGStab.
#
# THE DEFECT THIS HOLDS (found 2026-09-12). The pointer-normFactor entry point runs the device
# conditional-graph loop when it can and otherwise falls back to the host loop -- and the fallback call
# forwarded every argument EXCEPT polyDeg. So a solve that took the fallback ran the bare DIAGONAL where
# the caller asked for a degree-d series: a checkEvery above 1, BRAE_BICG_HOST_LOOP=1,
# BRAE_NORMFACTOR_HOST=1, or the graph path declining. Nothing in a residual line shows it -- the solve
# still reaches the tolerance the case named, it just stops somewhere else -- so it surfaced only when
# the FP-2 preconditioner policy put the ENERGY solve on the series and normfactor_device_identity's
# rhoBox arm found the two normFactor paths disagreeing from iteration 2.
#
# FIXTURE: validation/rhoBox, whose `"(U|h|e)"` entry names PBiCGStab with DILU and whose h is relaxed
# at 0.7 -- so on the CUDA mirror the FP-2 policy runs a degree-6 series there, announced.
#
#   ARM 1     the device graph loop against the host loop (BRAE_BICG_HOST_LOOP=1): every residual line
#             and every written field identical. This is the one the defect broke.
#   ARM 2     the device graph loop against the host-read normFactor (BRAE_NORMFACTOR_HOST=1), which
#             reaches the same fallback by the other door: identical.
#   ARM 3     the series is announced on h in every arm (else the arms above compare two DILU runs).
#   CONTROL   BRAE_DILU_HE=1 (the case's own DILU instead of the series) must DIFFER from the default,
#             and so must BRAE_POLY_KE-style degree 1 -- here, the honoured-DILU arm. Without this the
#             gate would pass on a fixture where the preconditioner cannot change the iterate at all.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: $BRAE not built"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
[ -d "$ROOT/validation/rhoBox" ] || { echo "SKIP: validation/rhoBox missing"; exit 77; }
set -u
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-78s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

prep() {
    mkdir -p "$1"; cp -r "$ROOT/validation/rhoBox/constant" "$ROOT/validation/rhoBox/system" "$1/"
    if [ -d "$ROOT/validation/rhoBox/0.orig" ]; then cp -r "$ROOT/validation/rhoBox/0.orig" "$1/0"
    else cp -r "$ROOT/validation/rhoBox/0" "$1/0"; fi
    python3 - "$1" <<'PY'
import re, sys
d = sys.argv[1]; c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime 50;', s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval 50;', s)
s = re.sub(r'\bwritePrecision\s+[^;]*;', 'writePrecision 15;', s)
open(c, 'w').write(s)
PY
}
run() {   # run <dir> <extra env>
    prep "$1"
    ( cd "$1" && env $2 BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" "$1" > log 2>&1 ) \
        || { echo "FAIL: brae crashed in $(basename "$1")"; tail -5 "$1/log"; exit 1; }
    grep -q "^Time = 50 " "$1/log" || { echo "FAIL: $(basename "$1") did not reach iteration 50"; exit 1; }
}
lines() { grep -E "^Time = |Solving for" "$1/log"; }
fields() {   # fields <a> <b> -> prints differing names
    local d=0 n=0 f b
    for f in "$1"/50/*; do
        [ -f "$f" ] || continue
        b=$(basename "$f"); n=$((n+1))
        cmp -s "$f" "$2/50/$b" || { echo "     differs: $b"; d=$((d+1)); }
    done
    [ "$n" -gt 0 ] || { echo "     no written fields"; return 1; }
    [ "$d" = 0 ]
}

run "$W/dev"       ""
run "$W/hostloop"  "BRAE_BICG_HOST_LOOP=1"
run "$W/hostnf"    "BRAE_NORMFACTOR_HOST=1"
run "$W/dilu"      "BRAE_DILU_HE=1"

for a in dev hostloop hostnf; do
    grep -q "solvers/h preconditioner: case asks 'DILU', brae preconditions with a degree-6 truncated Neumann series" "$W/$a/log" \
        && say "ARM 3  the $a arm announces the degree-6 series on h" ok \
        || say "ARM 3  the $a arm announces the degree-6 series on h" FAIL
done
# ...and the two normFactor paths really are two paths, or ARM 2 compares a run with itself.
grep -q "normFactor: device-resident" "$W/dev/log" \
    && say "ARM 3  the default arm keeps the normFactor on the device" ok \
    || say "ARM 3  the default arm keeps the normFactor on the device" FAIL
grep -q "normFactor: read to the host" "$W/hostnf/log" \
    && say "ARM 3  the hostnf arm reads it to the host (the other door to the fallback)" ok \
    || say "ARM 3  the hostnf arm reads it to the host (the other door to the fallback)" FAIL

if diff <(lines "$W/dev") <(lines "$W/hostloop") > /dev/null; then
    say "ARM 1  device graph loop == host loop: every residual line identical" ok
else
    say "ARM 1  device graph loop == host loop: every residual line identical" FAIL
    diff <(lines "$W/dev") <(lines "$W/hostloop") | head -4
fi
fields "$W/dev" "$W/hostloop" && say "ARM 1  ...and every written field at 50 is byte-identical" ok \
                              || say "ARM 1  ...and every written field at 50 is byte-identical" FAIL

if diff <(lines "$W/dev") <(lines "$W/hostnf") > /dev/null; then
    say "ARM 2  device graph loop == host-read normFactor: every residual line identical" ok
else
    say "ARM 2  device graph loop == host-read normFactor: every residual line identical" FAIL
    diff <(lines "$W/dev") <(lines "$W/hostnf") | head -4
fi
fields "$W/dev" "$W/hostnf" && say "ARM 2  ...and every written field at 50 is byte-identical" ok \
                            || say "ARM 2  ...and every written field at 50 is byte-identical" FAIL

if diff <(lines "$W/dev") <(lines "$W/dilu") > /dev/null; then
    say "CONTROL  the case's own DILU gives a DIFFERENT iterate here (so the arms can fail)" FAIL
else
    say "CONTROL  the case's own DILU gives a DIFFERENT iterate here (so the arms can fail)" ok
fi

echo
[ "$fail" = 0 ] && echo "PASS" || echo "FAIL"
exit "$fail"

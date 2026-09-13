#!/usr/bin/env bash
# A DIVERGED RUN MUST REFUSE, not report a wall time for a run that computed nothing.
#
# T-5 (bench/results/rhoSimpleFoam_tutorials_gb10.md, 2026-09-13). squareBend run subsonic -- its own
# `transonic yes` flipped to no with the inlet at 0.1 kg/s -- diverges. OpenFOAM aborts on it: the
# thermo inversion throws on the temperature within about twenty iterations. brae's CUDA arm ran all
# 100 iterations printing `e nan   p nan   k nan`, exited 0, and handed back a wall time. That is the
# silent substitution this project refuses everywhere else, so rhoSimpleFoamDriver.cu now checks the
# residuals it has already brought to the host for its summary line and throws on the first one that
# is not finite.
#
#   ORACLE      OpenFOAM rhoSimpleFoam on the same case must FAIL (non-zero exit). If OpenFOAM ever
#               completes this case the reproducer has stopped reproducing and the gate says so
#               rather than testing nothing.
#   ARM         brae on the same case must fail too, name the field, and say the run diverged.
#   CONTROL     brae with BRAE_ALLOW_NONFINITE=1 must run to completion. This proves the refusal is
#               the new check and not some other failure on the same input.
#   FAIL-PROOF  brae on the tutorial AS SHIPPED (transonic, 0.5 kg/s) must still reach the last
#               iteration with the check armed -- the guard must not fire on a healthy run.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-100}
[ -x "$BRAE" ] || { echo "SKIP: $BRAE not built"; exit 77; }
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true   # before set -u: the bashrc reads unset variables
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: OpenFOAM's meshers not on PATH"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
TUT="${FOAM_TUTORIALS:-}/compressible/rhoSimpleFoam/squareBend"
[ -d "$TUT" ] || { echo "SKIP: tutorials not found"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-74s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# stage <dir> <transonic yes|no> <massFlowRate>
stage() {
    local d="$1" tr="$2" mf="$3"
    rm -rf "$d"; cp -r "$TUT" "$d"; rm -rf "$d"/0 "$d"/[1-9]* "$d"/log.*
    cp -r "$d/0.orig" "$d/0"
    python3 - "$d" "$N" "$tr" "$mf" <<'PY'
import re, sys
d, n, tr, mf = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'\nfunctions\s*\{.*', '\n', s, flags=re.S)
for k, v in (('startFrom','latestTime'), ('stopAt','endTime'), ('endTime',n), ('writeInterval',n)):
    s = re.sub(r'^%s\s+.*' % k, '%-15s %s;' % (k, v), s, flags=re.M)
open(c, 'w').write(s + '\n')
p = d + '/system/fvSolution'; s = open(p).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
s, k = re.subn(r'\btransonic\s+\w+\s*;', 'transonic       %s;' % tr, s)
assert k == 1, 'expected one transonic entry'
open(p, 'w').write(s)
u = d + '/0/U'; s = open(u).read()
s, k = re.subn(r'(massFlowRate\s+constant\s+)[-0-9.eE]+', r'\g<1>' + mf, s)
assert k == 1, 'expected one massFlowRate entry'
open(u, 'w').write(s)
PY
    ( cd "$d" && blockMesh > log.blockMesh 2>&1 ) || return 1
    return 0
}

echo "rho_nonfinite_refusal: a diverged run must refuse ($N iterations, squareBend)"

# ---- the diverging case -------------------------------------------------------------------------
stage "$W/div" no 0.1 || { echo "SKIP: blockMesh failed"; exit 77; }

# ORACLE: OpenFOAM itself does not survive this input.
cp -r "$W/div" "$W/div_of"
( cd "$W/div_of" && rhoSimpleFoam > log.of 2>&1 ); of_rc=$?
of_reached=$(grep -c '^Time = ' "$W/div_of/log.of" 2>/dev/null || echo 0)
if [ "$of_rc" -ne 0 ]; then
    say "ORACLE   OpenFOAM aborts on it (exit $of_rc after $of_reached iterations)" PASS
else
    say "ORACLE   OpenFOAM completed this case -- the reproducer no longer diverges" FAIL
fi

# ARM: brae must refuse, and say what and why.
cp -r "$W/div" "$W/div_brae"
( cd "$W/div_brae" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$W/div_brae" > log.brae 2>&1 ); b_rc=$?
b_reached=$(grep -c '^Time = ' "$W/div_brae/log.brae" 2>/dev/null || echo 0)
if [ "$b_rc" -ne 0 ] && grep -qi "diverged" "$W/div_brae/log.brae" \
   && grep -qiE "residual for '[^']+' is (nan|infinite)" "$W/div_brae/log.brae"; then
    say "ARM      brae refuses and names the field (exit $b_rc after $b_reached iterations)" PASS
else
    say "ARM      brae did not refuse (exit $b_rc after $b_reached iterations)" FAIL
    grep -iE "nan|diverg" "$W/div_brae/log.brae" | head -3 | sed 's/^/           /'
fi
# and it must stop EARLY, not at the last iteration: a refusal on the final line saves nothing
if [ "$b_reached" -gt 0 ] && [ "$b_reached" -lt "$N" ]; then
    say "ARM      it stops when the divergence appears, not at the end" PASS
else
    say "ARM      expected a stop before iteration $N, got $b_reached" FAIL
fi

# CONTROL: the hatch restores the old behaviour, so the refusal is what stopped it.
cp -r "$W/div" "$W/div_ctl"
( cd "$W/div_ctl" && BRAE_ALLOW_NONFINITE=1 BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$W/div_ctl" > log.ctl 2>&1 )
c_reached=$(grep -c '^Time = ' "$W/div_ctl/log.ctl" 2>/dev/null || echo 0)
if [ "$c_reached" -eq "$N" ]; then
    say "CONTROL  BRAE_ALLOW_NONFINITE=1 runs all $N iterations (the old behaviour)" PASS
else
    say "CONTROL  expected $N iterations with the check off, got $c_reached" FAIL
fi

# ---- FAIL-PROOF: the tutorial as shipped must be untouched --------------------------------------
if stage "$W/ok" yes 0.5; then
    ( cd "$W/ok" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$W/ok" > log.ok 2>&1 ); ok_rc=$?
    ok_reached=$(grep -c '^Time = ' "$W/ok/log.ok" 2>/dev/null || echo 0)
    if [ "$ok_rc" -eq 0 ] && [ "$ok_reached" -eq "$N" ]; then
        say "FAILPROOF the healthy tutorial still runs all $N iterations" PASS
    else
        say "FAILPROOF the guard fired on a healthy run (exit $ok_rc, $ok_reached iterations)" FAIL
    fi
else
    say "FAILPROOF could not stage the healthy case" FAIL
fi

[ "$fail" -eq 0 ] && echo "rho_nonfinite_refusal: PASS" || echo "rho_nonfinite_refusal: FAIL"
exit "$fail"

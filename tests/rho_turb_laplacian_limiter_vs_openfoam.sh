#!/usr/bin/env bash
# THE TURBULENCE CLOSURE TAKES THE CASE'S `limited <psi>` LAPLACIAN, checked against OpenFOAM's OWN
# fvm::laplacian, at ASSEMBLY level.
#
# OpenFOAM's laplacian entry carries its own snGrad scheme (laplacianScheme.H:121-141 builds
# tsnGradScheme_ from the entry's Istream; laplacianScheme.C:44-78 reads the scheme word off the
# laplacianSchemes entry), so `Gauss linear limited 0.33` caps the non-orthogonal correction of
# laplacian(alpha*rho*DepsilonEff, epsilon) exactly as it does laplacian(muEff, U).
#
# brae's CUDA arm did not. rhoTurbulenceHook.cu assigned kin.correctedLaplacian and nothing else, so
# KEpsilonInput::snGradLimitCoeff kept its 0.0 default (kEpsilon.cuh:146) and 0 means UNCAPPED at the only
# consumer (turbulence_transport.cu). The driver DID parse the cap, into opt.co.snGradLimitCoeff
# (rhoSimpleFoamDriver.cu:322) -- a different member of a different struct from the one the closure reads,
# so the value was dead. Momentum, energy and pressure ran the capped correction while k and epsilon ran
# the uncapped one, under the case's own scheme name and with no notice. Audit 2026-09-15.
#
# THE ORACLE IS OPENFOAM. tools/dumpKEpsilon is OpenFOAM's own kEpsilon class -- same sources, registered
# with makeRASModel, linked against -lturbulenceModels -- with writes added and the equations untouched.
# kEpsilonDump.C:348 dumps fvm::laplacian(alpha*rho*DepsilonEff(), epsilon_) as a matrix of its OWN, the
# single observable the limiter moves. brae emits the same four arrays from its device assembly
# (TransportScheme::stageTag), in the convention kEpsilon_cpp.cu::captureSystem uses.
#
# WHY A DIFFERENTIAL AND NOT AN ABSOLUTE COMPARISON. At iteration 1 the closure already sees a post-
# momentum, post-pressure state and the two codes differ there, so DepsilonEff differs and the assembled
# laplacian inherits it: brae vs OpenFOAM on stage_epsLapSrc measures 6.5e-05, TEN TIMES the limiter's own
# effect (measured 2026-09-15). Differencing `limited` against `corrected` INSIDE each code cancels that
# drift and leaves exactly the limiter's contribution, which is what this gate is about. Measured:
# brae 6.650449e-06, OpenFOAM 6.650449e-06, apart by 1.71e-11.
#
#   ARM      max | (brae_limited - brae_corrected) - (OF_limited - OF_corrected) | on stage_epsLapSrc.
#   CONTROL  |OF_limited - OF_corrected| must be >= CONTROL_RATIO x the bound, or the limiter never
#            clipped on this fixture and the arm is vacuous. On an ORTHOGONAL mesh the correction is zero
#            and capping it changes nothing, which is why validation/rhoLimKE is tilted to 16.3 degrees
#            maximum non-orthogonality, and why every other rho fixture -- all `corrected` or
#            `orthogonal`, all near-orthogonal -- could host no version of this test.
#
# Measured 2026-09-15: ARM 1.71e-11, CONTROL 6.65e-06, a separation of 3.9e5. Reported but NOT gated:
# stage_epsLapD and stage_epsLapDUpper, whose own deltas are ~7e-10 -- the limiter barely touches the
# implicit coefficients, which is correct, so no bound over them could discriminate.
#
# FAIL-PROOF, measured: comment out `kin.snGradLimitCoeff = opt.co.snGradLimitCoeff;` in
# rhoTurbulenceHook.cu and rebuild -- brae's own delta collapses from 6.65e-06 to 7.8e-12, because the
# closure then assembles `limited` identically to `corrected`, and the ARM fails by six orders.
#
# TRAP, and it cost a cycle: the seed state is copied from an OpenFOAM time directory, and that directory
# carries uniform/time with the index it was written at. Left in place, Time::timeIndex() resumes from
# THERE, kEpsilonDump's `timeIndex() == BRAE_DUMP_STAGE_ITER` never matches, and the run completes happily
# having written no stage files at all. prep() below deletes 0/uniform.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDDIR="${BUILD:-$ROOT/build}"
BIN="$BUILDDIR/brae_rhoSimpleFoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
DEV=${DEV:-300}
BOUND=${BOUND:-2e-10}
CONTROL_RATIO=${CONTROL_RATIO:-1000}

[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$ROOT/validation/rhoLimKE" ] || { echo "SKIP: validation/rhoLimKE missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
command -v blockMesh     > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }
[ -f "${FOAM_USER_LIBBIN:-/nonexistent}/libdumpKEpsilon.so" ] \
    || { echo "SKIP: libdumpKEpsilon.so not built (cd tools/dumpKEpsilon && wmake libso)"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

prep()
{
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$ROOT/validation/rhoLimKE/constant" "$ROOT/validation/rhoLimKE/system" "$1/"
    if [ $# -ge 5 ]; then cp -r "$5" "$1/0"; rm -rf "$1/0/uniform"
    else cp -r "$ROOT/validation/rhoLimKE/0.orig" "$1/0"; fi
    ( cd "$1" && blockMesh > log.blockMesh 2>&1 ) || { echo "blockMesh failed"; exit 1; }
    sed -i "s/Gauss linear limited 0.33/Gauss linear $2/; s/default limited 0.33/default $2/" "$1/system/fvSchemes"
    sed -i "s/RASModel        kEpsilon;/RASModel        $3;/" "$1/constant/turbulenceProperties"
    END="$4" RAS="$3" python3 - "$1" <<'PREPPY'
import os, re, sys
c = os.path.join(sys.argv[1], 'system/controlDict'); s = open(c).read()
s = re.sub(r'startTime \d+;', 'startTime 0;', s)
s = re.sub(r'endTime \d+;', 'endTime %s;' % os.environ['END'], s)
s = re.sub(r'writeInterval \d+;', 'writeInterval %s;' % os.environ['END'], s)
s = re.sub(r'writePrecision \d+;', 'writePrecision 15;', s)
if os.environ['RAS'] == 'kEpsilonDump' and 'libdumpKEpsilon' not in s:
    s += '\nlibs ("libdumpKEpsilon.so");\n'
open(c, 'w').write(s)
PREPPY
}

one()
{
    python3 - "$1" <<'ONEPY'
import os, re, sys
c = os.path.join(sys.argv[1], 'system/controlDict'); s = open(c).read()
s = re.sub(r'endTime \d+;', 'endTime 1;', s)
s = re.sub(r'writeInterval \d+;', 'writeInterval 1;', s)
open(c, 'w').write(s)
v = os.path.join(sys.argv[1], 'system/fvSolution'); s = open(v).read()
s = re.sub(r'residualControl \{[^}]*\}', '', s)
open(v, 'w').write(s)
ONEPY
}

echo "rho turbulence laplacian limiter, against OpenFOAM's own fvm::laplacian (tools/dumpKEpsilon)"

# Uniform fields have zero gradient, so the non-orthogonal correction is zero and nothing could
# discriminate from 0.orig -- the fields must be developed first.
prep "$W/dev" "limited 0.33" kEpsilon "$DEV"
( cd "$W/dev" && rhoSimpleFoam > log.run 2>&1 ) || { echo "  developing the seed failed"; exit 1; }
SEEDT=$(ls -d "$W/dev"/[1-9]* 2>/dev/null | xargs -n1 basename | sort -n | tail -1)
[ -n "$SEEDT" ] || { echo "  no developed state written"; exit 1; }
printf '  seed state: OpenFOAM at iteration %s\n' "$SEEDT"

for spec in "of_lim|limited 0.33|kEpsilonDump" "of_corr|corrected|kEpsilonDump" \
            "br_lim|limited 0.33|kEpsilon"     "br_corr|corrected|kEpsilon"; do
    IFS='|' read -r tag lap ras <<< "$spec"
    prep "$W/$tag" "$lap" "$ras" 1 "$W/dev/$SEEDT"
    one  "$W/$tag"
    if [ "$ras" = kEpsilonDump ]; then
        ( cd "$W/$tag" && BRAE_DUMP_STAGE_ITER=1 rhoSimpleFoam > log.run 2>&1 ) \
            || { echo "  OpenFOAM ($tag) failed"; tail -4 "$W/$tag/log.run"; exit 1; }
        [ -f "$W/$tag/1/stage_epsLapSrc" ] \
            || { echo "  $tag wrote no stage_epsLapSrc (0/uniform left in place?)"; exit 1; }
    else
        ( cd "$W/$tag" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda BRAE_DUMP_STAGE="$W/dump_$tag" "$BIN" -case "$W/$tag" > log.run 2>&1 ) \
            || { echo "  brae ($tag) failed"; tail -4 "$W/$tag/log.run"; exit 1; }
        [ -f "$W/dump_$tag/epsLapSrc" ] \
            || { echo "  $tag wrote no epsLapSrc (stageTag not set?)"; exit 1; }
    fi
done

python3 - "$W" "$BOUND" "$CONTROL_RATIO" <<'CMPPY'
import re, sys
W, bound, ratio = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
def of(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<\w+>\s*\n(\d+)\s*\n\(\n(.*?)\n\)\n;', s, re.S)
    return [float(x) for x in m.group(2).split('\n')] if m else None
def br(p):
    ls = open(p).read().split('\n')
    return [float(x) for x in ls[1:1 + int(ls[0].split()[1])]]
rows = [('epsLapSrc', 'stage_epsLapSrc', True),
        ('epsLapD', 'stage_epsLapD', False),
        ('epsLapDUpper', 'stage_epsLapDUpper', False)]
bad = 0
need = bound * ratio
for tag, ofn, gated in rows:
    bl, bc = br(W + '/dump_br_lim/' + tag), br(W + '/dump_br_corr/' + tag)
    ol, oc = of(W + '/of_lim/1/' + ofn),    of(W + '/of_corr/1/' + ofn)
    if ol is None or len(bl) != len(ol):
        print('  %-12s shape mismatch, skipped' % tag); continue
    db = [x - y for x, y in zip(bl, bc)]
    do = [x - y for x, y in zip(ol, oc)]
    arm = max(abs(x - y) for x, y in zip(db, do))
    ctl = max(abs(x) for x in do)
    if gated:
        print('  CONTROL  |OF limited-corrected|  %-14.6e (needs >= %-10.3e) %s'
              % (ctl, need, 'OK' if ctl >= need else 'FAIL'))
        if ctl < need: bad = 1
        print('  ARM      %-12s delta vs OF  %-14.6e (bound %-9.3e) %s'
              % (tag, arm, bound, 'OK' if arm <= bound else 'FAIL'))
        if arm > bound: bad = 1
    else:
        print('  reported %-12s delta vs OF  %-14.6e  (own delta %.3e, too small to gate)'
              % (tag, arm, ctl))
sys.exit(bad)
CMPPY
rc=$?
[ "$rc" = 0 ] && echo "PASS" || { echo "FAIL"; exit 1; }

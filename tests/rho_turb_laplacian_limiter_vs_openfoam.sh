#!/usr/bin/env bash
# THE TURBULENCE CLOSURE TAKES THE CASE'S `limited <psi>` LAPLACIAN, like every other equation does.
#
# OpenFOAM's laplacian entry carries its OWN snGrad scheme (laplacianScheme.H:121-141 constructs
# tsnGradScheme_ from the entry's Istream; laplacianScheme.C:44-78 reads the scheme word off the
# laplacianSchemes entry), so `Gauss linear limited 0.33` caps the non-orthogonal correction of
# laplacian(DkEff,k) and laplacian(DepsilonEff,epsilon) exactly as it does laplacian(muEff,U).
#
# brae's CUDA arm did not. rhoTurbulenceHook.cu assigned kin.correctedLaplacian and nothing else, so
# KEpsilonInput::snGradLimitCoeff kept its 0.0 default (kEpsilon.cuh:146) and 0 means UNCAPPED at the
# only consumer (turbulence_transport.cu:172). The driver DID parse the cap and store it, at
# opt.co.snGradLimitCoeff (rhoSimpleFoamDriver.cu:322) -- a different member of a different struct from
# the one the closure reads, so the value was dead. Momentum, energy and pressure ran the capped
# correction while k and epsilon ran the uncapped one, under the case's own scheme name and with no
# notice. Found by audit 2026-09-15; fixed by one line in rhoTurbulenceHook.cu.
#
# NOTHING COULD SEE IT. tests/rho_limiters_vs_openfoam.sh is the only other `limited <psi>` gate and its
# fixture is laminar by its own header, so no turbulence laplacian was ever exercised under a cap; and
# every rho fixture in validation/ was `Gauss linear corrected` or `orthogonal`. This gate exists to
# close that hole, and validation/rhoLimKE exists because no fixture could host it.
#
#   ARM      validation/rhoLimKE, device arm (BRAE_RHOSIMPLEFOAM_MIRROR=cuda), t=1..N against real
#            rhoSimpleFoam on the same case. Gated on k and epsilon; U, T and p reported beside.
#   CONTROL  OpenFOAM's `limited 0.33` run must differ from its `corrected` run by >= CONTROL_RATIO x
#            the bound, or the limiter never clipped on this fixture and the arm is inert. This is the
#            check that keeps the gate honest: on an ORTHOGONAL mesh the correction is zero, so capping
#            it changes nothing and a green arm would mean nothing. rhoLimKE's mesh is tilted to
#            16.3 degrees maximum non-orthogonality for exactly this reason -- at rhoBoxSym's 4.18 the
#            control separation measured only 5.3x, which is not a gate.
#
# Every linear solver pinned to 1e-12 / relTol 0 / 2000; residualControl emptied; from 0/. Without that
# the arms drift apart on the linear solve (brae substitutes the pressure solver) and the scheme signal
# disappears underneath it: measured 2026-09-15 at the fixture's shipped tolerances, brae vs OpenFOAM on
# k was 2.99e-04 while the whole limiter effect was 1.59e-03 -- a separation of 5.3x, far too thin.
#
# Fail-proof: revert the one line in rhoTurbulenceHook.cu (kin.snGradLimitCoeff = opt.co.snGradLimitCoeff)
# and the ARM goes RED on k and epsilon while U, T and p stay green -- the signature of the defect.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDDIR="${BUILD:-$ROOT/build}"
BIN="$BUILDDIR/brae_rhoSimpleFoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-10}
# Bounds are ~2x the measured arm, which is ~2x brae's OWN baseline on this fixture. They are loose by
# this suite's standards and that is not the limiter's doing: with BOTH codes on `corrected` and no
# limiter anywhere, brae's device arm still reads k 1.58e-02 and epsilon 2.85e-01 from OpenFOAM here
# (measured 2026-09-15), because a 16.3 degree non-orthogonal diffuser is simply hard. The limiter is
# carried as well as everything else is. What makes this a gate is the CONTROL: k moves 1.41 and epsilon
# 23.6 when the cap is dropped, ~35x and ~24x ABOVE these bounds, so an arm that ignores the cap cannot
# sit under them. Measured arm with the fix: k 1.77e-02, epsilon 4.21e-01.
K_BOUND=${K_BOUND:-4e-2}; E_BOUND=${E_BOUND:-1.0}
CONTROL_RATIO=${CONTROL_RATIO:-20}

[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$ROOT/validation/rhoLimKE" ] || { echo "SKIP: validation/rhoLimKE missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
command -v blockMesh     > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# $1 dst   $2 laplacian scheme word(s), e.g. "limited 0.33" or "corrected"
stage()
{
    rm -rf "$1"; cp -r "$ROOT/validation/rhoLimKE" "$1"
    rm -rf "$1"/[1-9]* "$1"/0 "$1"/log.*
    cp -r "$1/0.orig" "$1/0"
    LAP="$2" python3 - "$1" "$N" <<'STAGE'
import os, re, sys
d, n = sys.argv[1], sys.argv[2]
lap = os.environ['LAP']

c = os.path.join(d, 'system/controlDict'); s = open(c).read()
s = re.sub(r'startTime \d+;', 'startTime 0;', s)
s = re.sub(r'endTime \d+;', 'endTime %s;' % n, s)
s = re.sub(r'writeInterval \d+;', 'writeInterval 1;', s)
s = re.sub(r'writePrecision \d+;', 'writePrecision 15;', s)
open(c, 'w').write(s)

f = os.path.join(d, 'system/fvSchemes'); s = open(f).read()
# Both blocks together: brae folds them into one flag (scheme_parse.cuh:695), so a fixture where they
# disagree would measure THAT defect too and confound this gate.
s = re.sub(r'laplacianSchemes \{[^}]*\}', 'laplacianSchemes { default Gauss linear %s; }' % lap, s)
s = re.sub(r'snGradSchemes\s+\{[^}]*\}',
           'snGradSchemes   { default %s; }' % (lap if lap != 'corrected' else 'corrected'), s)
open(f, 'w').write(s)

# Pin the linear solves so the arms meet at their own floor and the SCHEME is what differs.
v = os.path.join(d, 'system/fvSolution'); s = open(v).read()
s = re.sub(r'residualControl \{[^}]*\}', '', s)
s = re.sub(r'solvers \{.*?\n\}',
           'solvers {\n'
           # p is SYMMETRIC on this fixture (SIMPLE transonic no), so it takes DIC; DILU is asymmetric
           # and OpenFOAM refuses it outright (lduMatrixPreconditioner.C:97).
           '  p { solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0; maxIter 2000; }\n'
           '  "(U|h|e|k|epsilon)" { solver PBiCGStab; preconditioner DILU; tolerance 1e-12; relTol 0; maxIter 2000; }\n'
           '}', s, flags=re.S)
open(v, 'w').write(s)
STAGE
    ( cd "$1" && blockMesh > log.blockMesh 2>&1 ) || { echo "blockMesh failed in $1"; exit 1; }
}

field_diff()   # $1 dirA  $2 dirB  $3 time  $4 field -> max |a-b| on stdout
{
    python3 - "$1" "$2" "$3" "$4" <<'DIFF'
import re, sys, os, math
a, b, t, f = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
def rd(p):
    if not os.path.exists(p): return None
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(\w+)>\s*\n(\d+)\s*\n\(\n(.*?)\n\)\n;', s, re.S)
    if not m:
        u = re.search(r'internalField\s+uniform\s+(.*?);', s, re.S)
        return ('uniform', u.group(1).strip()) if u else None
    body = m.group(3).split('\n')
    if m.group(1) == 'vector':
        return ('v', [tuple(float(x) for x in l.strip()[1:-1].split()) for l in body])
    return ('s', [float(x) for x in body])
A, B = rd(os.path.join(a, t, f)), rd(os.path.join(b, t, f))
if A is None or B is None or A[0] == 'uniform' or B[0] == 'uniform':
    print('nan'); sys.exit(0)
if A[0] == 'v':
    print('%.6e' % max(math.dist(x, y) for x, y in zip(A[1], B[1])))
else:
    print('%.6e' % max(abs(x - y) for x, y in zip(A[1], B[1])))
DIFF
}

worst()   # $1 dirA  $2 dirB  $3 field -> max over t=1..N
{
    local m=0 d
    for t in $(seq 1 "$N"); do
        d=$(field_diff "$1" "$2" "$t" "$3")
        [ "$d" = nan ] && continue
        m=$(python3 -c "print(max($m, $d))")
    done
    echo "$m"
}

echo "rho turbulence laplacian limiter: does \`limited <psi>\` reach k and epsilon on the device arm?"

stage "$W/of_lim"  "limited 0.33"
stage "$W/of_corr" "corrected"
stage "$W/br_lim"  "limited 0.33"

( cd "$W/of_lim"  && rhoSimpleFoam > log.run 2>&1 ) || { echo "  OpenFOAM (limited) failed"; exit 1; }
( cd "$W/of_corr" && rhoSimpleFoam > log.run 2>&1 ) || { echo "  OpenFOAM (corrected) failed"; exit 1; }
( cd "$W/br_lim"  && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BIN" -case "$W/br_lim" > log.run 2>&1 ) \
    || { echo "  brae (device, limited) failed:"; tail -5 "$W/br_lim/log.run"; exit 1; }

# CONTROL FIRST. If the limiter never clipped on this fixture the arm below is inert, and a green arm
# would mean nothing at all.
ck=$(worst "$W/of_lim" "$W/of_corr" k)
ce=$(worst "$W/of_lim" "$W/of_corr" epsilon)
need_k=$(python3 -c "print($K_BOUND * $CONTROL_RATIO)")
need_e=$(python3 -c "print($E_BOUND * $CONTROL_RATIO)")
printf '  CONTROL, OpenFOAM limited-vs-corrected over t=1..%s\n' "$N"
say "  k       $ck  (needs >= $need_k)"       "$(python3 -c "print('OK' if $ck >= $need_k else 'FAIL')")"
say "  epsilon $ce  (needs >= $need_e)"       "$(python3 -c "print('OK' if $ce >= $need_e else 'FAIL')")"

printf '  ARM, brae device vs real OpenFOAM, both `limited 0.33`, worst over t=1..%s\n' "$N"
for spec in "k $K_BOUND" "epsilon $E_BOUND"; do
    set -- $spec
    d=$(worst "$W/br_lim" "$W/of_lim" "$1")
    say "  $1  $d  (bound $2)" "$(python3 -c "print('OK' if $d <= $2 else 'FAIL')")"
done
printf '  reported, not gated (the momentum/energy/pressure arms have always carried the cap):\n'
for f in U T p; do
    printf '    %-8s %s\n' "$f" "$(worst "$W/br_lim" "$W/of_lim" "$f")"
done

[ "$fail" = 0 ] && echo "PASS" || echo "FAIL"
exit "$fail"

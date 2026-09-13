#!/usr/bin/env bash
# The FP-2 preconditioner policy on the rhoSimpleFoam mirror's CUDA arm, against real OpenFOAM.
#
# A `solver PBiCGStab; preconditioner DILU;` entry on k, omega (or epsilon) and the energy field used to
# run brae's level-scheduled DILU, bit-identical to OpenFOAM's and sequential by construction: on
# aerofoilNACA0012 (121 levels) the k and omega solves were 862 of the turbulence phase's 1081 kernel
# launches per iteration and 74% of its GPU time, and the single-block walk that halved it is still 1.7
# of the phase's 2.25 GPU ms. Since 2026-09-12 such an entry takes the truncated Neumann series the pair
# already takes on a GAMG entry, wherever fvMatrix::relax bounds it (degree d = ceil(ln 0.1 / ln alpha)),
# announced per field; BRAE_DILU_KE=1 and BRAE_DILU_HE=1 keep DILU. Same converged solution, a different
# iterate at a loose relTol -- the momentum sweep's trade.
#
#   FIXTURE     the aerofoilNACA0012 tutorial as it ships (PBiCGStab+DILU on U, k, omega, e; relaxation
#               0.7 on e, k and omega, so degree 6), functions stripped, residualControl removed, 30
#               iterations at the tutorial's OWN tolerances (the loose relTol is the point: at 1e-14 the
#               preconditioner cannot be seen and rho_tutorials_vs_openfoam already holds that).
#   ARM SAID    the default run announces the series on k, omega and e; the honoured run
#               (BRAE_DILU_KE=1 BRAE_DILU_HE=1) announces nothing on them.
#   ARM 1       the honoured run's k, omega and e initial residuals at iteration 30 sit within
#               [0.9, 1.1] of OpenFOAM's (measured 1.00, 1.00, 1.01: both codes run OpenFOAM's own
#               preconditioner there) and the default's within [0.9, 1.2] (measured 1.07, 1.04,
#               1.03: the series stops the solve elsewhere under relTol 0.1, and that is all it
#               moves). Both arms are deterministic since D-1, so the bounds are the measurement
#               rounded outward to one digit.
#   ARM 2       bound(): the default arm floors no more cells than the honoured one over the 30
#               iterations (the series must not cost health; brae prints OpenFOAM's `bounding`
#               line). On this fixture neither arm floors a cell, so this is a guard, not a
#               discriminator.
#   ARM 3       the default arm's turbulence phase (BRAE_PHASE_TIME=1) is under 0.8x the honoured
#               one's, which is the only reason the policy exists (measured 1.9 against 3.4 ms/it,
#               0.56x).
#   CONTROL     the honoured arm: with the hatches set the policy must leave no trace -- no notice,
#               the DILU walk built -- and a silently unapplied policy fails the SAID arms the other
#               way round.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-30}
[ -x "$BRAE" ] || { echo "SKIP: $BRAE not built"; exit 77; }
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true   # before set -u
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
SRC="${FOAM_TUTORIALS:-}/compressible/rhoSimpleFoam/aerofoilNACA0012"
[ -d "$SRC" ] || { echo "SKIP: tutorial $SRC not found"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-84s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

cp -r "$SRC" "$W/mesh"; ( cd "$W/mesh" && rm -rf 0 [1-9]* dynamicCode log.* && ./Allrun.pre > log.pre 2>&1 ) || { echo "FAIL: Allrun.pre"; exit 1; }
[ -f "$W/mesh/constant/polyMesh/owner" ] || { echo "FAIL: no mesh"; exit 1; }
stage() {
    rm -rf "$1"; cp -r "$W/mesh" "$1"; rm -rf "$1"/0 "$1"/[1-9]* "$1"/log.*; cp -r "$1/0.orig" "$1/0"
    N="$N" python3 - "$1" <<'PY'
import os, re, sys
d, n = sys.argv[1], os.environ['N']
p = os.path.join(d, 'system/controlDict'); s = open(p).read()
s = re.sub(r'\nfunctions\s*\{.*', '\n', s, flags=re.S)
for k, v in (('startFrom', 'latestTime'), ('endTime', n), ('writeInterval', n), ('writeFormat', 'ascii'), ('stopAt', 'endTime')):
    if re.search(r'^%s\s+' % k, s, re.M): s = re.sub(r'^%s\s+.*' % k, '%-15s %s;' % (k, v), s, flags=re.M)
    else: s += '\n%-15s %s;\n' % (k, v)
open(p, 'w').write(s + '\n')
p = os.path.join(d, 'system/fvSolution'); s = open(p).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
open(p, 'w').write(s)
PY
}
stage "$W/of"; ( cd "$W/of" && rhoSimpleFoam > log.of 2>&1 ) || { echo "FAIL: OpenFOAM did not run"; tail -5 "$W/of/log.of"; exit 1; }
run() { ( cd "$1" && env "${@:2}" BRAE_PHASE_TIME=1 BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$1" > "$1/log.brae" 2>&1 ); }
stage "$W/def"; run "$W/def" X=1                               || { echo "FAIL: the default run crashed"; tail -8 "$W/def/log.brae"; exit 1; }
stage "$W/hon"; run "$W/hon" BRAE_DILU_KE=1 BRAE_DILU_HE=1     || { echo "FAIL: the honoured run crashed"; tail -8 "$W/hon/log.brae"; exit 1; }

for f in k omega e; do
    grep -q "solvers/$f preconditioner: case asks 'DILU', brae preconditions with a degree-6 truncated Neumann series" "$W/def/log.brae" \
        && say "SAID  the default run announces the degree-6 series on $f" ok \
        || say "SAID  the default run announces the degree-6 series on $f" FAIL
    grep -q "solvers/$f preconditioner" "$W/hon/log.brae" \
        && say "SAID  the honoured run announces nothing on $f's preconditioner" FAIL \
        || say "SAID  the honoured run announces nothing on $f's preconditioner" ok
done
grep -q "DILU: " "$W/hon/log.brae" && say "SAID  the honoured run built the DILU walk" ok || say "SAID  the honoured run built the DILU walk" FAIL
grep -q "DILU: " "$W/def/log.brae" && say "SAID  the default run built no DILU walk (nothing asks for one)" FAIL || say "SAID  the default run built no DILU walk (nothing asks for one)" ok

python3 - "$W" "$N" <<'PY' || fail=1
import re, sys
W, IT = sys.argv[1], int(sys.argv[2])
of, it = {}, 0
for line in open(W + '/of/log.of', errors='latin-1'):
    if line.startswith('Time = '):
        it = int(float(line.split('=')[1])); continue
    m = re.match(r'\S+:\s+Solving for (\w+), Initial residual = ([\d.eE+-]+)', line)
    if m and it: of.setdefault(it, {}).setdefault(m.group(1), float(m.group(2)))
def brae(p):
    d = {}
    for line in open(p, errors='latin-1'):
        m = re.match(r'Time = (\d+)\s+(.*)$', line)
        if m: d[int(m.group(1))] = {f: float(v) for f, v in re.findall(r'(\w+) ([\d.eE+-]+)', m.group(2))}
    return d
hon, dft = brae(W + '/hon/log.brae'), brae(W + '/def/log.brae')
if IT not in of or IT not in hon or IT not in dft:
    print("  the runs did not all reach iteration %d" % IT); sys.exit(1)
bad = 0
# honoured: [0.5, 2.0]; default: measured on the first green run, bounds are the measured values rounded
# outward to one digit, never loosened afterwards.
HON = (0.9, 1.1)
DEF = {'k': (0.9, 1.2), 'omega': (0.9, 1.2), 'e': (0.9, 1.2)}
for f in ('k', 'omega', 'e'):
    o, h, d = of[IT].get(f), hon[IT].get(f), dft[IT].get(f)
    if o is None or h is None or d is None: print("  %s missing" % f); bad = 1; continue
    rh, rd = h / o, d / o
    okh = HON[0] <= rh <= HON[1]; okd = DEF[f][0] <= rd <= DEF[f][1]
    print("  ARM 1  %-6s it %d  OpenFOAM %.4e  honoured %.4e (%.2fx, %s)  default %.4e (%.2fx, %s)"
          % (f, IT, o, h, rh, 'ok' if okh else 'FAIL', d, rd, 'ok' if okd else 'FAIL'))
    bad |= (not okh) | (not okd)
# ARM 2: bounding
def bounds(p):
    n = 0
    for line in open(p, errors='latin-1'):
        if line.startswith('bounding '): n += 1
    return n
bh, bd = bounds(W + '/hon/log.brae'), bounds(W + '/def/log.brae')
okb = bd <= bh
print("  ARM 2  bounding lines over %d iterations: honoured %d, default %d   %s" % (IT, bh, bd, 'ok' if okb else 'FAIL'))
bad |= (not okb)
# ARM 3: the turbulence phase
def turb(p):
    for line in open(p, errors='latin-1'):
        m = re.search(r'turbulence ([\d.]+) s \(([\d.]+) ms/it\)', line)
        if m: return float(m.group(2))
    return None
th, td = turb(W + '/hon/log.brae'), turb(W + '/def/log.brae')
okt = th is not None and td is not None and td < 0.8 * th
print("  ARM 3  turbulence phase ms/it: honoured (DILU) %s, default (series) %s (%.2fx, bound 0.8x)   %s"
      % (th, td, (td / th) if (th and td) else float('nan'), 'ok' if okt else 'FAIL'))
bad |= (not okt)
sys.exit(1 if bad else 0)
PY
echo
[ "$fail" = 0 ] && echo "PASS" || echo "FAIL"
exit "$fail"

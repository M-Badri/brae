#!/usr/bin/env bash
# The rhoSimpleFoam mirror's CUDA arm is bit-reproducible run to run -- and the one place it was not.
#
# D-1 (bench/rhoSimpleFoam/FASTPATH.md, 2026-09-12). squareBendLiq and gasMixing/injectorPipe wrote
# different low bits between two runs of ONE binary: identical through the epsilon solve of iteration 4,
# then `epsOut` apart, then everything downstream (BRAE_STAGE_DUMP_DIR, every stage of iteration 4
# compared byte for byte). Between the solve and epsOut sits Foam::bound, whose face average was gathered
# with atomicAdd scatters (kEpsilon.cu boundGatherKernel / boundGatherBndKernel): the order a cell's faces
# landed in was the hardware's, so the value a clamped cell took differed in its last bits per run. The
# gather now forms each cell's sums in OpenFOAM's own face order (boundGatherCellKernel), and this gate
# holds the property CONTRIBUTING.md promises.
#
#   ARM 1    squareBendLiq, 20 iterations, twice: every `Time =` residual line identical, and every
#            written field at 20 (U p T k epsilon nut) an identical file.
#   ARM 2    gasMixing/injectorPipe, the same.
#   ARM 3    squareBend, the same (it was reproducible before; it must stay so).
#   CONTROL  squareBendLiq under BRAE_BOUND_SCATTER=1 -- the atomic scatter restored -- twice: the two
#            runs must DIFFER, in the residual lines or the written fields. This is a race, so the
#            control is probabilistic; measured 6 of 6 pairs apart within 20 iterations before the fix
#            (first visible at the fourth printed digit by iteration 7-11), and the arm prints where.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-20}
[ -x "$BRAE" ] || { echo "SKIP: $BRAE not built"; exit 77; }
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true   # before set -u: the bashrc reads unset variables
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: OpenFOAM's meshers not on PATH"; exit 77; }
TUT="${FOAM_TUTORIALS:-}/compressible/rhoSimpleFoam"
[ -d "$TUT/squareBendLiq" ] || { echo "SKIP: tutorials not found"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-80s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

mesh() {   # mesh <tutorial path> <name>  (the tutorial gate's recipe)
    local src="$TUT/$1" d="$W/mesh_$2"
    rm -rf "$d"; cp -r "$src" "$d" || return 1
    if [ -x "$d/Allrun.pre" ]; then ( cd "$d" && ./Allrun.pre > log.pre 2>&1 )
    else ( cd "$d" && cp -r 0.orig 0 && blockMesh > log.blockMesh 2>&1 ); fi
    [ -f "$d/system/topoSetDict" ] && ( cd "$d" && topoSet > log.topoSet 2>&1 )
    if [ -d "$d/processor0/constant/polyMesh" ]; then
        ( cd "$d" && reconstructParMesh -constant > log.reconstructParMesh 2>&1 && rm -rf processor* )
    fi
    [ -f "$d/constant/polyMesh/owner" ]
}
stage() {   # stage <meshed> <dest>
    rm -rf "$2"; cp -r "$1" "$2"
    rm -rf "$2"/0 "$2"/[1-9]* "$2"/processor* "$2"/log.* "$2"/postProcessing "$2"/dynamicCode
    cp -r "$2/0.orig" "$2/0"
    N="$N" python3 - "$2" <<'PYEOF'
import os, re, sys
d, n = sys.argv[1], os.environ['N']
p = os.path.join(d, 'system/controlDict'); s = open(p).read()
s = re.sub(r'\nfunctions\s*\{.*', '\n', s, flags=re.S)
for k, v in (('startFrom', 'latestTime'), ('endTime', n), ('writeInterval', n), ('writeFormat', 'ascii'),
             ('writePrecision', '15'), ('writeCompression', 'off'), ('stopAt', 'endTime')):
    if re.search(r'^%s\s+' % k, s, re.M): s = re.sub(r'^%s\s+.*' % k, '%-15s %s;' % (k, v), s, flags=re.M)
    else: s += '\n%-15s %s;\n' % (k, v)
open(p, 'w').write(s + '\n')
p = os.path.join(d, 'system/fvSolution'); s = open(p).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
open(p, 'w').write(s)
PYEOF
}
same() {   # same <dirA> <dirB> -> 0 if residual lines and written fields are identical
    local a=$1 b=$2 ok=0
    diff <(grep '^Time = ' "$a/log.brae") <(grep '^Time = ' "$b/log.brae") > /dev/null || ok=1
    for f in U p T k epsilon omega nut; do
        [ -f "$a/$N/$f" ] || continue
        cmp -s "$a/$N/$f" "$b/$N/$f" || ok=1
    done
    return $ok
}
firstDiff() { diff <(grep '^Time = ' "$1/log.brae") <(grep '^Time = ' "$2/log.brae") | grep -m1 '^<' | cut -c1-40; }
run() { ( cd "$1" && env "${@:2}" BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$1" > "$1/log.brae" 2>&1 ); }

arm() {   # arm <label> <tutorial path> <name> [VAR=value ...]
    local label=$1 path=$2 name=$3; shift 3
    if ! mesh "$path" "$name"; then say "$label: $name meshed" FAIL; return; fi
    stage "$W/mesh_$name" "$W/${name}_a"; run "$W/${name}_a" "$@" || { say "$label: $name run a finished" FAIL; return; }
    stage "$W/mesh_$name" "$W/${name}_b"; run "$W/${name}_b" "$@" || { say "$label: $name run b finished" FAIL; return; }
    [ "$(grep -c '^Time = ' "$W/${name}_a/log.brae")" = "$N" ] || { say "$label: $name completed $N iterations" FAIL; return; }
    if same "$W/${name}_a" "$W/${name}_b"; then
        say "$label  $name: two runs, residual lines and written fields byte-identical" ok
    else
        say "$label  $name: two runs DIFFER (first residual difference: $(firstDiff "$W/${name}_a" "$W/${name}_b"))" FAIL
    fi
}
arm "ARM 1" squareBendLiq squareBendLiq
arm "ARM 2" gasMixing/injectorPipe injectorPipe
arm "ARM 3" squareBend squareBend
# CONTROL: the scatter restored must be seen to drift, or this gate could not detect the defect it closes.
stage "$W/mesh_squareBendLiq" "$W/ctl_a"; run "$W/ctl_a" BRAE_BOUND_SCATTER=1
stage "$W/mesh_squareBendLiq" "$W/ctl_b"; run "$W/ctl_b" BRAE_BOUND_SCATTER=1
if same "$W/ctl_a" "$W/ctl_b"; then
    say "CONTROL  squareBendLiq under BRAE_BOUND_SCATTER=1: two runs identical -- the race did not show (probabilistic)" FAIL
else
    say "CONTROL  squareBendLiq under BRAE_BOUND_SCATTER=1: two runs differ ($(firstDiff "$W/ctl_a" "$W/ctl_b"))" ok
fi
echo
[ "$fail" = 0 ] && echo "PASS" || echo "FAIL"
exit "$fail"

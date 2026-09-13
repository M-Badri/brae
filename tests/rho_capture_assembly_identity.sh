#!/usr/bin/env bash
# The captured momentum assembly computes the same bits as the direct one.
#
# FP-10. squareBend issues about 1,000 kernel launches per iteration at ~4 ms of host API time, and the
# GPU idles while the host issues them. BRAE_CAPTURE_ASSEMBLY=1 captures the momentum assembly once and
# replays it as a CUDA graph, which costs one launch instead of 38.
#
# WHAT MAKES IT DELICATE, and why this gate exists: a graph bakes in the ADDRESSES its kernels use. The
# assembly's temporaries used to be stack objects taking blocks from the device pool on every call, and
# the matrix it fills was constructed fresh each iteration -- so a replay wrote into memory the pool had
# handed to somebody else. That is not theory: with pool temporaries the first replay dies with an
# illegal memory access, and with a fresh matrix the memory checker catches `axpyKernel` reading 0x100.
# Both are fixed by the workspaces (rhoUEqn.cu, device_divdevreff.cu) and the persistent matrix, and
# this gate is what holds them fixed. Anyone who adds a pool-allocated temporary to the assembly, or
# reinstates a std::move that swaps a buffer's pointer, breaks this and nothing else would say so.
#
#   ARM 1     squareBend (112k, transonic, SIMPLEC): 30 iterations captured against 30 direct -- every
#             residual line identical and every written field byte-identical.
#   ARM 2     the aerofoil (16k, kOmegaSST, cellLimited grad(U)): the same, on a different scheme set.
#   ARM 3     the captured run announces the capture, and the direct one does not (else ARM 1 compares
#             two direct runs and passes for the wrong reason).
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-30}
[ -x "$BRAE" ] || { echo "SKIP: $BRAE not built"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
[ -n "${FOAM_TUTORIALS:-}" ] || { echo "SKIP: no OpenFOAM tutorials"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-74s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

prep() {   # prep <mesh cache> <tutorial subpath>
    local MC=$1 T=$2
    [ -f "$MC/constant/polyMesh/owner" ] && return 0
    rm -rf "$MC"; cp -r "$FOAM_TUTORIALS/$T" "$MC" || return 1
    ( cd "$MC" && ./Allrun.pre > log.pre 2>&1 ) || ( cd "$MC" && blockMesh > log.bm 2>&1 ) || return 1
    [ -f "$MC/constant/polyMesh/owner" ]
}
stage() {   # stage <dest> <mesh cache>
    rm -rf "$1"; cp -r "$2" "$1"; rm -rf "$1"/0 "$1"/[1-9]* "$1"/log.* "$1"/postProcessing "$1"/dynamicCode
    [ -d "$1/0.orig" ] && cp -r "$1/0.orig" "$1/0"
    N="$N" python3 - "$1" <<'PY'
import os, re, sys
d = sys.argv[1]; n = os.environ['N']
p = d + '/system/controlDict'; s = open(p).read()
for k, v in (('startFrom','latestTime'), ('endTime',n), ('writeInterval',n), ('stopAt','endTime'), ('writeFormat','ascii')):
    s = re.sub(r'^%s\s+.*' % k, '%-15s %s;' % (k,v), s, flags=re.M) if re.search(r'^%s\s+'%k, s, re.M) else s + '\n%-15s %s;\n' % (k,v)
s = re.sub(r'\nfunctions\s*\{.*', '\n', s, flags=re.S); open(p,'w').write(s+'\n')
p = d + '/system/fvSolution'; s = open(p).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s); open(p,'w').write(s)
PY
}
pair() {   # pair <label> <mesh cache> <tutorial>
    local label=$1 mc=$2
    prep "$mc" "$3" || { echo "  SKIP $label: its mesh did not build here"; return 0; }
    for arm in cap dir; do
        stage "$W/${label}_$arm" "$mc"
        local e=""; [ "$arm" = cap ] && e="BRAE_CAPTURE_ASSEMBLY=1"
        ( cd "$W/${label}_$arm" && env ${e:-X=1} BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$W/${label}_$arm" > log 2>&1 ) \
            || { echo "  FAIL $label/$arm: the run crashed"; tail -3 "$W/${label}_$arm/log"; fail=1; return 0; }
        [ "$(grep -c '^Time = ' "$W/${label}_$arm/log")" = "$N" ] \
            || { say "$label/$arm reached $N iterations" FAIL; return 0; }
    done
    if diff <(grep '^Time = ' "$W/${label}_cap/log") <(grep '^Time = ' "$W/${label}_dir/log") > /dev/null
    then say "$label: captured and direct, every residual line identical" ok
    else say "$label: captured and direct, every residual line identical" FAIL
         diff <(grep '^Time = ' "$W/${label}_cap/log") <(grep '^Time = ' "$W/${label}_dir/log") | head -3; fi
    local d=0 n=0 f b
    for f in "$W/${label}_dir/$N"/*; do
        [ -f "$f" ] || continue
        b=$(basename "$f"); n=$((n+1))
        cmp -s "$f" "$W/${label}_cap/$N/$b" || { echo "     differs: $b"; d=$((d+1)); }
    done
    [ "$n" -gt 0 ] && [ "$d" = 0 ] && say "$label: all $n written fields byte-identical" ok \
                                  || say "$label: $d of $n written fields differ" FAIL
    grep -q "UEqn assembly: captured" "$W/${label}_cap/log" \
        && say "$label: the captured arm says it captured" ok \
        || say "$label: the captured arm says it captured" FAIL
    grep -q "UEqn assembly: captured" "$W/${label}_dir/log" \
        && say "$label: the direct arm does NOT capture (control)" FAIL \
        || say "$label: the direct arm does NOT capture (control)" ok
}

pair squareBend "$W/mesh_sb" compressible/rhoSimpleFoam/squareBend
pair aerofoil   "$W/mesh_af" compressible/rhoSimpleFoam/aerofoilNACA0012

echo
[ "$fail" = 0 ] && echo "PASS" || echo "FAIL"
exit "$fail"

#!/usr/bin/env bash
# A VISUALISATION run, which is not a benchmark run and must not be confused with one.
#
# The benchmark harness runs a fixed 100 iterations with residualControl removed and then DELETES the
# written time directories (run_matrix.sh's run_arm does `rm -rf "$AD"/[1-9]*`). That is right for timing
# and useless for pictures: 100 iterations of a steady SIMPLE solver is an early transient, and the data
# is gone afterwards anyway. This script runs long, writes often, and keeps everything.
#
# Usage:  ./run_visual.sh [case] [targetCells] [iterations] [writeInterval] [outDir]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
CASE="${1:-aerofoilNACA0012}"
TARGET="${2:-1000000}"
ITERS="${3:-3000}"
WI="${4:-10}"
OUT="${5:-$HOME/visual_${CASE}_${TARGET}}"
BRAE="${BRAE:-$HERE/../../build/brae_rhoSimpleFoam}"
OFBASHRC="${OFBASHRC:-$(ls /usr/lib/openfoam/openfoam*/etc/bashrc 2>/dev/null | head -1)}"
set +u; source "$OFBASHRC" > /dev/null 2>&1; set -u

SRC="$FOAM_TUTORIALS/compressible/rhoSimpleFoam/$CASE"
[ -d "$SRC" ] || { echo "no tutorial $CASE"; exit 2; }
rm -rf "$OUT"; cp -r "$SRC" "$OUT"
rm -rf "$OUT"/0 "$OUT"/[1-9]* "$OUT"/processor* "$OUT"/log.* "$OUT"/dynamicCode
[ -d "$OUT/0.orig" ] && cp -r "$OUT/0.orig" "$OUT/0"

python3 - "$OUT" "$CASE" "$TARGET" "$ITERS" "$WI" <<'PY'
import re, sys, os
d, case, target, iters, wi = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
# the aerofoil is extruded: it scales IN PLANE only, so cells go as factor^2
knobs = {'aerofoilNACA0012': (['zCells','xUCells','xMCells','xDCells'], 2, 16000),
         'squareBend':       (['LITERAL'], 3, 112000),
         'squareBendLiq':    (['nxin','nxout','nxbend','ny','nz'], 3, 112000)}
f = d + '/system/blockMeshDict'
if case in knobs and os.path.exists(f):
    names, expo, native = knobs[case]
    factor = (int(target)/native) ** (1.0/expo)
    factor = float(round(factor)) if factor >= 1.5 else factor
    s = open(f).read()
    if names == ['LITERAL']:
        s = re.sub(r'\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)(\s*(?:simple|edge)Grading)',
                   lambda m: '(%d %d %d)%s' % (max(1,round(int(m[1])*factor)), max(1,round(int(m[2])*factor)),
                                               max(1,round(int(m[3])*factor)), m[4]), s)
    else:
        for k in names:
            s = re.sub(r'(^\s*%s\s+)(\d+)(\s*;)' % k,
                       lambda m: '%s%d%s' % (m[1], max(1, round(int(m[2])*factor)), m[3]), s, count=1, flags=re.M)
    open(f, 'w').write(s)
# KEEP the writes: write often, purge nothing. This is the whole point of the script.
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'\nfunctions\s*\{.*', '\n', s, flags=re.S)
for k, v in (('startFrom','latestTime'), ('stopAt','endTime'), ('endTime',iters),
             ('writeControl','timeStep'), ('writeInterval',wi), ('purgeWrite','0'),
             ('writeFormat','binary'), ('writeCompression','off')):
    s = re.sub(r'^%s\s+.*' % k, '%-16s %s;' % (k, v), s, flags=re.M) if re.search(r'^%s\s+' % k, s, re.M) \
        else s + '\n%-16s %s;\n' % (k, v)
open(c, 'w').write(s + '\n')
PY

if [ -x "$OUT/Allrun.pre" ]; then ( cd "$OUT" && ./Allrun.pre > log.pre 2>&1 )
else ( cd "$OUT" && blockMesh > log.blockMesh 2>&1 ); fi
if [ -d "$OUT/processor0/constant/polyMesh" ]; then
    ( cd "$OUT" && reconstructParMesh -constant > log.reconstructParMesh 2>&1 ); rm -rf "$OUT"/processor*
fi
[ -d "$OUT/0.orig" ] && { rm -rf "$OUT/0"; cp -r "$OUT/0.orig" "$OUT/0"; }
rm -rf "$OUT"/[1-9]*
NC=$(grep -aoE 'nCells:?[[:space:]]*[0-9]+' "$OUT/constant/polyMesh/owner" | grep -oE '[0-9]+' | head -1)
echo "case=$CASE cells=$NC iterations=$ITERS writeInterval=$WI -> ~$((ITERS/WI)) frames"
echo "out=$OUT"
a=$(date +%s)
( cd "$OUT" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$OUT" > log.run 2>&1 )
rc=$?; b=$(date +%s)
echo "rc=$rc wall=$((b-a))s  iterations done=$(grep -c '^Time = ' "$OUT/log.run")  frames=$(ls -d "$OUT"/[1-9]* 2>/dev/null | wc -l)"
grep '^Time = ' "$OUT/log.run" | tail -1

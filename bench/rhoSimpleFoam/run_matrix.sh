#!/usr/bin/env bash
# ============================================================================
#  brae rhoSimpleFoam: the FULL benchmark matrix -- every tutorial, several mesh sizes, several arms,
#  on whatever machine it is run on. Companion to run_benchmark.sh, which does squareBend only.
#
#  It emits ONE CSV (results.csv) carrying the host's own description alongside every row, so a run on
#  a GB10 and a run on an H100 concatenate into one table and one chart without editing.
#
#  ARMS
#    brae    brae's CUDA arm, whole solver on the GPU, one device
#    of      OpenFOAM rhoSimpleFoam on $CORES cores (mpirun + decomposePar; serial when CORES=1)
#    amgx    OpenFOAM with the pressure routed to AMGX      (needs libamgxFoam.so + AMGXLIB)
#    petsc   OpenFOAM with the pressure routed to PETSc GPU (needs libpetscFoam.so + PETSCLIB)
#    spuma   spuma's own rhoSimpleFoam on the GPU (needs a built spuma tree; see spuma_run.sh)
#
#  MESH SIZES. TARGETS is a list of cell counts; "native" is the tutorial's own mesh. Each case is
#  scaled through the knobs its blockMeshDict actually exposes (see CASE_SCALE below) -- the tutorials
#  do not share a dict style, so the knob names are per case and the scaling exponent says whether the
#  case grows in three directions or two. A target is approximate: the knobs are integers.
#
#  RULES, the same ones the rest of bench/ uses: fixed iteration count, solver wall only (meshing,
#  decomposePar and reconstructPar excluded on every arm), `functions` stripped and residualControl
#  removed on both sides, and a row is only reported as a time if the run REACHED the last iteration --
#  anything else is written out with its status so a diverged arm can never be read as a fast one.
#
#  Usage:  ./run_matrix.sh
#  Env:
#     BRAE      brae binary            (default ../../build/brae_rhoSimpleFoam)
#     CASES     which cases            (default: all six)
#     TARGETS   cell counts            (default "native 1000000")   e.g. "native 1000000 10000000"
#     ARMS      which arms             (default "brae of")
#     CORES     cores for the of arm   (default: nproc)
#     ITERS     iterations timed       (default 100)
#     OUT       output directory       (default ./results_matrix)
#     AMGXLIB   AMGX lib dir           (default /home/ghost/opt/amgx/lib)
#     PETSCLIB  PETSc lib dir          (default /home/ghost/space/amgx/petsc/arch-sysmpi/lib)
#     SNAP      1 (default) rounds the scale factor to a whole number so the scaled mesh is
#               geometrically similar to the tutorial's; 0 hits the target count more exactly
#     WORK      scratch                (default /tmp/brae_matrix)
# ============================================================================
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
BRAE="${BRAE:-$HERE/../../build/brae_rhoSimpleFoam}"
CASES="${CASES:-squareBend squareBendLiq squareBendLiqNoNewtonian angledDuctExplicitFixedCoeff aerofoilNACA0012 injectorPipe}"
TARGETS="${TARGETS:-native 1000000}"
ARMS="${ARMS:-brae of}"          # brae of amgx petsc spuma
CORES="${CORES:-$(nproc)}"
ITERS="${ITERS:-100}"
OUT="${OUT:-$HERE/results_matrix}"
WORK="${WORK:-/tmp/brae_matrix}"
export SNAP="${SNAP:-1}"   # read by the scaler below
AMGXLIB="${AMGXLIB:-/home/ghost/opt/amgx/lib}"
PETSCLIB="${PETSCLIB:-/home/ghost/space/amgx/petsc/arch-sysmpi/lib}"
OFBASHRC="${OFBASHRC:-$(ls /usr/lib/openfoam/openfoam*/etc/bashrc /opt/openfoam*/etc/bashrc 2>/dev/null | head -1)}"

set +u; source "$OFBASHRC" > /dev/null 2>&1; set -u
command -v blockMesh > /dev/null || { echo "ERROR: OpenFOAM not sourced (set OFBASHRC=)"; exit 1; }
[ -x "$BRAE" ] || { echo "ERROR: brae binary not at '$BRAE' (set BRAE=)"; exit 1; }

# ---------------------------------------------------------------------------
# Per case: tutorial path under $FOAM_TUTORIALS/compressible/rhoSimpleFoam, the blockMeshDict knobs to
# scale, and the exponent (3 = the mesh grows in three directions, 2 = in plane only, an extruded case).
# "LITERAL" means the dict writes its counts as `(nx ny nz) simpleGrading` rather than through named
# entries, and every such triple is scaled.
# ---------------------------------------------------------------------------
case_path(){ case "$1" in injectorPipe) echo "gasMixing/injectorPipe";; *) echo "$1";; esac; }
case_knobs(){
    case "$1" in
        squareBend|injectorPipe)              echo "LITERAL" ;;
        squareBendLiq|squareBendLiqNoNewtonian) echo "nxin nxout nxbend ny nz" ;;
        angledDuctExplicitFixedCoeff)         echo "/cellWidth" ;;   # a target cell SIZE: it scales inversely
        aerofoilNACA0012)                     echo "zCells xUCells xMCells xDCells" ;;
        *)                                    echo "" ;;
    esac
}
case_exp(){ case "$1" in aerofoilNACA0012) echo 2 ;; *) echo 3 ;; esac; }
# The decomposition each case's own tutorial uses; scotch where the geometry is not a box.
case_decomp(){
    case "$1" in
        aerofoilNACA0012)             echo "hierarchical" ;;
        angledDuctExplicitFixedCoeff) echo "scotch" ;;
        *)                            echo "hierarchical" ;;
    esac
}

hostdesc(){
    local gpu cpu
    gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"; gpu="${gpu:-none}"
    cpu="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')"
    [ -n "$cpu" ] || cpu="$(uname -m)"
    echo "$(hostname)|$gpu|$cpu|$(nproc)"
}

# Stage $case at $target cells into $dir. Echoes the realised nCells, or "FAILED".
stage(){
    local c="$1" target="$2" d="$3"
    local src="$FOAM_TUTORIALS/compressible/rhoSimpleFoam/$(case_path "$c")"
    [ -d "$src" ] || { echo FAILED; return; }
    rm -rf "$d"; cp -r "$src" "$d"; rm -rf "$d"/0 "$d"/[1-9]* "$d"/processor* "$d"/log.* "$d"/dynamicCode
    [ -d "$d/0.orig" ] && cp -r "$d/0.orig" "$d/0"
    python3 - "$d" "$c" "$target" "$ITERS" "$(case_knobs "$c")" "$(case_exp "$c")" <<'PY'
import re, sys, os
d, case, target, iters, knobs, expo = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6])
# controlDict: exactly ITERS iterations, no functionObjects (several need surfaces Allrun.pre builds)
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'\nfunctions\s*\{.*', '\n', s, flags=re.S)
for k, v in (('startFrom','latestTime'), ('stopAt','endTime'), ('endTime',iters), ('writeInterval',iters)):
    s = re.sub(r'^%s\s+.*' % k, '%-15s %s;' % (k, v), s, flags=re.M) if re.search(r'^%s\s+' % k, s, re.M) \
        else s + '\n%-15s %s;\n' % (k, v)
open(c, 'w').write(s + '\n')
# fvSolution: run exactly ITERS on both sides
p = d + '/system/fvSolution'; s = open(p).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
open(p, 'w').write(s)
if target == 'native':
    sys.exit(0)
# Scale the mesh. The factor is solved from the tutorial's own native count, which is only known after
# meshing, so it is estimated from the knobs themselves: the cell count is proportional to the product
# of the scaled directions, i.e. to factor**expo.
f = d + '/system/blockMeshDict'
if not os.path.exists(f):
    sys.exit(0)
s = open(f).read()
native = int(os.environ.get('NATIVE_CELLS', '0'))
if native <= 0:
    sys.exit(0)
factor = (int(target) / native) ** (1.0 / expo)
# SNAP (default on): round the factor to a whole number, so the scaled mesh is GEOMETRICALLY SIMILAR to
# the tutorial's own rather than a rounding of each block independently. Measured 2026-09-13: on
# squareBend's transonic bend, every non-integer factor tried gave OpenFOAM a diverged run (504k at
# x1.65, 975k at x2.07, 9.9M at x4.42) while both integer ones solved (896k at x2, 3.02M at x3) --
# including the finer of the two, so it is the block proportions and not the resolution. The realised
# cell count is what every row reports, so a snapped target is never silently misread as the asked one.
if os.environ.get('SNAP', '1') == '1' and factor >= 1.5:
    factor = float(round(factor))
if knobs == 'LITERAL':
    # every `(nx ny nz) simpleGrading` triple, scaled in all three directions
    s, n = re.subn(r'\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)(\s*(?:simple|edge)Grading)',
                   lambda m: '(%d %d %d)%s' % (max(1, round(int(m[1])*factor)),
                                               max(1, round(int(m[2])*factor)),
                                               max(1, round(int(m[3])*factor)), m[4]), s)
    if n == 0: sys.stderr.write('no literal (nx ny nz) triple in %s\n' % f)
else:
    # named entries, and ONLY their first (literal) definition: these dicts redefine some of them with
    # #eval expressions derived from the first, which then scale with it.
    for k in knobs.split():
        inv = k.startswith('/')          # a knob that is a cell SIZE, not a cell COUNT
        name = k[1:] if inv else k
        mul = (1.0/factor) if inv else factor
        def scaled(tok):
            v = float(tok)*mul
            # a count is written back as a count: blockMesh reads these with an int32 parser and
            # rejects "41.0" (measured: squareBendLiq and the aerofoil failed to mesh that way)
            return str(max(1, int(round(v)))) if '.' not in tok else repr(round(v, 6))
        s, n = re.subn(r'(^\s*%s\s+)([0-9.]+)(\s*;)' % re.escape(name),
                       lambda m: '%s%s%s' % (m[1], scaled(m[2]), m[3]),
                       s, count=1, flags=re.M)
        if n == 0: sys.stderr.write('knob %s not found in %s\n' % (name, f))
open(f, 'w').write(s)
PY
    # the case's own preparation if it has one (aerofoilNACA0012 extrudes; others just blockMesh)
    if [ -x "$d/Allrun.pre" ]; then ( cd "$d" && ./Allrun.pre > log.pre 2>&1 )
    else ( cd "$d" && blockMesh > log.blockMesh 2>&1 ); fi
    if [ -d "$d/processor0/constant/polyMesh" ]; then
        ( cd "$d" && reconstructParMesh -constant > log.reconstructParMesh 2>&1 )
        rm -rf "$d"/processor*
    fi
    [ -d "$d/0.orig" ] && { rm -rf "$d/0"; cp -r "$d/0.orig" "$d/0"; }
    rm -rf "$d"/[1-9]*
    local nc
    nc="$(grep -aoE 'nCells:?[[:space:]]*[0-9]+' "$d/constant/polyMesh/owner" 2>/dev/null | grep -oE '[0-9]+' | head -1)"
    echo "${nc:-FAILED}"
}

# Native cell count for a case, meshed once and cached, so the scale factor has a denominator.
native_cells(){
    local c="$1" g="$WORK/native_$c"
    if [ ! -f "$g/.ncells" ]; then
        local n; n="$(NATIVE_CELLS=0 stage "$c" native "$g")"
        echo "$n" > "$g/.ncells"
    fi
    cat "$g/.ncells"
}

run_arm(){   # $1 case dir, $2 arm -> "wall|itersReached"
    local d="$1" arm="$2" a b t n
    case "$arm" in
      of)
        if [ "$CORES" -gt 1 ]; then
            python3 - "$d" "$CORES" "$(case_decomp "$3")" <<'PY'
import sys
d, n, m = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if m == 'hierarchical':
    # a factorisation of n into three, longest direction first
    fac=[]; k=n
    for p in (2,3,5,7,11,13):
        while k % p == 0: fac.append(p); k//=p
    if k > 1: fac.append(k)
    xyz=[1,1,1]
    for f in sorted(fac, reverse=True): xyz[xyz.index(min(xyz))] *= f
    xyz.sort(reverse=True)
    coeffs='method hierarchical;\ncoeffs{n (%d %d %d);}\n' % tuple(xyz)
else:
    coeffs='method scotch;\n'
open(d+'/system/decomposeParDict','w').write(
 'FoamFile{version 2.0;format ascii;class dictionary;object decomposeParDict;}\n'
 'numberOfSubdomains %d;\n%s' % (n, coeffs))
PY
            ( cd "$d" && decomposePar -force > log.decomposePar 2>&1 )
            a=$(date +%s.%N); ( cd "$d" && mpirun -np "$CORES" rhoSimpleFoam -parallel > log.run 2>&1 ); b=$(date +%s.%N)
        else
            a=$(date +%s.%N); ( cd "$d" && rhoSimpleFoam > log.run 2>&1 ); b=$(date +%s.%N)
        fi ;;
      brae)
        a=$(date +%s.%N)
        ( cd "$d" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$d" > log.run 2>&1 )
        b=$(date +%s.%N) ;;
      amgx)
        export LD_LIBRARY_PATH="$AMGXLIB:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
        foamDictionary -entry libs -set '("libamgxFoam.so")' "$d/system/controlDict" > /dev/null 2>&1
        foamDictionary -entry solvers/p/solver -set amgx "$d/system/fvSolution" > /dev/null 2>&1
        # The wrapper's built-in config is PCG + aggregation, which diverges on the transonic (asymmetric)
        # pressure matrix; amgx_classical.json carries the config that solves it and says why.
        foamDictionary -entry solvers/p/amgxConfig -set "\"$HERE/amgx_classical.json\"" "$d/system/fvSolution" > /dev/null 2>&1
        a=$(date +%s.%N); ( cd "$d" && rhoSimpleFoam > log.run 2>&1 ); b=$(date +%s.%N) ;;
      petsc)
        # CLASSICAL AMG, not gamg. PETSc's gamg (aggregation) diverges on the transonic pressure matrix:
        # first solve 5 iterations, second 379 iterations to a residual of 358, and the case aborts on a
        # negative temperature; pc_gamg_sym_graph true does not save it. hypre's BoomerAMG solves the same
        # matrix in 3-5. That is the SAME verdict AMGX gives -- aggregation loses this matrix, classical
        # holds it -- so PETSc must be configured with --download-hypre to be a fair arm.
        export LD_LIBRARY_PATH="$PETSCLIB:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
        export PETSC_OPTIONS="-use_gpu_aware_mpi 0"
        foamDictionary -entry libs -set '("libpetscFoam.so")' "$d/system/controlDict" > /dev/null 2>&1
        python3 - "$d/system/fvSolution" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
new = ('    p\n    {\n        solver          petsc;\n        tolerance       1e-08;\n'
       '        relTol          0.1;\n        petsc\n        {\n            options\n            {\n'
       '                ksp_type      bcgs;\n                pc_type       hypre;\n'
       '                pc_hypre_type boomeramg;\n'
       '                mat_type      aijcusparse;\n                vec_type      cuda;\n'
       '            }\n        }\n    }\n')
s = re.sub(r'    p\n    \{.*?\n    \}\n', new, s, count=1, flags=re.S); open(p, 'w').write(s)
PY
        a=$(date +%s.%N); ( cd "$d" && rhoSimpleFoam > log.run 2>&1 ); b=$(date +%s.%N) ;;
      spuma)
        # SPUMA_TUNED=1 applies the solver set spuma's own published benchmark used: smoothSolver
        # symGaussSeidel on the transported fields, GAMG on pressure alone. The tutorials ask for GAMG on
        # EVERY field, and spuma obeys that literally -- five GAMG solves an iteration -- while brae
        # announces a substitution and runs a multicolour symGaussSeidel for U. Measuring spuma against
        # brae on the tutorial's own dictionary therefore compares spuma's worst configuration with
        # brae's chosen one. This arm is spuma at its best, and is reported as its own column.
        if [ "${SPUMA_TUNED:-0}" = 1 ]; then
            python3 - "$d/system/fvSolution" <<'PYS'
import re, sys
p = sys.argv[1]; s = open(p).read()
s = re.sub(r'"\(U\|e\|k\|epsilon\)"\s*\{[^{}]*\}',
           '"(U|e|h|k|epsilon|omega)"\n    {\n        solver          smoothSolver;\n'
           '        smoother        symGaussSeidel;\n        tolerance       1e-08;\n'
           '        relTol          0.1;\n    }', s)
open(p, 'w').write(s)
PYS
        fi
        a=$(date +%s.%N); ( "$HERE/spuma_run.sh" "$d" > "$d/log.run" 2>&1 ); b=$(date +%s.%N) ;;
    esac
    t=$(python3 -c "import sys; print('%.3f' % (float(sys.argv[2]) - float(sys.argv[1])))" "$a" "$b")
    n=$(grep -c '^Time = ' "$d/log.run" 2>/dev/null); n="${n:-0}"
    echo "$t|$n"
}

mkdir -p "$OUT" "$WORK"
CSV="$OUT/results.csv"
HD="$(hostdesc)"
[ -f "$CSV" ] || echo "host,gpu,cpu,ncores,case,target,ncells,arm,iters_asked,iters_done,wall_s,ms_per_iter,ns_per_cell,status" > "$CSV"
echo "host: $HD | arms: $ARMS | targets: $TARGETS | iters: $ITERS | of cores: $CORES"
printf "%-30s %10s %8s %10s %10s %9s  %s\n" case ncells arm wall_s ms/it ns/cell status

for c in $CASES; do
    NAT="$(native_cells "$c")"
    case "$NAT" in ''|*[!0-9]*) echo "  $c: native mesh FAILED, skipped"; continue ;; esac
    for target in $TARGETS; do
        D="$WORK/${c}_${target}"
        NC="$(NATIVE_CELLS=$NAT stage "$c" "$target" "$D")"
        case "$NC" in ''|*[!0-9]*) echo "  $c @ $target: meshing FAILED, skipped"; continue ;; esac
        for arm in $ARMS; do
            AD="$WORK/${c}_${target}_${arm}"; rm -rf "$AD"; cp -r "$D" "$AD"
            res="$(run_arm "$AD" "$arm" "$c")"
            t="${res%%|*}"; n="${res##*|}"
            if [ "$n" -ge "$ITERS" ]; then st=ok
                mspi=$(python3 -c "import sys; print('%.2f' % (float(sys.argv[1])*1000/float(sys.argv[2])))" "$t" "$ITERS")
                nspc=$(python3 -c "import sys; print('%.1f' % (float(sys.argv[1])*1e9/(float(sys.argv[2])*float(sys.argv[3]))))" "$t" "$ITERS" "$NC")
            else st="stopped_at_$n"; mspi=0; nspc=0; fi
            printf "%-30s %10s %8s %10.2f %10.2f %9.1f  %s\n" "$c" "$NC" "$arm" "$t" "$mspi" "$nspc" "$st"
            echo "$(echo "$HD" | tr '|' ','),$c,$target,$NC,$arm,$ITERS,$n,$t,$mspi,$nspc,$st" >> "$CSV"
            rm -rf "$AD"/[1-9]* "$AD"/processor* 2>/dev/null
        done
    done
done
echo "CSV -> $CSV"

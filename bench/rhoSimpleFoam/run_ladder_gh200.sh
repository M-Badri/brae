#!/usr/bin/env bash
# The cross-solver ladder on the GH200: every arm at matched mesh sizes on ONE machine, which is the
# comparison that means something (a GB10-vs-GH200 row compares two machines, not two solvers).
#
#   arms    brae (1 GPU) | OpenFOAM (all 64 cores) | AMGX | PETSc | spuma
#   ladder  15k 100k 1M 10M 24M, realised to whatever each case's mesher can actually build
#
# WHY THE TOP RUNG IS 24M AND NOT 30M: brae runs out of device memory above roughly 24 million cells on
# a 97 GB GH200 -- measured 2026-09-13, `brae cuda: pool cudaMalloc: out of memory` at 29,584,000 cells
# on the aerofoil, against 24,192,000 completed on squareBend. That is about 4 kB per cell. A rung brae
# cannot run yields no comparison, and OpenFOAM spent 4144 s producing the uncomparable half of it.
# THE CEILING IS NOT HIDDEN BY THIS: it is recorded in ../results/rhoSimpleFoam_gh200.md, and it is
# expensive -- at brae's flat 108.9 ns/cell the 29.6M rung would have been about 320 s against
# OpenFOAM's 4144 s, the largest margin in the campaign, lost to memory rather than to speed.
#
# SCOPE. The full product is 5 sizes x 6 cases x 5 arms = 150 runs and many hours: the aerofoil at 10M
# alone cost OpenFOAM 2449 s on 20 cores. Not every case reaches every rung either -- injectorPipe is a
# snappyHexMesh case whose background refinement topped out at 2.6M. So this defaults to the FULL ladder
# on the two cases that scale cleanly and bracket the behaviour (squareBend, aerofoilNACA0012) and
# native/1M/10M on the rest. Set FULL=1 for the whole product.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
BRAE="${BRAE:-$HOME/brae/build/brae_rhoSimpleFoam}"
AMGXLIB="${AMGXLIB:-$HOME/AMGX/build}"
PETSCLIB="${PETSCLIB:-$HOME/petsc/arch-gh200-hypre/lib}"
SPUMA_DIR="${SPUMA_DIR:-$HOME/spac/spuma-fresh}"
ARMS="${ARMS:-brae of amgx petsc spuma}"
ITERS="${ITERS:-100}"
CORES="${CORES:-$(nproc)}"
OUT="${OUT:-$HOME/ladder_gh200}"
export SPUMA_DIR
run(){ CASES="$1" TARGETS="$2" ARMS="$ARMS" ITERS="$ITERS" CORES="$CORES" OUT="$OUT" \
       WORK="${WORK:-/tmp/ladder_w}" BRAE="$BRAE" AMGXLIB="$AMGXLIB" PETSCLIB="$PETSCLIB" \
       bash "$HERE/run_matrix.sh"; }

if [ "${FULL:-0}" = 1 ]; then
    run "squareBend squareBendLiq squareBendLiqNoNewtonian angledDuctExplicitFixedCoeff aerofoilNACA0012 injectorPipe" \
        "15000 100000 1000000 10000000 24000000"
else
    run "squareBend aerofoilNACA0012" "15000 100000 1000000 10000000 24000000"
    run "squareBendLiq squareBendLiqNoNewtonian angledDuctExplicitFixedCoeff injectorPipe" \
        "native 1000000 10000000"
fi
echo "ladder CSV -> $OUT/results.csv"

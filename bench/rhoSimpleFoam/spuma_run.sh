#!/usr/bin/env bash
# Run spuma's rhoSimpleFoam on a staged case. spuma is a separate OpenFOAM tree, so it needs its OWN
# environment (WM_PROJECT_DIR, FOAM_ETC and its libraries) -- the calling harness has the stock
# OpenFOAM sourced, and the two cannot share one shell.
#
# The GPU is reached through spuma's memory pool, the same way its own run script does it:
#   rhoSimpleFoam -pool fixedSizeMemoryPool -poolSize <GB>
#
# Env: SPUMA_DIR (default $HOME/spuma), SPUMA_POOLGB (default 8), SPUMA_SDK (default the 26.5 SDK).
set -u
CASE="${1:?usage: spuma_run.sh <caseDir>}"
SPUMA_DIR="${SPUMA_DIR:-$HOME/spuma}"
# Pool size from the mesh. spuma_sweep.sh on the GB10 used 8 GB at ~4.9M cells and 56 GB at ~35.6M,
# i.e. about 1.6 kB/cell; take twice that for headroom, floor 4 GB. fixedSizeMemoryPool must be big
# enough for the case or the run dies inside it.
if [ -z "${SPUMA_POOLGB:-}" ] && [ -f "$1/constant/polyMesh/owner" ]; then
    NC=$(grep -aoE 'nCells:?[[:space:]]*[0-9]+' "$1/constant/polyMesh/owner" | grep -oE '[0-9]+' | head -1)
    SPUMA_POOLGB=$(python3 -c "import sys; n=int(sys.argv[1]); print(max(4, int(n*3400/1073741824)+2))" "${NC:-0}")
fi
SPUMA_POOLGB="${SPUMA_POOLGB:-8}"
SPUMA_SDK="${SPUMA_SDK:-/opt/nvidia/hpc_sdk/Linux_aarch64/26.5}"
export PATH="$SPUMA_SDK/compilers/bin:$PATH"
export have_cuda=true
export FOAM_SIGFPE=false
export WM_COMPILER=Nvidia
set +u
# shellcheck disable=SC1091
source "$SPUMA_DIR/etc/bashrc" > /dev/null 2>&1
set -u
BIN="$SPUMA_DIR/platforms/${WM_OPTIONS:-linuxARM64NvidiaDPInt32Opt}/bin/rhoSimpleFoam"
[ -x "$BIN" ] || { echo "spuma rhoSimpleFoam not built at $BIN"; exit 2; }
cd "$CASE" || exit 2
exec "$BIN" -pool fixedSizeMemoryPool -poolSize "$SPUMA_POOLGB"

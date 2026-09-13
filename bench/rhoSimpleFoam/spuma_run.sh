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

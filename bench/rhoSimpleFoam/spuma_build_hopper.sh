#!/usr/bin/env bash
# spuma for Hopper, with the combination that actually works:
#   SDK 2026      -- 26.5's device linker emits duplicate __cudaRegisterLinkedBinary symbols for cc80 and
#                    cc90 under RDC (proven on the GB10 by changing only NVARCH: cc121 clean, cc90 and
#                    cc80 collide; then only the SDK: cc90 clean under 2026). 26.5 was pinned earlier on
#                    a wrong inference -- the first 2026 attempt failed on the LINKER, not the SDK.
#   lld for libs  -- GNU ld overflows the aarch64 GOT linking libOpenFOAM.so; spuma's own rule uses lld
#                    for LINKLIBSO only, and GNU ld for executables (lld rejects --add-needed).
#   RDC ON        -- -gpu=nordc builds but strips the device code ("No CUDA device code available").
# Builds the LIBRARIES and rhoSimpleFoam ONLY, not every solver.
set -u
N=$HOME/spac/spuma-fresh
[ -d "$N" ] || N=$HOME/spuma
SDK=/opt/nvidia/hpc_sdk/Linux_aarch64/2026/
mkdir -p "$N/ldshim" && ln -sf /usr/bin/ld.lld-18 "$N/ldshim/ld"
sed -i 's|PATH="/home/ghost/space/spuma-fresh/ldshim:\$\$PATH"|PATH="$(WM_PROJECT_DIR)/ldshim:$$PATH"|' \
    "$N/wmake/rules/General/Nvidia/link-c++" 2>/dev/null
export PATH="${SDK}compilers/bin:$PATH"
export have_cuda=true NVARCH=90 FOAM_SIGFPE=false WM_COMPILER=Nvidia
set +u
source "$N/etc/bashrc"
export have_cuda=true NVARCH=90 FOAM_SIGFPE=false
unset FOAM_EXTRA_CXXFLAGS
set -u
echo "tree=$N  nvc++=$(command -v nvc++)  NVARCH=$NVARCH"
grep -n LINKLIBSO "$N/wmake/rules/General/Nvidia/link-c++"
rm -rf "$N/platforms" "$N/build"
( cd "$N/wmake/src" && make > /tmp/sp_tools.log 2>&1 ); echo "tools rc=$?"
( cd "$N/src" && ./Allwmake -j 48 > /tmp/sp_libs.log 2>&1 )
echo "libs rc=$?  built=$(ls "$N"/platforms/*/lib/*.so 2>/dev/null | wc -l)  collisions=$(grep -cE 'redefinition of .__cudaRegister' /tmp/sp_libs.log 2>/dev/null)"
( cd "$N/applications/solvers/compressible/rhoSimpleFoam" && wmake -j 24 > /tmp/sp_rsf.log 2>&1 )
echo "rhoSimpleFoam rc=$?"
ls -la "$N"/platforms/*/bin/rhoSimpleFoam 2>/dev/null && echo SPUMA_HOPPER_OK || { echo SPUMA_HOPPER_FAIL; grep -nE "error:|Error [0-9]" /tmp/sp_libs.log /tmp/sp_rsf.log 2>/dev/null | head -4; }

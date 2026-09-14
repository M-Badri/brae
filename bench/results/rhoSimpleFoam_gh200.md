# rhoSimpleFoam on a GH200, and what changes from the GB10

Measured 2026-09-13 with `bench/rhoSimpleFoam/run_matrix.sh`, the same harness and the same rules as
the GB10 table (`rhoSimpleFoam_tutorials_gb10.md`): 100 fixed SIMPLE iterations, solver wall only,
`functions` stripped and `residualControl` removed, and a row reported as a time only if it reached the
last iteration.

| | GB10 | GH200 |
|---|---|---|
| GPU | NVIDIA GB10, sm_121 | NVIDIA GH200 480GB, sm_90, 97 GB HBM3 |
| CPU arm | 20 Grace cores | 64 Grace cores |
| CUDA | 13.0 | 13.0 (installed for this; the distro repo stops at 12.9) |
| OpenFOAM | v2412 | v2412 |

Each machine's CPU arm uses ALL of that machine's cores, so each row is "one GPU against the whole box".

## Getting brae onto it

Three things had to be fixed or installed, and the first is a real defect the second machine found:

1. **brae did not compile on CUDA 12 at all.** `neumannPrecon` was defined inside
   `#ifdef BRAE_HAS_GS_DEVICE` (gated on `CUDART_VERSION >= 13000`) and called from the host loop
   outside it. See the GB10 file for the fix. The GH200's distro CUDA is 12.8.
2. **CUDA 13 is not in the Ubuntu 22.04 arm64 repo** (12.9 is the ceiling), so the device-resident
   conditional-graph path could not be built from apt. The driver there is 580.105.08, which reports
   `CUDA Version: 13.0`, so the 13.0.2 runfile installs toolkit-only against the existing driver.
   Without this the GH200 would have run a different code path and the comparison would be worthless.
3. **`bc` is not installed on that image**, and the harness used it for every time it reported: the
   first GH200 run completed every case and recorded a blank wall time in every row. The harness now
   does its arithmetic in python3, which it already required. A benchmark harness meant for arbitrary
   hardware cannot depend on a tool that arbitrary hardware lacks.

**The same binary source gives the same numbers on both architectures.** squareBend, 10 iterations,
sm_121 and sm_90: every printed digit of every residual line identical, `U 2.0060e-01  e 1.3785e-01
p 6.9731e-02  k 9.4371e-02  epsilon 2.7325e-02` at iteration 10 on both.

## The two machines side by side

Rows where both machines ran the same mesh:

| case | cells | GB10 brae | OF-20c | ratio | GH200 brae | OF-64c | ratio | brae GB10 -> GH200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aerofoilNACA0012 | 16,000 | 1.1 s | 1.7 s | 1.64x | 0.9 s | 1.6 s | 1.73x | 1.15x |
| aerofoilNACA0012 | 1,024,000 | 38.0 s | 156.4 s | 4.12x | 11.1 s | 44.5 s | 4.00x | **3.41x** |
| aerofoilNACA0012 | 10,000,000 | 610.7 s | 2448.9 s | 4.01x | **108.2 s** | 916.7 s | **8.47x** | **5.64x** |
| angledDuct | 28,000 | 1.1 s | 1.3 s | 1.23x | 0.8 s | 0.9 s | 1.05x | 1.31x |
| injectorPipe | 74,650 | 2.1 s | 3.0 s | 1.41x | 1.4 s | 1.5 s | 1.10x | 1.55x |
| squareBend | 112,000 | 2.5 s | 3.9 s | 1.60x | 1.6 s | 1.8 s | 1.14x | 1.53x |
| squareBendLiq | 112,000 | 2.7 s | 3.6 s | 1.32x | 1.7 s | 1.6 s | **0.92x** | 1.59x |
| squareBendLiq | 896,000 | 18.9 s | 26.6 s | 1.41x | 6.8 s | 8.4 s | 1.24x | **2.80x** |
| squareBendLiqNoNewtonian | 112,000 | 2.0 s | 2.7 s | 1.34x | 1.3 s | 1.2 s | **0.96x** | 1.61x |
| squareBendLiqNoNewtonian | 896,000 | 16.0 s | 19.2 s | 1.20x | 5.4 s | 6.2 s | 1.16x | **2.98x** |

GH200-only rows (the snap-to-integer scaling lands these on different meshes than the GB10 run, which
predates that fix): squareBend 896,000 1.34x and 7,168,000 **2.12x**; squareBendLiq 7,168,000 1.96x;
NoNewtonian 7,168,000 1.75x; angledDuct 437,500 0.95x and 3,500,000 1.49x; injectorPipe 573,858 1.10x and 2,576,294 1.56x.

### Three things this says

**BRAE'S OWN SPEED-UP FROM THE HARDWARE GROWS WITH THE MESH: 1.15x at 16,000 cells, 3.41x at a million,
5.64x at ten million.** A GH200 is not 5.6x a GB10 on paper; what changes is how much of the iteration
is bandwidth rather than launch overhead. At 16,000 cells brae is launch-bound on both machines and the
faster memory buys almost nothing (the T-1 launch floor, measured on the GB10 at 1,155 kernels an
iteration too small to fill the device). At ten million it is bandwidth-bound and HBM3 is exactly what
it wants. The per-cell numbers say it plainly: 667 -> 611 ns on the GB10 as the aerofoil grows, against
582 -> 108 ns on the GH200.

**THE AEROFOIL'S 4x IS ALGORITHMIC, AND THE GH200 CONFIRMS IT.** T-6 concluded that the aerofoil's
margin comes from OpenFOAM's GAMG degrading on that anisotropic mesh (23 -> 185 -> 312 iterations),
not from brae getting faster. If that were hardware, moving machines would change it; it does not --
4.12x on the GB10, 4.00x on the GH200 at the same million-cell mesh. At ten million it becomes 8.47x
because brae ALSO gets the bandwidth win on top of the algorithmic one.

**ON SMALL MESHES, 64 CORES OF OPENFOAM BEAT brae.** squareBendLiq at 112,000 cells is 0.92x, the
NoNewtonian variant 0.96x, angledDuct at 437,500 cells 0.95x -- OpenFOAM wins those outright. On the
GB10's 20 cores brae won all three. This is the launch floor again, and it is now a number with a
competitor attached: a case has to be big enough to fill the GPU before brae's port is worth anything,
and "big enough" rises with how many CPU cores it is being compared against. T-1 and T-2 are about
exactly this, and the GH200 is where they would pay.

## T-4 unblocked: AMGX works on sm_90 (2026-09-13)

The AMGX arm had been empty for the whole campaign for one reason, recorded in `amgxSolver.C` itself:
AMGX 2.5.0's CLASSICAL aggregation path throws a Thrust error on sm_121, so the GB10 could only run
AGGREGATION, and aggregation cannot solve this pressure matrix -- 414 iterations to reach 0.62 relative
residual, or divergence to a negative temperature with the wrapper's stock PCG.

The GH200 is sm_90. Built there: AMGX from source (`-DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_NO_MPI=True`,
plus `-DCMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES=/usr/local/cuda/include`, without which its own examples
fail to configure), then `libamgxFoam.so` against it. squareBend at 112,000 cells, `transonic yes`,
FGMRES + classical AMG (PMIS/D2):

| | iterations to relTol 0.01 |
|---|---|
| AMGX aggregation, sm_121 (GB10) | 414, and it only reached 0.62 |
| **AMGX classical, sm_90 (GH200)** | **2 to 4** |

The run completes, exits 0, and does not diverge. `bench/rhoSimpleFoam/amgx_classical.json` carries that
config with the measurement that chose each of its two non-default settings, and `run_matrix.sh`'s amgx
arm now points at it instead of the wrapper's built-in default.

So the comparison brae-vs-AMGX is finally possible, and only on Hopper.

## Building the competitor arms on the GH200 (2026-09-13)

Four things had to be built or installed before brae could be compared against anything but OpenFOAM.
Recorded because each cost a cycle to find and none is in any README.

**AMGX** -- clone, then `cmake -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_NO_MPI=True
-DCMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES=/usr/local/cuda/include`. Without that last one AMGX's own
`examples/eigen_examples/CMakeLists.txt` calls `include_directories("")` and configure fails. Then
`libamgxFoam.so` with `AMGX_INC=~/AMGX/include AMGX_LIB=~/AMGX/build`; it needs neither foam2csr nor
PETSc, because the wrapper does its own LDU-to-CSR conversion.

**The NVIDIA HPC SDK** is not in the Ubuntu 22.04 arm64 repos. Add
`https://developer.download.nvidia.com/hpc-sdk/ubuntu/arm64`; it installs about 12 GB.

**spuma needs LLVM's linker, not GNU ld.** Linking `libOpenFOAM.so` with the CUDA device stub overflows
the aarch64 GOT:

    linkstub.c: relocation truncated to fit: R_AARCH64_LD64_GOTPAGE_LO15 against `__NV_CUDA_LOC'
    warning: too many GOT entries for -fpic, please recompile with -fPIC
    final link failed

`-fPIC` is already in spuma's own c++FLAGS, so the advice in the warning is a dead end -- the object at
fault is the stub `pgacclnk` generates, not spuma's code. The fix is `ld.lld`: `apt-get install lld`,
then a directory holding `ld -> /usr/bin/ld.lld` placed FIRST on PATH so `pgacclnk` picks it up as plain
`ld`. The working GB10 tree carries exactly this as `spuma-fresh/ldshim/ld`, which is how it was found.
With the shim, `libOpenFOAM.so` links and the error count goes to zero.

**spuma's GPU build fails on a duplicate CUDA registration symbol, and the cause is the PATH.** Past
the linker, `libfiniteVolume.so` fails with

    linkstub.c:7: error: redefinition of `__cudaRegisterLinkedBinary_58_home_ubuntu_spuma_src_
                          OpenFOAM_lnInclude_FieldFunctions_C_51a0'

`-gpu=nordc` makes that go away and IS NOT A FIX: it removes relocatable device code along with the
colliding stub, and the resulting binary aborts at start-up with `No CUDA device code available`. It
builds and it has no GPU in it.

Everything environmental was equalised against the GB10 tree that builds these same sources cleanly,
and each was ruled out by measurement rather than by argument:

| suspected | result |
|---|---|
| build parallelism | identical at `-j 48`, `-j 16`, `-j 8` -- deterministic, not a race |
| HPC SDK version | pinned 26.5, the GB10's version; unchanged |
| the ld shim's scope | spuma's own rule applies lld to SHARED LIBS only and GNU ld to executables (lld rejects the `--add-needed` wmake passes when linking an application). Scoped correctly; unchanged |
| linker version | GH200 had lld 14, the GB10 has 18; installed 18.1.8; unchanged |
| CUDA pairing | both `/usr/local/cuda -> 13.0` plus the SDK's bundled 13.2; identical |

The symbol encodes the source path -- `58` is the length of
`/home/ubuntu/spuma/src/OpenFOAM/lnInclude/FieldFunctions.C`, `51a0` a four-hex-digit hash -- and the
GB10 builds the identical sources at length 69, so the path looked like the last candidate. It is not:
rebuilt at `/home/ubuntu/spac/spuma-fresh`, the same 69 characters as the working tree, the collision
returns as `..._69_home_ubuntu_spac_spuma_fresh_..._2371`. (An intermediate run of that rebuild reported
zero collisions, which was not a fix -- deleting `platforms/` had removed wmake's own `wmkdepend`, so the
build died before it ever reached a device link.)

SO THIS IS UNRESOLVED, and what remains is a difference that cannot be equalised from here: the GB10's
`libfiniteVolume.so` was built on 2026-07-05, and the machine now carries a NEWER SDK alongside 26.5. A
tree whose artifacts predate a toolchain upgrade is not evidence that the toolchain still builds it --
the honest test is a clean full rebuild of spuma on the GB10, which would say whether this is specific
to the GH200 or whether the working tree simply has not been rebuilt since. Not run, because it would
take an hour of someone else's machine.

`-gpu=nordc` remains the only configuration that completes, and it is not usable: no device code.

Also fixed on the way: `wmake/rules/General/Nvidia/link-c++` hardcoded
`PATH="/home/ghost/space/spuma-fresh/ldshim:$PATH"` -- one machine's absolute path, which travels with
the tree and silently does nothing anywhere else. Now `$(WM_PROJECT_DIR)/ldshim`.

**Pin the SDK version.** `ls /opt/nvidia/hpc_sdk/Linux_aarch64/*/ | sort -V | tail -1` picks `2026`
over `26.5`; the GB10's working build pins 26.5. (The GOT failure happens under both, so the version was
not the cause -- but matching the known-good toolchain removes one variable.)

**PETSc** configures with `--with-cuda=1 --with-cuda-arch=90 --download-fblaslapack=1` against the
distro's `mpicc`/`mpicxx`/`mpif90`. Built WITHOUT `--with-amgx`: AMGX already has its own direct arm
through `libamgxFoam.so`, and coupling them only adds failure modes to a comparison that does not need it.

## Both GPU libraries need CLASSICAL AMG on this matrix (2026-09-13)

The tutorials run `transonic yes`, so the pressure matrix is asymmetric. Measured on squareBend at
112,000 cells, each library's stock aggregation-type multigrid against its classical one:

| arm | aggregation-type | classical |
|---|---|---|
| **AMGX** | AGGREGATION: 414 iterations, reaching only 0.62 relative residual (and the wrapper's default PCG, a symmetric solver, diverges outright) | CLASSICAL (PMIS/D2): **2 to 4 iterations** |
| **PETSc** | `gamg`: first solve 5 iterations, second 379 iterations to a residual of **358**, case aborts on a negative temperature. `pc_gamg_sym_graph true` does not save it | hypre `boomeramg`: **3 to 5 iterations** |

Two independent libraries, the same verdict: aggregation loses this matrix and classical holds it. That
is worth stating plainly because it is what made the arms unusable for the whole campaign -- and on the
GB10 it was unfixable, since AMGX 2.5.0's classical path throws on sm_121. PETSc needs
`--download-hypre` at configure time to have a classical option at all.

Both arms now use the classical configuration in `run_matrix.sh`, so a comparison against brae is
finally measuring the libraries at their best on this problem rather than at their defaults.

## brae's memory ceiling: about 4 kB per cell (2026-09-13)

Found by running the ladder past ten million cells on a 97 GB GH200.

    brae cuda: pool cudaMalloc: out of memory

at **29,584,000 cells** (aerofoilNACA0012), against **24,192,000 completed** on squareBend. So the
ceiling on this card sits between those two, and the footprint is roughly **4 kB per cell**.

IT IS NOT THE POOL, and it is not precision. Both were measured rather than argued:

| | measured |
|---|---|
| squareBend, 112,000 cells | 0.381 GB of `cudaMalloc` in 824 calls = **3.56 kB/cell** |
| aerofoil, 16,000 cells | 0.097 GB = 6.4 kB/cell (the fixed overheads have not amortised yet) |
| pool efficiency | only **11% idle** in the free list, 6 distinct sizes, 57 blocks -- it recycles well |
| AMG FP32 + CSR mirrors | disabling the CSR mirror saves **1 MB** at 16k -- not the consumer |

`BRAE_POOL_STATS=1` prints the running totals (`device_buffer.cuh`). The pool keeps a size-exact free
list and never calls `cudaFree` while enabled, which looks like hoarding and is not: at 6 distinct sizes
it recycles almost everything it takes.

FP32 IS NOT THE LEVER HERE, for two separate reasons. The solver's agreement with OpenFOAM is the whole
claim -- the converged 896k comparison is 7.08e-08 in U and the identity gates run at 1e-11 -- and FP32's
seven digits cannot carry that. And where FP32 IS legitimate, inside the preconditioner (a preconditioner
changes the iteration count, never the converged answer), brae already uses it: `device_amg.cuh` calls
them "FP32 mirrors of every level matrix", with `csrSrc` existing to refill them FROM the FP64 arrays.
They are additions, not replacements, so today they cost memory rather than save it -- and the
measurement above says they cost about a megabyte, so converting them would not move this.

WHAT THE CEILING COSTS, which is why it now outranks the launch-floor work: at 29,584,000 cells
OpenFOAM on 64 cores finished in **4,143.9 s** and brae scored nothing. brae's cost per cell is flat at
108.7 -> 108.9 ns from 1M to 10M, so that rung would have been roughly **320 s -- about 13x**, the
largest margin anywhere in this campaign, lost to memory rather than to speed.

The next step is attribution, not a guess: tag allocations by call site and rank the consumers. Two
hypotheses were formed and killed by measurement on the way here (the pool was hoarding; the path length
drove spuma's symbol collision), which is the argument for measuring this one before touching anything.

## The cross-solver ladder on the GH200 (2026-09-13/14)

One GPU against all 64 Grace cores, 100 fixed SIMPLE iterations, `bench/rhoSimpleFoam/run_matrix.sh`.
Solver wall only. A row is a time only if the run reached the last iteration.

| case | cells | brae | OpenFOAM-64c | AMGX | PETSc | brae/OF |
|---|---:|---:|---:|---:|---:|---:|
| squareBend | 14,200 | 0.8 | 1.2 | 3.5 | 4.5 | **1.48x** |
| squareBend | 97,470 | 1.4 | stopped at it 4 | 16.9 | 18.7 | - |
| squareBend | 896,000 | 7.0 | 9.5 | 142.8 | 148.2 | **1.35x** |
| squareBend | 7,168,000 | 50.9 | 107.2 | 1229.9 | 1278.5 | **2.11x** |
| squareBend | 24,192,000 | 180.1 | stopped at it 24 | stopped at it 38 | stopped at it 1 | - |
| squareBendLiq | 112,000 | 1.7 | 1.7 | 21.0 | 23.0 | **0.98x** |
| squareBendLiq | 896,000 | 6.8 | 8.4 | 151.7 | 157.2 | **1.23x** |
| squareBendLiq | 7,168,000 | 47.6 | 93.3 | 1354.2 | 1389.6 | **1.96x** |
| squareBendLiq | 24,192,000 | 166.3 | 336.7 | - | - | **2.02x** |
| squareBendLiqNoNewtonian | 112,000 | 1.2 | 1.2 | 13.8 | 15.7 | **1.01x** |
| squareBendLiqNoNewtonian | 896,000 | 5.4 | 6.2 | 106.7 | 112.2 | **1.16x** |
| squareBendLiqNoNewtonian | 7,168,000 | 39.0 | 68.0 | 936.5 | 972.2 | **1.74x** |
| squareBendLiqNoNewtonian | 24,192,000 | 131.3 | 242.7 | - | - | **1.85x** |
| angledDuctExplicitFixedCoeff | 28,000 | 0.7 | 0.9 | 4.2 | 5.3 | **1.21x** |
| angledDuctExplicitFixedCoeff | 437,500 | 3.5 | 3.4 | 48.3 | 51.8 | **0.96x** |
| angledDuctExplicitFixedCoeff | 3,500,000 | 23.0 | 34.1 | 445.4 | 467.3 | **1.48x** |
| aerofoilNACA0012 | 14,938 | 0.8 | 1.6 | 2.9 | 4.0 | **1.99x** |
| aerofoilNACA0012 | 64,000 | 1.5 | 3.5 | 8.5 | 9.8 | **2.38x** |
| aerofoilNACA0012 | 1,024,000 | 11.1 | 44.6 | 127.7 | 132.7 | **4.01x** |
| aerofoilNACA0012 | 10,000,000 | 108.9 | 921.0 | 1482.8 | stopped at it 3 | **8.46x** |
| aerofoilNACA0012 | 29,584,000 | stopped at OOM | 4143.9 | - | - | - |
| injectorPipe | 74,650 | 1.4 | 1.5 | 14.1 | 13.6 | **1.09x** |
| injectorPipe | 573,858 | 5.1 | 5.5 | 110.0 | 110.1 | **1.09x** |
| injectorPipe | 2,576,294 | 21.5 | 33.8 | 587.4 | 580.2 | **1.57x** |

**brae is faster on 19 of the 21 rungs where both codes completed.**

Three caveats travel with that table and must not be dropped from it:

- **The AMGX and PETSc columns do not measure those libraries.** Both arms run OpenFOAM SERIALLY with only
  the pressure equation on the GPU, against the `of` arm's 64 cores. That is the whole of their 10-25x
  deficit. AMGX as built here (`-DCMAKE_NO_MPI=True`) cannot run multi-rank at all. Rebuild AMGX with MPI
  and run both under `mpirun -np 64` before either column is published.
- **brae carries its whole cold start; OpenFOAM does not carry `decomposePar`.** The timer wraps brae's
  entire process -- mesh read, device set-up, AMG hierarchy build, field reads, solve, write -- while the
  OpenFOAM arm times only `mpirun ... rhoSimpleFoam -parallel`, with decomposePar and reconstructPar
  outside it. At 24M that partitioning is minutes. The small-mesh rungs are the ones this distorts: brae's
  start-up measured about 0.4 s of a 1.02 s aerofoil run, so 0.98x and 0.96x are mostly set-up, not solve.
- `stopped at it 4` on squareBend at 97,470 is the mesh generator, not OpenFOAM: `SNAP` only rounds scale
  factors at or above 1.5, so the sub-native rungs still get non-integer factors.

## Why spuma has no column: it builds for Blackwell and not for Hopper (2026-09-14)

Root-caused by changing ONE variable. The clean tree on the GB10, same SDK 26.5, same lld 18, same
sources, same path, relocatable device code on:

| target | result |
|---|---|
| `-gpu=cc121` (Blackwell) | `libfiniteVolume.so` **builds clean, 0 collisions** |
| `-gpu=cc90` (Hopper) | **collision**, `__cudaRegisterLinkedBinary_69_home_ghost_space_spuma_clean_src_OpenFOAM_lnInclude_FieldFunctions_C_7f4f`, build fails |

So it is the TARGET ARCHITECTURE, not the GH200, not the path, not the SDK version, not the linker --
all of which were equalised and eliminated first. It also explains why the GB10 tree works: it has only
ever been built cc121.

Two things that look like fixes and are not, both recorded because each cost a cycle:

- `-gpu=nordc` completes the build by removing relocatable device code, and the binary then aborts at
  start-up with `No CUDA device code available`. It builds and has no GPU in it.
- `-gpu=cc90a` reports zero collisions because nvc++ 26.5 does not accept it -- valid values are
  `35 50 60 61 62 70 72 75 80 86 87 88 89 90 100 101 103 110 120 121`, with no `a` suffix. It printed its
  usage text and stopped, so the build never reached a device link. A zero that means "never got there"
  is not a zero; the same false negative appeared earlier when a deleted `platforms/` removed wmake's own
  `wmkdepend`.

The reproducer is now minimal -- one library, one flag, one machine -- and belongs upstream with spuma.

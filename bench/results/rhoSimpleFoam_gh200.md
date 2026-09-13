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

**spuma also needs `-gpu=nordc`.** Past the linker, `libfiniteVolume.so` fails with

    linkstub.c:7: error: redefinition of `__cudaRegisterLinkedBinary_58_home_ubuntu_spuma_src_
                          OpenFOAM_lnInclude_FieldFunctions_C_5348'

-- two device binaries in one link both registering the same source path. It is deterministic, not a
parallel-build race: identical at `-j 48`, `-j 16` and `-j 8`. The GB10's build log has zero occurrences,
so it is specific to this tree (the symbol encodes the source path, and the two trees' paths differ in
length -- 58 characters here against 69 there). `FOAM_EXTRA_CXXFLAGS=-gpu=nordc` turns off relocatable
device code, which removes the device-link stub that collides, and the build goes straight through.

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

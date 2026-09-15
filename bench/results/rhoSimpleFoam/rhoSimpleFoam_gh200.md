# rhoSimpleFoam on a GH200, and what changes from the GB10

Measured 2026-09-13/15 with `bench/rhoSimpleFoam/run_matrix.sh`: 100 fixed SIMPLE iterations, solver
wall only, `functions` stripped, `residualControl` removed, and a row reported as a time only if it
reached the last iteration. Same rules as `rhoSimpleFoam_tutorials_gb10.md`.

| | GB10 | GH200 |
|---|---|---|
| GPU | NVIDIA GB10, sm_121 | NVIDIA GH200 480GB, sm_90, 97 GB HBM3 |
| CPU arm | 20 Grace cores | 64 Grace cores |
| CUDA | 13.0 | 13.0 (installed for this; the distro repo stops at 12.9) |
| OpenFOAM | v2412 | v2412 |

Each machine's CPU arm uses ALL of that machine's cores: every row is one GPU against the whole box.
Raw rows in `gh200_csv/`.

## Getting brae onto it

- **brae did not compile on CUDA 12 at all** -- `neumannPrecon` was defined inside
  `#ifdef BRAE_HAS_GS_DEVICE` (gated on `CUDART_VERSION >= 13000`) and called from outside it. A real
  defect the second machine found.
- **CUDA 13 is not in the Ubuntu 22.04 arm64 repo** (12.9 is the ceiling); the 13.0.2 runfile installs
  toolkit-only against driver 580.105.08. Without it the GH200 runs a different code path.
- **`bc` is not on that image** and the harness used it for every reported time -- the first run recorded
  a blank wall in every row. Arithmetic now in python3.

Same source, same numbers on both architectures: squareBend at 10 iterations, sm_121 and sm_90, every
printed digit identical (`U 2.0060e-01  e 1.3785e-01  p 6.9731e-02  k 9.4371e-02  epsilon 2.7325e-02`).

## The two machines side by side

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

- **brae's hardware speed-up grows with the mesh**: 1.15x at 16k, 3.41x at 1M, 5.64x at 10M. At 16k it
  is launch-bound on both machines; at 10M it is bandwidth-bound and HBM3 is what it wants. Per cell:
  667 -> 611 ns on the GB10 as the aerofoil grows, against 582 -> 108 ns on the GH200.
- **The aerofoil's 4x is algorithmic, and the GH200 confirms it.** It comes from OpenFOAM's GAMG
  degrading on that mesh (23 -> 185 -> 312 iterations), not from brae accelerating. Hardware-invariant:
  4.12x on the GB10, 4.00x on the GH200 at the same 1M mesh. It reaches 8.47x at 10M only because brae
  then also gets the bandwidth win.
- **On small meshes 64 cores of OpenFOAM beat brae**: squareBendLiq 112k 0.92x, NoNewtonian 0.96x,
  angledDuct 437.5k 0.95x. On the GB10's 20 cores brae won all three. A case must be big enough to fill
  the GPU, and "big enough" rises with the core count it is compared against.

## AMGX works on sm_90, and both GPU libraries need CLASSICAL AMG

AMGX 2.5.0's classical path throws a Thrust error on sm_121, so the GB10 could only run AGGREGATION --
which loses this matrix. The GH200 is sm_90, so the arm is finally possible. Built with
`-DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_NO_MPI=True -DCMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES=/usr/local/cuda/include`
(without the last, AMGX's own examples fail to configure).

The tutorials run `transonic yes`, so the pressure matrix is asymmetric. squareBend, 112,000 cells:


| arm | aggregation-type | classical |
|---|---|---|
| **AMGX** | AGGREGATION: 414 iterations, reaching only 0.62 relative residual (and the wrapper's default PCG, a symmetric solver, diverges outright) | CLASSICAL (PMIS/D2): **2 to 4 iterations** |
| **PETSc** | `gamg`: first solve 5 iterations, second 379 iterations to a residual of **358**, case aborts on a negative temperature. `pc_gamg_sym_graph true` does not save it | hypre `boomeramg`: **3 to 5 iterations** |

Two independent libraries, same verdict: aggregation loses this matrix, classical holds it. PETSc needs
`--download-hypre` at configure time to have a classical option at all. Both arms now use the classical
configuration; `amgx_classical.json` carries the measurement behind each non-default setting.

## brae's memory ceiling: about 4 kB per cell

`brae cuda: pool cudaMalloc: out of memory` at **29,584,000 cells** (aerofoil), against **24,192,000
completed** (squareBend). Not the pool, and not precision -- both measured, not argued:

| | measured |
|---|---|
| squareBend, 112,000 cells | 0.381 GB of `cudaMalloc` in 824 calls = **3.56 kB/cell** |
| aerofoil, 16,000 cells | 0.097 GB = 6.4 kB/cell (the fixed overheads have not amortised yet) |
| pool efficiency | only **11% idle** in the free list, 6 distinct sizes, 57 blocks -- it recycles well |
| AMG FP32 + CSR mirrors | disabling the CSR mirror saves **1 MB** at 16k -- not the consumer |

`BRAE_POOL_STATS=1` prints the totals. FP32 is not the lever: the agreement claim is 7.08e-08 in U at
896k with identity gates at 1e-11, which seven digits cannot carry; and where FP32 IS legitimate --
inside the preconditioner -- brae already uses it, as additions to the FP64 arrays rather than
replacements, costing about a megabyte.

**What the ceiling costs**: at 29.58M OpenFOAM finished in 4,143.9 s and brae scored nothing. brae is
flat at 108.7 -> 108.9 ns/cell from 1M to 10M, so that rung would have been roughly **320 s, about 13x**
-- the largest margin in the campaign, lost to memory rather than to speed. Next step is attribution by
call site, not a guess.

## The cross-solver ladder

One GPU against all 64 Grace cores, 100 fixed iterations, solver wall only.

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

**brae is faster on 19 of the 21 rungs where both codes completed.** Three caveats travel with that:

- **The AMGX and PETSc columns do not measure those libraries.** Both run OpenFOAM SERIALLY with only
  the pressure equation on the GPU, against the `of` arm's 64 cores -- that is the whole of their 10-25x
  deficit. AMGX as built (`-DCMAKE_NO_MPI=True`) cannot run multi-rank. Rebuild with MPI before
  publishing either column.
- **brae carries its whole cold start; OpenFOAM does not carry `decomposePar`.** brae's timer wraps mesh
  read, device set-up, AMG build, solve and write; the OpenFOAM arm times only the solver. brae's
  start-up is about 0.4 s of a 1.02 s aerofoil run, so the 0.98x and 0.96x rows are mostly set-up.
- `stopped at it 4` on squareBend at 97,470 is the mesh generator: `SNAP` only rounds factors at or
  above 1.5, so sub-native rungs still get non-integer ones.

## spuma builds for Hopper. It is the SDK, not the architecture

Changing one variable at a time, at `-gpu=cc90` with RDC on:

| toolchain | result |
|---|---|
| HPC SDK 26.5 | duplicate `__cudaRegisterLinkedBinary` at device link, build fails |
| **HPC SDK 2026** | **builds clean** |

26.5's device linker emits duplicate registration symbols for cc80 and cc90 under relocatable device
code; 2026 fixes it. cc121 was never affected, which is why the GB10 tree -- only ever built cc121 --
looked like evidence of a path or machine difference, and why an architecture-only experiment concluded
"Blackwell yes, Hopper no". That conclusion was half the variable.

Eliminated first, so nobody retries them: build parallelism (identical at `-j 48/16/8`, not a race),
linker version (lld 14 vs 18, installed 18.1.8, unchanged), CUDA pairing (identical), source path length
(rebuilt at the GB10's exact 69 characters, collision returned), and `-gpu=cc90a` (nvc++ 26.5 rejects the
`a` suffix, so its "zero collisions" was a build that never reached a device link).

**Why 26.5 was wrongly pinned**: an early 2026 attempt failed and was read as "2026 does not work". It
was the LINKER -- GNU ld overflows the aarch64 GOT linking `libOpenFOAM.so`, under both SDKs. Two faults
were varied at once and the wrong one blamed. Fixing the linker first, then varying the SDK, separated
them.

Recipe in `bench/rhoSimpleFoam/spuma_build_hopper.sh`. Three things must hold: SDK 2026; lld for
`LINKLIBSO` only (GNU ld for executables -- lld rejects the `--add-needed` wmake passes); and RDC left
ON, since `-gpu=nordc` completes the build and produces a binary with no device code in it.

## Every spuma number here is measured

`spuma_extrapolation/` held a script projecting the walls of rungs too slow to finish. It was never
committed and was removed from the tree on 2026-09-15; the `.gitignore` entry is kept as a guard.

Consequence: **spuma's aerofoil column stops at 1,024,000 cells.** At 10M it runs about 248 s per
iteration -- 100 of them is roughly seven hours, the 24.3M rung about seventeen -- so both were stopped.
Those cells are EMPTY, not projected. An empty cell says "not measured", which is true.

All 20 rows in `gh200_csv/spuma_tuned_gh200.csv` are completed runs, `iters_done` 100 of 100. One reads
`ok_from_log`: squareBend at 24,192,000, 7,413 s -- it finished, but the harness did not time it
cleanly, so the wall came from the solver's own `ExecutionTime`. A measurement of a completed run, not
an extrapolation of an incomplete one, marked differently because it was obtained differently.

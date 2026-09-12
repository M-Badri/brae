# rhoSimpleFoam: the six shipped tutorials, brae's OF-mirror (CUDA arm) vs OpenFOAM v2412 on 20 Grace cores -- GB10, 2026-09-12

The six `compressible/rhoSimpleFoam` tutorials OpenFOAM v2412 ships, as shipped and as brae runs them
in `tests/rho_tutorials_vs_openfoam.sh` (all six run on both mirror arms since commit a5266d0), timed
under the rules of `../rhoSimpleFoam/run_benchmark.sh`: 100 fixed SIMPLE iterations, wall time of the
solver run only (each tutorial's own Allrun.pre / blockMesh / snappyHexMesh, decomposePar and
reconstructPar excluded), residualControl removed and the `functions` block stripped on BOTH sides
(the sampling function objects would not be paid equally), everything else the tutorial's own --
solvers, tolerances, schemes, boundary conditions, thermo. brae is commit a5266d0's
`build/brae_rhoSimpleFoam` with `BRAE_RHOSIMPLEFOAM_MIRROR=cuda`, timed as a whole process; OpenFOAM
is `mpirun -np 20 rhoSimpleFoam -parallel` (`nProcs : 20` in every log), decomposed `hierarchical`
with the tutorial's own layout scaled to 20 (`(5 2 2)` for the bends and injectorPipe, `(10 1 2)` for
the 2D aerofoil), `scotch` for angledDuct which ships no decomposeParDict. Every arm completed its
iterations (`Time =` lines counted in each log). One GPU; the machine otherwise idle (a resident
python process held 228 MiB of GPU memory at 0% utilisation throughout). Driver:
`tut_bench.sh` in the session scratch, a transcription of the tutorial gate's mesh() and stage().

## 100 iterations, and iterations 101-200 alone

The 100-iteration wall includes each code's start-up, which on meshes this small is a large share
(brae: mesh load, device set-up, AMG hierarchy; OpenFOAM: MPI start-up, field reads). A second run at
200 iterations gives the cost of iterations 101-200 with the start-up removed -- the per-iteration
number.

| tutorial                     |   cells | model                | brae 100 it | OF-20c 100 it | whole-run, brae is | brae ms/it | OF-20c ms/it | per iteration, brae is |
|------------------------------|--------:|----------------------|------------:|--------------:|-------------------:|-----------:|-------------:|-----------------------:|
| aerofoilNACA0012             |  16,000 | kOmegaSST            |       1.9 s |         1.7 s |              0.89x |         13 |            8 |                  0.62x |
| angledDuctExplicitFixedCoeff |  28,000 | kEpsilon             |       1.4 s |         1.3 s |              0.93x |          9 |            7 |                  0.78x |
| squareBend                   | 112,000 | kEpsilon             |       2.7 s |         3.6 s |              1.33x |         21 |           32 |                  1.52x |
| squareBendLiq                | 112,000 | kEpsilon, liquid     |       4.5 s |         3.9 s |              0.87x |         33 |           23 |                  0.70x |
| squareBendLiqNoNewtonian     | 112,000 | generalizedNewtonian |       2.6 s |         2.8 s |              1.08x |         17 |           22 |                  1.29x |
| gasMixing/injectorPipe       |  74,650 | kEpsilon             |       3.1 s |         2.9 s |              0.94x |         21 |           20 |                  0.95x |

(200-iteration walls: aerofoil 3.2 / 2.5, angledDuct 2.3 / 2.0, squareBend 4.8 / 6.8, squareBendLiq
7.8 / 6.2, squareBendLiqNoNewtonian 4.3 / 5.0, injectorPipe 5.2 / 4.9, brae / OpenFOAM. Clock variance
on this machine is about 7% run to run, measured on the size sweep's scale-1 rerun.)

Reading it. These are small meshes: the largest is the 112k-cell squareBend that the size sweep in
`rhoSimpleFoam_squareBend_gb10.md` starts from, and the sweep is where the GPU's advantage grows with
size (1.6x at 112k, 1.3-1.4x at 900k-3M whole-run, 2.0x per iteration at 896k). Below ~30k cells the
GPU is latency-bound -- a 16,000-cell 2D aerofoil costs brae 13 ms per iteration where 20 cores take 8
-- and the whole-run times of every tutorial except the two liquid bends are within 15% of each other
either way. Two cases stand out and both are explained by the same measurement below: squareBendLiq
is the slowest relative case (0.70x per iteration), squareBend the fastest (1.52x), on the SAME mesh
with the SAME closure.

## Where squareBendLiq's iteration goes, against squareBend on the same 112,000 cells

`BRAE_PHASE_TIME=1`, 100 iterations, ms per iteration:

| phase        | squareBend | squareBendLiq | squareBendLiq, BRAE_GS_HOST_SMOOTHER=0 |
|--------------|-----------:|--------------:|---------------------------------------:|
| UEqn         |        3.4 |           3.4 |                                    3.8 |
| EEqn         |        2.3 |           9.8 |                                    9.6 |
|   he solve   |        1.0 |           6.1 |                                    4.7 |
| pEqn         |        8.0 |           6.6 |                                    6.8 |
| turbulence   |        4.3 |          13.3 |                                    9.8 |
| four phases  |       18.0 |          33.2 |                                   30.0 |

The momentum and pressure blocks are the same or cheaper on the liquid case (its p is subsonic,
AMG-preconditioned CG, against squareBend's transonic AMG-preconditioned BiCGStab). The 15 ms it adds
are the energy and turbulence SOLVES, and the reason is the linear solver the case names: squareBend
asks `GAMG` on `(U|e|k|epsilon)`, which brae substitutes (announced) with a device BiCGStab -- diagonal
on e, a degree-22 Neumann series on k and epsilon -- while squareBendLiq asks `smoothSolver` with
`symGaussSeidel` on the same fields, and brae's default for a GaussSeidel-family smoothSolver on e, k,
epsilon or omega is OpenFOAM's own sweep executed on ONE CPU thread (`smoothSolver: host smoother
(OpenFOAM's sweep on the CPU, 1 thread(s))` in the log; device_amg_gauss_seidel.cu:733-741, the matrix,
rhs and field downloaded per solve and psi re-uploaded per pass). `BRAE_GS_HOST_SMOOTHER=0` moves the
sweep to the device loop and takes the iteration from 33.2 to 30.0 ms (whole run 4.5 to 3.9 s at 100
iterations, 7.8 to 6.8 at 200; per iteration 33 to 29 ms, 0.79x of OpenFOAM); the device
Gauss-Seidel is still 2.3x the cost of the BiCGStab path on the turbulence block (9.8 against 4.3),
because a sequential smoother is a poor fit for the GPU whichever side runs it. The same switch on the
other two smoothSolver tutorials changes nothing measurable: angledDuct 1.4 to 1.3 s (9 to 10 ms/it),
injectorPipe 3.1 to 2.7 s at 100 and 5.2 to 4.7 at 200 (21 to 20 ms/it). What the liquid case's EEqn
also carries, which squareBend's does not: the he-to-T inversion is OpenFOAM's Newton loop per cell,
the energy boundary coefficients are rebuilt from the live p and T at every assembly, and its T walls
are the `expression` PatchFunction1 evaluated on the host from the patch's downloaded cells
(rhoSimpleFoam.cu evaluatePatchExpressions). Those three are in the 9.8 - 6.1 = 3.7 ms of the EEqn
phase outside the solve, against 1.3 on squareBend.

## What is not on the GPU

Answering "is everything CUDA?" for this arm, from the code (2026-09-12 audit, the full list is in
`rhoSimpleFoam_squareBend_gb10.md`): the discretisation, the boundary updates, the thermo, the
closures and every Krylov loop on the default paths run on the device. Per iteration the host still
reads back a few scalars (the he-to-T failure flag twice, three continuity-error reductions, every
solve's residual report, the momentum smoother's stop rule), uploads a vector of ones on SIMPLEC
cases, and -- the one item that costs time on these tutorials -- runs the e/k/epsilon/omega sweeps on
one CPU thread wherever the case names a GaussSeidel-family smoothSolver, which four of the six do
(squareBendLiq, squareBendLiqNoNewtonian, angledDuct on e; injectorPipe on k and epsilon). A case's
flowRateInletVelocity (squareBend's family) adds one blocking reduction per patch twice per iteration,
and an `expression` PatchFunction1 on T (squareBendLiq) adds the host evaluation above.

## Trajectory agreement at the last iteration (not a convergence statement)

rel L2 of U / p at iteration 100, brae against OpenFOAM, both at the tutorial's own tolerances:
aerofoil 5.9e-04 / 3.0e-04, angledDuct 2.5e-04 / 3.4e-05, squareBend 8.1e-04 / 5.0e-04, squareBendLiq
1.5e-02 / 4.2e-04, squareBendLiqNoNewtonian 1.5e-03 / 2.1e-05, injectorPipe 1.3e-02 / 1.6e-06. These
are two codes at different points of their own solver's path at a loose tolerance; the agreement of the
discretisation itself is the tutorial gate's, 1e-11 and below at 1e-14 tolerances over the first three
iterations (`tests/rho_tutorials_vs_openfoam.sh`).

## Where each tutorial's iteration goes (BRAE_PHASE_TIME=1, 100 iterations, ms per iteration)

The four phases per iteration, and the linear solve inside three of them; the last column is the
marginal per-iteration cost from the 200-iteration runs above, so the difference between it and the
four-phase sum is what the iteration spends outside the phases (boundary updates before the momentum
assembly, the thermo, the continuity report, the residual bookkeeping).

| tutorial                     |   cells | model                | UEqn | EEqn | pEqn | turb | U solve | he solve | p solve | four | marginal ms/it |
|------------------------------|--------:|----------------------|-----:|-----:|-----:|-----:|--------:|---------:|--------:|-----:|---------------:|
| aerofoilNACA0012             |  16,000 | kOmegaSST            |  1.1 |  3.1 |  3.8 |  5.7 |     0.5 |      2.2 |     2.9 | 13.7 |             13 |
| angledDuctExplicitFixedCoeff |  28,000 | kEpsilon             |  1.5 |  1.4 |  2.4 |  3.1 |     0.2 |      1.0 |     1.4 |  8.4 |              9 |
| squareBend                   | 112,000 | kEpsilon             |  3.4 |  2.3 |  8.0 |  4.3 |     2.3 |      1.0 |     5.7 | 18.0 |             21 |
| squareBendLiq                | 112,000 | kEpsilon, liquid     |  3.4 |  9.8 |  6.6 | 13.3 |     2.5 |      6.1 |     5.0 | 33.1 |             33 |
| squareBendLiqNoNewtonian     | 112,000 | generalizedNewtonian |  3.7 |  6.5 |  7.4 |  0.8 |     2.6 |      5.0 |     5.6 | 18.4 |             17 |
| injectorPipe                 |  74,650 | kEpsilon             |  4.1 |  3.1 |  6.3 |  9.6 |     3.3 |      2.3 |     4.3 | 23.1 |             21 |

The same four phases per cell -- nanoseconds per cell per iteration -- which is the number that
compares a 16k mesh with a 112k one:

| tutorial                     | UEqn | EEqn | pEqn | turb | four |
|------------------------------|-----:|-----:|-----:|-----:|-----:|
| aerofoilNACA0012             |   69 |  194 |  237 |  356 |  856 |
| angledDuctExplicitFixedCoeff |   54 |   50 |   86 |  111 |  300 |
| squareBend                   |   30 |   21 |   71 |   38 |  161 |
| squareBendLiq                |   30 |   88 |   59 |  119 |  296 |
| squareBendLiqNoNewtonian     |   33 |   58 |   66 |    7 |  164 |
| injectorPipe                 |   55 |   42 |   84 |  129 |  309 |

Read per cell, squareBend (upwind, GAMG entries substituted by device BiCGStab, kEpsilon) is the floor
at 161 ns/cell, and everything above it names a module: the kOmegaSST closure on the 16k aerofoil at
356 ns/cell of turbulence against kEpsilon's 38 (a latency-bound mesh, but 9x per cell); the
smoothSolver cases' energy and turbulence solves on the CPU smoother (squareBendLiq 88 and 119 ns/cell
against squareBend's 21 and 38); injectorPipe's turbulence at 129 ns/cell, which carries leastSquares
gradients, `Gauss limitedLinear 1` on k and epsilon, the Euler ddt, and the host smoother; the liquid
energy phase outside its solve (squareBendLiq and NoNewtonian: the he-to-T Newton inversion per cell,
the live energy boundary, the expression walls). None of those modules has had a speed measurement of
its own yet; this table is where the digging starts.

## FP-1: the scalar smoothSolver entries on the colour engine (2026-09-12)

The first row of `../rhoSimpleFoam/FASTPATH.md`. Where a case names `smoothSolver` with a
GaussSeidel-family smoother on e/h, k, epsilon or omega, the CUDA arm now sweeps that system in COLOUR
order through the momentum engine with one component (deviceColourGaussSeidelFused, nComp 1), under
the entry's own stop rule, instead of OpenFOAM's index order on one CPU thread. The order is announced
per field (`solvers/e smoother: case asks 'symGaussSeidel' in OpenFOAM's index order; brae sweeps in
COLOUR order ...`) and `BRAE_GS_ORDER=ofOrder` restores the previous path; unset, the switch follows
`BRAE_U_SOLVER`, so every gate that pins `ofOrder` for the momentum's exact iterate pins the scalars'
as well. Same binary otherwise, same cases and staging as the tables above, `BRAE_PHASE_TIME=1`, 100
iterations, three runs each; the "before" column is three runs of the same binary the day before the
change (a5266d0).

| tutorial                     | phase        | before, 3 runs (ms/it) | after, 3 runs (ms/it) | opt-out (BRAE_GS_ORDER=ofOrder) |
|------------------------------|--------------|-----------------------:|----------------------:|--------------------------------:|
| squareBendLiq                | he solve     |                    6.1 |             1.3 / 1.3 / 1.3 |                       6.8 |
| squareBendLiq                | EEqn         |         9.8 / 9.6 / 9.5 |             6.5 / 5.7 / 6.6 |                       9.9 |
| squareBendLiq                | turbulence   |      13.3 / 14.6 / 13.9 |             3.9 / 3.9 / 4.1 |                      14.7 |
| squareBendLiq                | four phases  |      33.1 / 34.0 / 33.4 |          21.3 / 20.0 / 21.6 |                      34.6 |
| injectorPipe                 | turbulence   |         9.6 / 9.6 / 9.6 |             4.5 / 4.4 / 4.6 |                       9.9 |
| injectorPipe                 | four phases  |      23.1 / 23.1 / 23.4 |          18.8 / 18.5 / 18.9 |                      23.6 |
| angledDuctExplicitFixedCoeff | he solve     |                    1.0 |             0.2 / 0.2 / 0.2 |                         - |
| angledDuctExplicitFixedCoeff | four phases  |         8.4 / 9.1 / 9.0 |             7.7 / 8.0 / 7.4 |                         - |
| squareBendLiqNoNewtonian     | he solve     |                    5.0 |             1.3 / 1.4 / 1.3 |                         - |
| squareBendLiqNoNewtonian     | four phases  |      18.4 / 18.1 / 18.2 |          15.3 / 15.5 / 15.3 |                         - |

squareBendLiq's iteration drops from 33 to 21 ms, and its energy solve (1.3) and turbulence block
(3.9-4.1) now sit where squareBend's device BiCGStab path puts them on the same mesh (1.0 and 4.3) --
the row's target (he solve <= 1.5, turbulence <= 6) is met. What squareBendLiq still carries over
squareBend is the energy phase outside its solve (5.2 against 1.3 ms/it: the liquid he-to-T inversion,
the live energy boundary and the expression walls, FP-6 and FP-7). injectorPipe's turbulence halves
and its iteration goes from 23 to 19 ms; the two smaller cases lose their host-smoother energy solve
(1.0 and 5.0 ms/it down to 0.2 and 1.3).

Correctness, all on the same day: `tests/scalar_colour_gs_vs_openfoam.sh` on rhoKE at 1e-14 --
EXACT 1.27e-12, REF (ofOrder) 2.28e-12, DEFAULT 1.96e-12 against bound 1e-9, the relTol-0.1 CONTROL
1.3e-02 and the maxIter-1 FAIL-PROOF 1.4e-01 above it, both of OpenFOAM's smoothers exercised
(symGaussSeidel on h, GaussSeidel on the pair); `test_colour_gs_fused` arm (m), one component
bit-identical to the fused solve's component 0; `rho_smoothsolver_vs_openfoam` (angledDuct as shipped,
30 iterations at the tutorial's own relTol), the default arm now e 1.04x, k 1.03x, epsilon 1.28x of
OpenFOAM's residuals against 1.00x / 1.11x / 1.17x when only U took the colour sweep, inside the bounds
it already carried; `u_colour_gs_vs_openfoam`, `rho_patch_expression_vs_openfoam` (squareBendLiq's
walls at 1e-12), `rho_gasmixing_vs_openfoam` and `rho_tutorials_vs_openfoam` unchanged at their bounds.

Whole-run wall under the new default, the same staging and the same OpenFOAM numbers as the first
table (OpenFOAM's side did not change):

| tutorial                     | brae 100 it, before -> after | brae 200 it, before -> after | brae ms/it | OF-20c ms/it | per iteration, brae is (was) |
|------------------------------|-----------------------------:|-----------------------------:|-----------:|-------------:|-----------------------------:|
| angledDuctExplicitFixedCoeff |                 1.4 -> 1.2 s |                 2.3 -> 1.9 s |          7 |            7 |                 1.0x (0.78x) |
| squareBendLiq                |                 4.5 -> 3.3 s |                 7.8 -> 5.0 s |         17 |           23 |                 1.35x (0.70x) |
| squareBendLiqNoNewtonian     |                 2.6 -> 2.3 s |                 4.3 -> 3.8 s |         15 |           22 |                 1.47x (1.29x) |
| gasMixing/injectorPipe       |                 3.1 -> 2.6 s |                 5.2 -> 4.5 s |         19 |           20 |                 1.05x (0.95x) |

The slowest tutorial relative to 20 cores, squareBendLiq, goes from 0.70x to 1.35x per iteration; the
two smoothSolver cases that were behind are at parity or ahead. squareBend and the aerofoil name no
smoothSolver on their scalars and are untouched.

## FP-2, first lever: the DILU walk on the aerofoil is the cost, not the closure (2026-09-12)

The second tracker row asked why kOmegaSST costs 356 ns per cell on the 16k aerofoil. An nsys profile
(20 iterations, `--cuda-graph-trace=node` so the kernels inside the solver graphs are counted, NVTX
phase ranges from `BRAE_PHASE_NVTX=1`) attributes every kernel to its phase:

| phase      | launches per iteration | of which DILU level kernels | GPU ms/it | wall ms/it (phase timer) |
|------------|-----------------------:|----------------------------:|----------:|-------------------------:|
| turbulence |                   1081 |                         862 |       2.9 |                      5.8 |
| EEqn       |                    539 |                         449 |       1.5 |                      3.2 |
| pEqn       |                   1794 |                           0 |       3.7 |                      3.9 |
| UEqn       |                    117 |                           0 |       0.5 |                      1.1 |

The case names `PBiCGStab` with `preconditioner DILU` on U, k, omega and e, so k, omega and e each run
brae's level-scheduled DILU (121 levels on this mesh): one kernel per level per half-sweep, three kernels
per level per solve counting the factorisation, and the launch gaps between 862 tiny kernels are half the
turbulence phase's wall. The SST physics itself is about 220 launches and 0.9 ms per iteration.

brae already had a single-block walk of the same levels (item 70, bit-identical, `__syncthreads` between
levels), selected by a mean-level-width rule of 128 measured on two points before the solver loops went
into graphs; this mesh reads 132 and took the per-level path. Re-measured end to end with DILU on every
field and both walks forced, the four phases in ms per iteration:

| case             |   cells | levels | mean width | widest | per-level | single-block |     |
|------------------|--------:|-------:|-----------:|-------:|----------:|-------------:|-----|
| aerofoilNACA0012 |  16,000 |    121 |        132 |    200 |      14.1 |         10.4 | win |
| squareBend x0.7  |  38,416 |    187 |        205 |    392 |      36.0 |         22.0 | win |
| squareBend x1    | 112,000 |    268 |        417 |    800 |      57.5 |         47.9 | win |
| squareBend x1.4  | 307,328 |    376 |        817 |   1568 |     121.5 |        135.4 | loss |
| squareBend x2    | 896,000 |    538 |       1665 |   3200 |     257.4 |        396.1 | loss |

The rule is now 512 (the crossover lies between 417 and 817) and the widest-level guard is gone, since
the block strides a level in chunks. Both walks compute the same bits, which `dilu_single_block_identity`
holds on three fixtures; `dilu`, `dilu_vs_openfoam`, `transonic_p_dilu`, `rho_sbmatched_transient`,
`rho_sst_device`, `rho_komegasst` and `rho_tutorials` all pass unchanged on the new default.

Aerofoil under the new default, three runs of 100 iterations:

| phase       | before (3 runs) | after (3 runs) |
|-------------|----------------:|---------------:|
| turbulence  |   5.7 / 5.9 / 5.8 |  3.4 / 3.2 / 3.4 |
| EEqn        |   3.1 / 3.3 / 3.3 |  2.1 / 2.0 / 1.9 |
| four phases | 13.7 / 14.4 / 14.1 | 10.6 / 10.0 / 10.3 |

Whole run 1.9 -> 1.5 s at 100 iterations and 3.2 -> 2.4 at 200, so 9 ms per iteration against
OpenFOAM's 8 on 20 cores (0.89x, from 0.62x) and 1.13x on the 100-iteration wall (from 0.89x). The
closure's own launches are the next step on this row, worth about 1 ms per iteration here.

## FP-2, second step: the closure's own launches, fused where the chains share a loop (2026-09-12)

With the DILU walk single-block, the aerofoil's turbulence phase profiled at 230 launches per
iteration, 2.26 ms of GPU time and 3.26 ms of wall: the three DILU applies and the two factorisations
about 1.7 ms of the GPU time, the SST physics about 0.5 ms behind roughly 1 ms of launch gaps. Four
chains were fused, each kernel keeping the exact expression text of the kernels it replaces so nvcc
contracts the same multiply-adds, with `__dmul_rn` pinning the products that used to cross a kernel
boundary: S2 + production + G (3 launches to 1), CDkOmega + F1 + F2 (3 to 1), the gamma/beta blends +
the GbyNu limit (3 to 1), and the compressible effective diffusivity DEff*rho + nu*rho on cells and on
boundary faces (4 to 1, three sites). In the shared transport assembly the five `axpy(-1, l, M)`
laplacian subtractions became one launch, which the kEpsilon closure takes as well.

Bit-identity, the old kernels against the fused ones on the same binary line: aerofoilNACA0012 and
squareBend, 5 iterations, every written field byte-identical (U, p, T, k, omega/epsilon, nut) and every
residual line identical; all 18 SST stage dumps (`BRAE_SST_DUMP_DIR`: G, CD, F1, F23, GbyNuLim, S2,
gradU, ...) byte-identical. injectorPipe's residual lines are identical and its written fields are
not, but that case writes different low bits between two runs of ONE binary as well (the D-1 drift
noted with squareBendLiq), so it cannot serve as a bit-identity witness.

| aerofoil, turbulence phase | before | after |
|----------------------------|-------:|------:|
| launches per iteration     |    230 |   205 |
| GPU ms/it (profiled)       |   2.26 |  2.25 |
| wall ms/it (profiled)      |   3.26 |  3.10 |
| wall ms/it, 3 plain runs   | 3.2 / 3.4 / 3.4 | 3.1 / 3.2 / 3.4 |
| four phases, 3 plain runs  | 10.6 / 10.0 / 10.3 | 9.9 / 10.1 / 10.4 |

Inside the run-to-run noise, as the profile predicted: 25 launches at ~4 us is 0.1 ms. The honest
reading of FP-2 after both steps is that the kOmegaSST closure's kernels are not where its time goes.
On this case the turbulence phase is the two DILU-preconditioned BiCGStab solves the fvSolution names
(sequential by construction, 1.7 ms of the 2.25), and the iteration as a whole is the pressure phase's
1783 launches of the AMG V-cycle on a 16,000-cell mesh (FP-10 / FP-12). The row's target of 2x kEpsilon
per cell (76 ns) is not met at 194 ns and cannot be met without a decision about the preconditioner on
`PBiCGStab`+`DILU` entries -- the Neumann series that already serves the GAMG entries, announced, with
an opt-out -- which is a policy change and is left as the row's open question. Gates on the fused
kernels: rho_komegasst_vs_openfoam, rho_sst_device_vs_openfoam, rho_step_cuda_turbulent,
rho_step_cuda_euler, rho_kepsilon_cuda, turb_limitedlinear_vs_openfoam, rho_turb_limitedlinear_vs_openfoam,
rho_gasmixing_vs_openfoam, rho_naca_restart_vs_openfoam, rho_tutorials_vs_openfoam -- all at their bounds.

## D-1: the run-to-run drift was Foam::bound's atomic scatter, now a gather in OpenFOAM's order (2026-09-12)

squareBendLiq and gasMixing/injectorPipe wrote different low bits between two runs of ONE binary
(squareBendLiq: U 1.3591e-01 against 1.3552e-01 at iteration 20, iteration 1 identical), where
squareBend on the same mesh was bit-identical. Localised with the stage dumps rather than by bisecting
switches: the first differing stage on squareBendLiq is `epsOut` at iteration 4, with `epsSolveOut`
identical -- so the epsilon solve agrees to the bit and Foam::bound between the two disagrees. The
average it takes, fvc::average(max(psi, lowerBound)), was two atomicAdd scatters (faces to owner and
neighbour, then boundary faces), which sum a clamped cell's faces in whichever order the hardware
serialises them; a cell that solved negative takes that sum as its value, and from there the two runs
diverge. squareBend never clamps a cell so it never showed it. The `FP32=0 identical` pairs in the
FP-2 table were trajectories that happened not to clamp, not evidence.

The average is now one gather per cell in OpenFOAM's own order (fvcSurfaceIntegrate.C:30-46: faces
in increasing index, owner faces from ownerStart and neighbour faces through losort merged by face
index, then the boundary faces through bndCellStart/bndPerm), the sum in the same sequence the host
reference `bound_cpp.cu` takes. `BRAE_BOUND_SCATTER=1` keeps the old scatter as the gate's control.

`tests/rho_run_to_run_identity.sh` (LABELS slow): squareBendLiq, injectorPipe and squareBend, two
20-iteration runs each, every residual line and every written field (U p T k epsilon omega nut)
byte-identical; under the scatter control squareBendLiq's two runs differ at iteration 7 (six of six
pairs differed before the fix). The eight closure gates (rho_kepsilon_cuda, turb_precon,
rho_sst_device, rho_gasmixing, rho_tutorials, rho_patch_expression, rho_komegasst,
rho_step_cuda_turbulent) pass at their bounds; the gather is one launch where the scatter was two and
is not on any phase's clock. injectorPipe can now witness bit-identity for the next fusion.

## FP-2, the decision: a relaxed `PBiCGStab`+`DILU` entry takes the Neumann series (2026-09-12)

The row's open question, answered by measurement. On the CUDA arm a `preconditioner DILU` entry on k,
epsilon/omega or the energy field now takes the truncated Neumann series wherever fvMatrix::relax
bounds it -- the same derivation the pair already takes on a GAMG entry (`neumannDegreeIfRelaxed`,
degree d = ceil(ln 0.1 / ln alpha), capped at 24) -- announced per field as `[approximated]`, with
`BRAE_DILU_KE=1` and `BRAE_DILU_HE=1` keeping DILU. An entry with no relaxation bound keeps DILU
regardless. Same linear system, same tolerance, a different iterate where the tolerance leaves one
free; the momentum sweep's trade, stated the same way.

Aerofoil, three runs of 100 iterations, the same staging as before (the tutorial's own tolerances):

| phase       | DILU kept (3 runs) | series (3 runs) |
|-------------|-------------------:|----------------:|
| turbulence  |    3.2 / 3.2 / 3.2 |  1.8 / 1.6 / 1.7 |
| EEqn        |    1.9 / 1.9 / 1.9 |  1.4 / 1.3 / 1.2 |
| he solve    |    1.1 / 1.1 / 1.1 |  0.3 / 0.3 / 0.3 |
| four phases | 10.1 / 10.2 / 10.0 |  8.4 / 7.7 / 7.8 |

Against 20 cores' 8 ms per iteration that is 1.0x per iteration (from 0.62x when the row opened and
0.89x after the walk rule). At iteration 100 the two arms' residual lines agree to two digits (U
2.506e-03 against 2.503e-03, k 5.34e-04 against 5.20e-04): the series stops the solve elsewhere under
relTol 0.1 and that is all it moves.

sbMatched at ITS pinned solvers (tolerance 1e-12, relTol 0; relaxation k, epsilon 0.9 and e 0.8, so
degrees 22 and 11), 20 iterations: turbulence 13.5 ms/it against DILU's 79, EEqn 5.1 against 30, the
iteration-20 residual line identical to five digits. So the series is not a loose-relTol trick: at a
tight tolerance the DILU walk's 112k-cell level schedule is what BiCGStab pays on every iteration.

Gate: `tests/rho_dilu_entry_policy_vs_openfoam.sh` -- the aerofoil, 30 iterations at its own tolerances,
series against DILU-kept against real OpenFOAM: the notices (and their absence under the hatches,
with the DILU walk built only where something asks for it); k, omega and e initial residuals at
iteration 30 within [0.9, 1.2] of OpenFOAM's for the series (measured 1.07, 1.04, 1.03) and [0.9, 1.1]
for DILU kept (1.00, 1.00, 1.01); no `bounding` line on either arm; the series' turbulence phase under
0.8x DILU's (0.59x). Both arms are deterministic since D-1, so the bounds are the measurement rounded
outward.

Four gates pin DILU with the hatches, and say why in their headers. `rho_sbmatched_transient` and
`rho_gradp_lsq_simplec` hold the ASSEMBLY at 1e-10 with the solvers pinned to 1e-12 / 1e-14, and at
those tolerances the residual leaves the iterate free at about 1e-9 (the series lands k 5.4e-09,
epsilon 3.3e-09 from OpenFOAM's; DILU, OpenFOAM's own algorithm, 8e-12). `eeqn_limitedlinear` measures
a SCHEME against OpenFOAM at 1e-10 while validation/rhoLU's own relTol is 0.1, where the series reads
2.33e-06 and DILU 1.30e-10. `dilu_single_block_identity` compares the two DILU WALKS, and its sbMatched
and rhoBox arms had gone vacuous -- both arms agreeing because neither built a walk any more; its own
"walked one block" check caught that, which is what such a check is for. Every bound is unchanged.

AND THE POLICY FOUND A DEFECT IN THE SOLVER, which is the reason to put a second field on the series.
PBiCGStab has two loops -- a device conditional-graph loop and the host loop it falls back to when
checkEvery is above 1, `BRAE_BICG_HOST_LOOP=1` or `BRAE_NORMFACTOR_HOST=1` is set, or the graph
declines -- and the fallback call forwarded every argument except `polyDeg`. A solve that took it ran
the bare DIAGONAL where the caller asked for a degree-d series. Nothing in a residual line shows it:
the solve still reaches the case's tolerance, it just stops somewhere else. It surfaced because
`normfactor_device_identity` holds the two normFactor paths byte-identical, and once the energy solve
took the series its rhoBox arm diverged from iteration 2 (49 of 51 lines). Fixed by forwarding the
degree; `tests/bicg_polydeg_host_loop.sh` now holds the same case identical through the graph loop,
the host loop and the host-read normFactor, with the case's own DILU as the control that proves the
preconditioner moves the iterate there. Fail-proof RUN: dropping the degree again turned both gates
red. The turbulence pair has taken this series since the flat-plate fix, so any of its solves that
fell back ran the diagonal too.
The rest ran on the new default: `ctest -R rho -LE slow` (72), rho_sst_device, rho_naca_restart,
rho_tutorials, rho_run_to_run_identity, sa_precon, turb_precon, precon_policy_one_rule -- all pass.

## FP-3: the leastSquares gradient -- the dd tensor is the mesh's, and one fit carries three fields (2026-09-12)

nsys on gasMixing/injectorPipe at 74,650 cells (`--cuda-graph-trace=node`, NVTX phases, 20 iterations)
put 3.6 of the iteration's 15.8 GPU ms in the leastSquares gradient: ten `lsqInvDdKernel` launches at
1.99 ms and ten `lsqGradKernel` at 1.57. The first number is the finding. OpenFOAM builds
leastSquaresVectors ONCE per mesh (a MeshObject, invalidated on a move); brae rebuilt the inverted dd
tensor inside every gradient call, and that tensor depends only on the geometry -- the d vectors, the
weights, |Sf| and the empty-patch skip -- never on the field being fitted.

Three levers, each bit-identical by construction:

- The tensor is cached on the DeviceMesh (`lsqInvDdFor`), built on first request and dropped in
  `refreshDeviceMeshGeometry`, which is where OpenFOAM's MeshObject is invalidated too. Ten launches
  per iteration became one per mesh. `BRAE_LSQ_INVDD=recompute` restores the old rebuild.
- Up to three fields are fitted in ONE launch (`deviceLeastSquaresGradFused`), the least-squares twin
  of `deviceGaussGradFused`: the same three face loops in the same order, each field's sum in its own
  registers, the shared operands (d, |Sf|/|d|^2, the cell's (invDd & d)) read once. A raw form writes
  into slices the caller owns, so `deviceLeastSquaresGradU` now fills the 9*nC grad(U) tensor in one
  launch where it took three fits, nine device-to-device copies and THREE `cudaStreamSynchronize` --
  one per component, each draining the GPU pipeline. The momentum assembly's four component loops and
  divDevRhoReff's take the same path.
- Each scalar assembly evaluates a field's boundary values and its gradient ONCE per (base scheme,
  cellLimited coefficient) pair it asks for. The limitedLinear limiter, linearUpwind's correction and
  the corrected laplacian all read the case's `grad(<field>)` entry, so the energy assembly fitted
  grad(he) twice and each closure fitted its field twice, per iteration.

| injectorPipe, 74,650 cells | before | after |
|----------------------------|-------:|------:|
| lsqInvDdKernel per iteration | 10.0 (1.99 ms) | 0.1 (0.01 ms) |
| lsqGradKernel per iteration  | 10.0 (1.57 ms) | 7.0 (1.50 ms) |
| device-to-device copies      | 192.5 (0.74 ms) | 189.5 (0.56 ms) |
| GPU ms per iteration         | 15.79 | 13.73 |
| stream syncs in the gradient | 3 per grad(U) | 0 |

Wall, three runs of 100 iterations each (BRAE_PHASE_TIME=1, the tutorial as it ships):

| phase       | before (3 runs) | after (3 runs) |
|-------------|----------------:|---------------:|
| UEqn        | 4.3 / 4.3 / 4.3 | 4.2 / 4.2 / 4.2 |
| EEqn        | 3.3 / 3.4 / 3.4 | 2.8 / 2.8 / 2.8 |
| pEqn        | 6.9 / 7.2 / 7.0 | 6.8 / 6.8 / 6.7 |
| turbulence  | 4.5 / 4.4 / 4.5 | 3.6 / 3.5 / 3.6 |
| four phases | 18.9 / 19.2 / 19.1 | 17.3 / 17.3 / 17.2 |

BIT-IDENTITY, end to end: a reference binary built from the working tree with the seven FP-3 files
reverted to HEAD, against the new one, 20 iterations of injectorPipe (leastSquares everywhere) and of
squareBend (Gauss, where the shared-gradient lever is live and the fused fit is not): every residual
line identical and all nine written fields byte-identical on both cases. Unit gate
`tests/test_lsq_grad_fused.cu` (ctest `lsq_grad_fused`) holds the fused fit to memcmp against
`deviceLeastSquaresGrad` for n = 1, 2 and 3 on a sheared box and on an empty-patch variant, with the
raw form, the one-ulp cross-contamination controls, the empty-face skip and the cache's identity and
its per-mesh control. Its fail-proof was RUN: reading field 0's boundary values for every field turned
10 arms red and the exit code to 1.

THE ROW'S TARGET IS NOT MET AND THE REASON IS NOT THE GRADIENT. FP-3 asked for UEqn at or under 40
ns/cell on injectorPipe; it reads 56 (4.2 ms at 74,650 cells), down from 58. That case names
`grad(U) cellLimited Gauss linear 0.99`, so the fused least-squares fit never fires in the momentum
phase at all: what the phase spends is 1.9 GPU ms of `cellLimitGradKernel` (12 launches) and the
colour sweeps. The limited assembly is FP-4's row, and this measurement is the evidence for it. What
FP-3 did reach -- the energy and turbulence phases, where the scalar fits live -- came down 15% and
20%.

## The whole suite, after all three changes (2026-09-12)

`ctest -j 4`, 472 tests, on a tree rebuilt from scratch first -- the earlier round reported a false
failure from a `build/brae` five hours older than the sources, which is the trap CLAUDE.md names. 462
pass. Of the ten that do not, `dilu_single_block_identity` was the policy going vacuous (fixed above,
now green) and the other nine fail IDENTICALLY at HEAD, checked by building a reference tree with every
file this session touched reverted: `liquid_correct`, `etot`, `hetot`, `linear_solver_setup`,
`uniform_function1`, `mean_velocity_force`, `eval_scoped`, `pimple_loop_contract` (unit tests, all
aborting or failing the same way at HEAD) and `coupledinterfacescheme_vs_openfoam`, whose pipeCyclic
run diverges to a non-finite residual at iteration 33 with the HEAD binary exactly as with this one.
They are open defects on this branch, none of them in what was changed here.

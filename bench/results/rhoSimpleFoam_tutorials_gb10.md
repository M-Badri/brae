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

## FP-4: the cellLimited limiter is a third of the momentum phase, and it is the scheme's own cost (2026-09-12)

What the momentum phase spends on gasMixing/injectorPipe (74,650 cells), three runs of 100 iterations
each, the tutorial's schemes against the same case with pieces removed:

| arm                                            | UEqn ms/it | four phases |
|------------------------------------------------|-----------:|------------:|
| the tutorial (limitedLinearV, cellLimited 0.99) | 4.0 / 4.1 / 4.2 | 17.3 / 17.5 / 17.6 |
| grad(U) unlimited, everything else the same     | 2.7 / 2.7 / 2.7 | 15.7 / 15.1 / 15.2 |
| upwind divergence, gradients untouched          | 3.6 / 3.6 / 3.6 | 16.6 / 16.4 / 16.9 |
| upwind and unlimited Gauss                      | 2.5 / 2.5 / 2.5 | 14.4 / 14.4 / 14.5 |

So the cellLimited limiter alone is 1.4 ms of the phase, the limited divergence 0.5, and the limited
assembly costs 1.64x the upwind one. The row asked for 1.5x.

`deviceCellLimitGradFused` limits up to three fields in one launch, the third of this family after the
Gauss gradient and the leastSquares fit, bit-identical per field. `tests/test_cell_limit_grad_fused.cu`
(ctest `cell_limit_grad_fused`) holds it to memcmp for n = 1, 2 and 3, at k = 1 and at k = 0.5 (the
widening branch), on a sheared box and an empty-patch variant, with one-ulp cross-contamination
controls both ways and a control asserting the limiter bites at all. Its fail-proof was run: reading
field 0's range for every field turned 10 arms red. End to end, a reference binary with the FP-4 files
at HEAD writes byte-identical fields and residual lines on injectorPipe, aerofoilNACA0012 and
squareBend.

IT DID NOT MOVE THE CLOCK, and that is the row's finding. Launches per iteration went 12 to 8 and their
GPU time 1.90 to 1.74 ms, inside run-to-run noise on a 13.7 ms iteration; the aerofoil is unchanged too
(UEqn 1.1 ms/it, four phases 8.2 to 8.4 either way). The limiter's cost is its SIX face-loop passes
over scattered neighbour values, and fusing N fields removes none of them -- only the re-reads of the
addressing, which the earlier two fusions had shown to be worth 3x when the passes themselves were the
traffic.

TWO FURTHER LEVERS WERE MEASURED AND REJECTED, which is the useful part of the row:

- Occupancy. The fused kernel takes 92 registers against the single-field kernel's 58, which halves the
  blocks per multiprocessor. Capping it to 80 with `__launch_bounds__` moved 0.780 ms to 0.771. Not the
  constraint; the hint was removed rather than left in place looking like a decision.
- Pass sharing. The gradient's three face passes and the limiter's range pass read the SAME values at
  the same faces, so a kernel was written that does both in one pass, nine passes down to six. It was
  held bit-identical to gradient-then-limiter (memcmp, n = 1, 2, 3, k = 1 and 0.5, both meshes) and
  measured: the two sites it served went 1.026 to 0.999 ms per iteration, 0.04 of 13.7. The second pass
  was already L2-resident, so sharing it bought the loop overhead and nothing else. Reverted rather than
  kept as 200 lines of duplicated arithmetic that would have to stay bit-identical to two other kernels
  forever.

The honest close: the limited assembly is 1.64x upwind's and the remaining gap is the scheme's own
face-gather traffic, not brae's packaging of it. The row's target is not met and no lever in this row's
list will meet it. What is still on the table for this case is elsewhere: the pressure phase is 6.35 of
the iteration's 13.6 GPU ms (FP-12), and the leastSquares fits of the scalar fields are 1.51 in seven
single-field launches that no site can group.

## Where every tutorial stands after FP-1 to FP-4 (2026-09-13)

100 iterations, `BRAE_PHASE_TIME=1`, the tutorials as they ship. The OpenFOAM-20-core figures are the
recorded ones from the campaign table above; OpenFOAM has not changed.

| tutorial                     | UEqn | EEqn | pEqn | turb | four phases | OF-20c ms/it |
|------------------------------|-----:|-----:|-----:|-----:|------------:|-------------:|
| aerofoilNACA0012             |  1.1 |  1.3 |  3.9 |  1.7 |         8.1 |            8 |
| angledDuctExplicitFixedCoeff |  2.0 |  0.9 |  2.6 |  1.3 |         6.7 |            7 |
| squareBend                   |  3.2 |  2.2 |  8.6 |  4.1 |        18.2 |           32 |
| squareBendLiq                |  3.7 |  5.8 |  7.8 |  4.0 |        21.2 |           23 |
| squareBendLiqNoNewtonian     |  3.7 |  3.3 |  8.4 |  0.9 |        16.2 |           24 |
| injectorPipe                 |  4.2 |  2.9 |  6.5 |  3.7 |        17.2 |           20 |

The pressure equation is the largest phase on every one of them, 37% to 52% of the four, and the solve
inside it is 1.4 to 5.7 ms per iteration. That is why FP-12 is the next row and not squareBendLiq's
energy phase, which is larger in one place (5.8 against squareBend's 2.2) but only in one place.

## FP-12, first findings: the V-cycle is 95% overhead, and the two obvious dials make it worse (2026-09-13)

gasMixing/injectorPipe, 74,650 cells, 11 levels, about 15 V-cycles per iteration (one pressure solve,
15 PCG iterations at the case's relTol 0.05). nsys with `--cuda-graph-trace=node` and NVTX phases: the
pressure phase is 1,814 launches and 6.57 GPU ms, so 121 kernels per V-cycle, and every one of them is
already inside a CUDA graph. The FP32 SpMV is 2.60 ms of it:

| level (cells) | SpMV us per launch | elementwise kernel on the same grid |
|---------------|-------------------:|------------------------------------:|
| 74,650        |              10.05 |                        1.09 (zeroT) |
| 2,304         |               5.51 |                                0.77 |
| under 1,000   |               4.20 |                                0.76 |

The finest level is 32 times larger than the 2,300-cell one and only 1.8 times slower, and the coarse
SpMV costs five times what an elementwise kernel costs on the same grid. The V-cycle's arithmetic is
trivial -- three fine-level SpMVs move about 6 MB, some 12 us at this GPU's bandwidth, against 290 us
of measured cycle. It is a per-kernel floor multiplied by 121.

FOUR LEVERS MEASURED, three of them rejected:

- Coarsest size (the row's first lever). Sweeping `BRAE_AMG_TARGET` over 32, 100, 200, 500, 1000, 2000
  and 5000: injectorPipe's p solve reads 4.5, 4.2, 5.8, 8.4, 12.7, 23.9, 34.7 ms/it and squareBend's
  6.2, 5.6, 11.4, 32.6, 74.8, 77.6, 56.3. The default of 64 is right and the direct coarse solve is
  what punishes a bigger coarsest level. Fewer levels is not the way out.
- Launch shape. At 256 threads a 1,000-cell level is four blocks, so four multiprocessors do the work.
  Sizing the block to the level (32/64/128 by cell count) is bit-identical -- same thread-per-cell
  arithmetic, same sum order -- and was measured: SpMV 2.602 to 2.533 ms on injectorPipe, and nothing
  on squareBend (pEqn 8.6 to 8.6-8.9). Reverted; the measurement is in the comment where the next
  person will look.
- The Gauss-Seidel smoother (`BRAE_AMG_GS`): injectorPipe's p solve 4.4 to 13.0 ms/it. Three times
  worse.
- Smoothed aggregation (`BRAE_AMG_SA`), which the row named: the SOLVE gets 25% faster, 4.4 to 3.3
  ms/it, and the phase gets worse, 7.1 to 12.8, because its RAP setup runs on every SIMPLE iteration.
  It also scatters with atomics, so it is not deterministic (the D-1 class of defect), and squareBend
  diverges under it. Not a default as it stands. What it does say is that a better prolongation is
  worth 25% of the solve IF its setup can be hoisted off the per-iteration path.

For reference on the same case, the FP32 V-cycle that is already the default is worth 0.8 ms/it
(`BRAE_AMG_FP32=0` reads p solve 5.2 against 4.4).

THE LEVER THIS ROW POINTS AT, with its prize measured: fuse the coarse hierarchy into ONE kernel, the
way `device_dilu.cu` walks its levels in a single block. The levels at or below 4,096 cells contribute
1.24 ms of SpMV, roughly 0.5 ms of smoother, restriction, prolongation, residual and zeroing, and the
0.55 ms coarse LU solve -- about 2.3 of the phase's 6.6 ms per iteration, in kernels that would become
one launch per cycle instead of about 80.

## FP-12, the coarse-hierarchy fusion: written, bit-identical, and slower (2026-09-13)

The lever the last round pointed at was the one that worked for the DILU level walk: from the first
level small enough for a single block, run the whole rest of the V-cycle -- down, the coarsest LU
solve, and back up -- in ONE kernel, with `__syncthreads` where the launch boundaries were. It was
written, with every statement copied from the kernel it replaces (the FP32 SpMV, the weighted-Jacobi
smoother, the residual, the restriction gather, the injection prolongation, the zeroing and the
one-block LU substitution), so a grid-stride loop in one block computes each cell exactly as one thread
of a many-block launch did.

IT IS EXACTLY BIT-IDENTICAL: every residual line over 100 iterations of gasMixing/injectorPipe matches
the separate-launch path, which is the arithmetic claim the design makes. It is also slower at every
threshold:

| fuse levels below (cells) | pEqn ms/it | p solve ms/it |
|---------------------------|-----------:|--------------:|
| 64                        |        6.7 |           4.3 |
| 128                       |        6.7 |           4.3 |
| 256                       |        6.9 |           4.2 |
| 512                       |        6.9 |           4.3 |
| 1024                      |        7.1 |           4.7 |
| 2048                      |        7.0 |           4.8 |
| separate launches         |        6.5 |           4.2 |

WHY, and this is what the row gains. These kernels are ALREADY graph nodes, so node-to-node overhead is
not what the V-cycle pays. Fusing only below 64 cells removes a dozen node boundaries on levels where
one block of 256 threads is ample parallelism, and it still gained nothing, so the boundaries are
cheap. What the coarse levels actually pay is the device-side duration of a scattered indirect gather:
the FP32 SpMV takes 4.2 us on a level under 1,000 cells where the elementwise zeroT on the same grid
takes 0.76, because the owner/losort indirection with a few hundred threads has no parallelism to hide
its latency. Putting that on ONE multiprocessor makes it strictly worse.

So the premise behind this lever -- that the V-cycle is launch-bound -- is wrong for this code, and the
earlier reading of "121 kernels per cycle, 95% overhead" needs correcting: the overhead is real but it
is INSIDE each kernel, not between them. The reverted kernel's note sits in device_amg_vcycle.cu so the
next person does not rediscover it.

WHAT IS LEFT ON THIS ROW, and it is now the only candidate the measurements support: the coarse
operator's LAYOUT. The SpMV's inner loop is two indirections (`nei[f]` then `psi[...]`, and `losort[k]`
then `owner[f]` then `psi[...]`) over a face list that is not ordered for coalescing. A per-level CSR
built once per hierarchy, with the values regathered on each Galerkin update and the columns
concatenated owner-then-losort so the sum order is unchanged, would make the inner loop one contiguous
read and stay bit-identical. The ceiling: the SpMV is 2.5 of the phase's 6.6 ms per iteration, and the
elementwise kernels on the same grids run 5x faster.

## FP-12: the coarse operator as contiguous rows (2026-09-13)

The one candidate the last round's measurements left standing. The FP32 SpMV read its off-diagonals
through two indirections -- `upper[f]` with `psi[nei[f]]` over the cell's owner faces, then `lower[f]`
with `psi[owner[f]]` over losort -- into arrays ordered by FACE, not by cell. Each grid now carries its
rows contiguously instead: entry i of row c is a (value, column) pair, the cell's owner faces in
ownerStart order first and then its neighbour faces in losort order, which is the exact sequence the
face-form kernel sums in. Same terms, same order, same bits.

The values are refilled once per solve from the FP64 face arrays, and that REPLACES the two casts that
grid's upper and lower needed, so the layout costs no extra launch. The structure is built once per
hierarchy, on the host, because the agglomeration is static for the life of the mesh.
`BRAE_AMG_CSR=0` restores the face form; `BRAE_AMG_CSR_BELOW` restricts it to grids under a size.

| injectorPipe, 74,650 cells | face form | contiguous rows |
|----------------------------|----------:|----------------:|
| SpMV, launches per iteration |     451.5 |           451.5 |
| SpMV, ms per iteration       |     2.602 |           1.634 |
| casts, launches per iteration |     93.2 |            71.2 |
| whole iteration, launches     |     2,425 |           2,414 |
| whole iteration, GPU ms       |     13.79 |           12.77 |

The matrix-vector kernel is 37% faster for the same 451 launches, and the iteration loses a full
millisecond of GPU time. Three runs of 100 iterations on four cases, pressure solve in ms per
iteration:

| case          | face form | contiguous rows |
|---------------|----------:|----------------:|
| injectorPipe  | 4.3 / 4.3 / 4.3 | 3.9 / 4.0 / 3.8 |
| aerofoil      | 2.9 / 2.9 / 2.9 | 2.5 / 2.5 / 2.5 |
| squareBendLiq | 5.2 / 5.2 / 5.1 | 4.8 / 4.8 / 4.8 |
| squareBend    | 5.8 / 5.7 / 5.7 | 5.8 / 5.7 / 5.7 |

squareBend is flat because it is the transonic case: its pressure matrix is asymmetric and the solve is
BiCGStab with the V-cycle as a preconditioner, where the cycle is a smaller share of the solve. The
other three gain 8 to 14 percent of the pressure solve.

EVERY GRID, INCLUDING THE FINEST. Swept on injectorPipe, pressure solve in ms per iteration: 4.3 with
the face form, then 4.3, 4.2, 4.0, 3.9, 3.8 as the threshold rises through 512, 2,048, 8,192, 16,384
and every grid. The fine grid gains too: its owner half is already sequential in the face arrays, but
its neighbour half is read through losort, and the row layout makes both contiguous. That is why the
default is every grid rather than the coarse ones this row started with.

BIT-IDENTITY: every residual line over 100 iterations matches the face form on injectorPipe,
squareBend, aerofoilNACA0012 and squareBendLiq. 34 gates covering the AMG, the PCG and BiCGStab
drivers, the transonic path, determinism, run-to-run identity and the tutorials pass at their bounds.

## FP-12, the transonic tail: it is not a V-cycle problem, it is the host (2026-09-13)

squareBend is the one case the row layout did not help, so it was profiled on its own. Two findings.

FIRST, its V-cycle is a different one. The transonic pressure matrix is asymmetric, and that path runs
the FP64 cycle with the two-stage Gauss-Seidel smoother, not the FP32 weighted-Jacobi cycle the other
tutorials take. The contiguous-row layout is built for the FP32 mirrors, so it never applied here. What
it would be worth is measurable: the FP64 SpMV is 1.41 of the pressure phase's 4.58 GPU ms per
iteration (2.46 ms across the whole iteration), and the FP32 one gained 37% from the same change.

SECOND, and this reframes the row: the GPU is idle for most of the phase.

| phase (nsys, 20 iterations) | squareBend wall / busy | injectorPipe wall / busy |
|-----------------------------|-----------------------:|-------------------------:|
| UEqn                        |  3.74 / 2.20 ms  (59%) |    4.28 / 3.07 ms  (72%) |
| EEqn                        |  2.81 / 1.16 ms  (41%) |    2.96 / 1.52 ms  (52%) |
| pEqn                        | 11.98 / 4.58 ms  (38%) |   14.59 / 5.55 ms  (38%) |
| turbulence                  |  4.88 / 2.99 ms  (61%) |    3.82 / 2.50 ms  (65%) |

The profiler inflates the host side, so the honest unprofiled figures are the GPU-busy column against
the plain four-phase wall: about 60% on squareBend (10.9 of 18.1 ms) and about 75% on injectorPipe
(12.6 of 16.8). Either way the pressure phase is the worst of the four on both cases, at 38%.

WHERE THE IDLE TIME IS, measured by bracketing every gap over 60 us in the pressure phase with the
kernels either side of it, over 20 iterations:

| between                            | count | total ms | mean us |
|------------------------------------|------:|---------:|--------:|
| the AMG build (once)               |     1 |     73.4 |  73,401 |
| publishValues -> scalarCopy        |    17 |     13.2 |     774 |
| reciprocalV -> amul                |    20 |     12.2 |     609 |
| finalSum -> frUpdate               |    19 |      8.8 |     463 |
| bicgEndCond -> scalarCopy          |     9 |      3.5 |     388 |
| bicgMidCond -> zero                |     9 |      3.4 |     378 |

Excluding the one-off hierarchy build, that is 2.4 ms per iteration of idle GPU inside the pressure
phase alone. The mailbox spin is not the cause -- it is a tight busy-wait with no backoff. Attributing
the API calls inside those windows says what is: 656 `cudaLaunchKernel` calls across 17 gaps, about 38
launches per gap, plus a graph launch at 68 us apiece. The host cannot issue work as fast as the GPU
retires it, because the kernels are short and only the SOLVERS are captured into graphs. The
assemblies, the boundary updates and the reporting are not.

`finalSum -> frUpdate` at 463 us once per iteration is its own item: a blocking reduction feeding the
flow-rate inlet condition, which is FP-9's row, and it is worth half a millisecond per iteration here.

So the transonic tail of FP-12 is not a V-cycle question. The two things worth doing on this case are
FP-10's -- capture the assembly phases, whose addressing is mesh-constant, the way the solvers already
are -- and, second order, extending the contiguous-row layout to the FP64 SpMV so the asymmetric path
gets what the symmetric one just got.

## FP-10 / FP-8: what the host is really doing, and what removing two of its stalls was worth (2026-09-13)

FP-10 says the iteration is launch-bound and wants the assembly phases captured into graphs. Before
writing that, the host side was counted properly on squareBend (112,000 cells, 20 iterations under
nsys, NVTX phases):

| host operation                  | per iteration | API time per iteration |
|---------------------------------|--------------:|-----------------------:|
| blocking `cudaMemcpy`           |          49.6 |                 7.04 ms |
| `cudaDeviceSynchronize`         |           5.0 |                 1.39 ms |
| `cudaLaunchKernel`              |         1,004 |                 4.45 ms |

Two of those blocking copies were plainly wasteful and are now gone.

- THE SIMPLEC ROW SUM built a host vector of nCells doubles, filled it with ones and uploaded it on
  EVERY iteration -- a blocking 896 KB copy, 0.53 ms of API time per iteration on this mesh. There has
  been a device-resident `deviceOnes` all along, and the incompressible twin already used it.
- THE CONTINUITY REPORT made three blocking reductions per iteration, one of which was sum(V) -- a
  property of the mesh, recomputed every time. It now computes sum(V) once (keyed on the volume buffer,
  which only a mesh move replaces) and takes the other two through one mailbox read instead of two
  blocking copies.

Measured, same case, same profile: blocking host operations 54.6 to 50.6 per iteration and their API
time 8.43 to 6.80 ms, with the time outside the four phases 3.91 to 2.65. 21 gates covering the
continuity line, the SIMPLEC path, the tutorials and run-to-run identity pass.

AND THE WALL DID NOT MOVE: 18.1 / 18.1 / 18.9 ms per iteration before, 18.4 / 18.4 / 19.3 after, which
is noise. That is the finding, and it corrects how the earlier numbers should be read.

A blocking copy's API duration is NOT idle-GPU time. Most of it is the host waiting for work the GPU
was going to do anyway; removing the wait does not remove the work. The waste is only where the GPU
sits IDLE, and that was measured separately by bracketing the gaps: 2.4 ms per iteration inside the
pressure phase, in windows where the host is issuing about 38 kernels apiece while the GPU has nothing
queued. Removing 4 of some 55 drains moves that by about its share, which is inside the run-to-run
noise. The two fixes are kept because they are right -- less host work, fewer drains, a wasteful upload
gone -- not because they made this case faster.

A THIRD BLOCKING SITE WENT WITH THEM, and it also confirmed the reading. The compressible stress term
copied each velocity-gradient component into the 9*nC tensor with a BLOCKING device-to-device
`cudaMemcpy` -- nine per momentum assembly, in a phase measured at 59% GPU-busy -- where its two
siblings (`deviceGradU`, `deviceLeastSquaresGradU`) use the async form on the per-thread stream. It is
now async too. Same bytes, same order, same bits, and 40 gates covering the momentum assembly, the
stress term, the closures and the tutorials pass. The wall, again, did not move: squareBend runs 200
iterations in 4.42 / 4.44 / 4.52 s against 4.41 / 4.43 before.

That was also the one operation that made the momentum assembly impossible to capture into a graph: a
blocking copy is illegal during stream capture. So the assembly is now capture-safe, and it does no
host reads at all.

WHERE THE HOST TIME ACTUALLY IS, after three fixes that each removed real blocking work and none of
which moved the clock: the launch API itself. squareBend issues 1,004 `cudaLaunchKernel` calls per
iteration costing 4.45 ms of API time, against about 19 ms per iteration of wall and 11 of GPU-busy.
Removing 13 blocking operations per iteration did nothing because the time they occupied was mostly
waiting for work the GPU still has to do; the 1,004 launches are not waiting for anything, they ARE the
host's work. A captured phase replays at about 0.7 us per node against 4.4 us per launch, which is why
the capture is worth roughly 3.7 ms per iteration here and the drain-removal was worth nothing.

The prerequisite stands and is now the whole of the remaining work on this row: the assemblies'
temporaries are pool-allocated per call, so their addresses are not stable across iterations and a
captured graph would replay against whatever the pool later hands to someone else. They have to be
hoisted into a per-solver workspace -- 155 buffer declarations across the five assembly files -- before
any phase can be captured safely. The momentum assembly is the natural first one: it is capture-safe
as of this change, it has no host reads, and it is 144 launches per iteration.

## FP-10: capturing the momentum assembly, what it took and what it was worth (2026-09-13)

The prerequisite was real, and it took three steps to satisfy, each found by a failure rather than by
reading:

1. THE ASSEMBLY'S OWN TEMPORARIES. `assembleUEqn`'s 18 declaration sites were stack objects taking
   blocks from the device pool on every call. They are now named members of a per-mesh workspace, bound
   at each site by reference (or, for the `[3]` arrays, by pointer) under the same names, so not one
   line of the arithmetic changed. Bit-identical against a binary built without it on squareBend,
   injectorPipe and aerofoilNACA0012: every residual line and all nine written fields.
2. THE HELPER IT CALLS. The stress term allocates sixteen buffers of its own, and it also took
   ownership of three with `std::move` -- which swaps a buffer's device pointer on every call, the one
   thing a captured graph cannot survive. It has its own workspace now, and the moved-from buffers are
   aliased instead of moved.
3. THE MATRIX IT FILLS. `MomentumMatrix UEqn` was constructed fresh every iteration, so its twelve buffers
   were pool blocks too -- and EMPTY on entry, which is why the memory checker caught `axpyKernel`
   reading address 0x100 on the first replay. It is now persistent per mesh.

With all three, the capture works: 200 iterations of squareBend, captured against direct, every
residual line identical.

WHAT IT WAS WORTH, and the number corrects the estimate that motivated it:

| squareBend, per iteration | direct | captured |
|---------------------------|-------:|---------:|
| `cudaLaunchKernel` calls  |  1,003 |      965 |
| their API time            | 4.08 ms | 3.91 ms |
| graph launches            |    1.8 |      2.6 |
| kernels on the GPU        |  1,333 |    1,333 |
| wall, 200 iterations      | 4.46 / 4.49 s | 4.40 / 4.51 s |

The momentum assembly is 38 launches, not the 144 the phase contains -- the rest of that phase is its
SOLVE. So capturing it removes 38 launches and 0.17 ms of host API time per iteration, and the wall
does not move outside noise. The earlier estimate of 3.7 ms per iteration assumed all 1,004 launches
were capturable; they are not. Most of them are inside solver loops whose trip count depends on a
residual the host has to read, and a graph cannot hold those.

THE HONEST ARITHMETIC FOR THE REST OF THE ROW: the four assemblies together are of the order of 150 of
the 1,003 launches, so capturing all of them is worth about 0.7 ms per iteration of host API time on a
22 ms iteration -- 3%, for the same refactor repeated four more times over about 110 more buffer
declarations. That is not a good trade at this point in the campaign, and the row should not be
finished by doing it. What is left in those 1,003 launches is the SOLVERS: the colour Gauss-Seidel
momentum sweep, the turbulence solves and the DILU walks, which are host-driven loops rather than fixed
sequences.

WHAT IS KEPT, and why it is not dead code: the three workspaces stay, because they are bit-identical,
they remove pool churn from the hot path, and they are the prerequisite for any future capture. The
capture itself stays behind `BRAE_CAPTURE_ASSEMBLY`, and `tests/rho_capture_assembly_identity.sh`
exercises it on squareBend and the aerofoil -- 30 iterations each, residual lines and written fields
byte-identical, with the announcement checked in both directions so the arms cannot pass by comparing
two direct runs. Without that gate the capture path would be untested, and the two failures above are
exactly what an untested one looks like.

## FP-6 and FP-7: squareBendLiq's energy phase, and the third confirmation of the same lesson (2026-09-13)

squareBendLiq is the one tutorial still slower than 20 cores, and its energy phase costs 5.8 ms per
iteration against squareBend's 2.2 on the same mesh. Profiled with NVTX phases, that phase is 4.9 ms of
wall against 1.9 of GPU work, and the idle is ONE gap of 3.15 ms per iteration, sitting between two
boundary-value kernels.

THE THERMO FAILURE FLAG (FP-6) was the first suspect and is now fixed, on the evidence rather than on
the outcome. The he-to-T inversion allocated a fresh one-int buffer per call, uploaded it with a
blocking copy and read it back with another, for the cell pass and again for the boundary pass: four
4-byte round trips per iteration, 0.97 ms of API time on squareBend. It is now a persistent pair of
ints reset by a one-thread kernel, and the driver reads both once per iteration inside the mailbox read
the continuity report already makes. The wall did not move, which by now is the expected answer for
removing a drain that was waiting on queued work.

THE EXPRESSION PATCHES (FP-7) were the gap. Every accessor of the expression context called `.host()`
on a WHOLE field and sliced the patch out of it: 112,000 cells or 22,400 faces copied to evaluate an
expression over one wall, once per field named, once per patch, once per iteration. Two changes, and
the order in which they paid is the point:

| squareBendLiq            | D2H over 20 iterations | energy phase, 3 runs |
|--------------------------|-----------------------:|---------------------:|
| whole-field downloads    |               142.3 MB | 5.7 / 5.9 / 4.7 ms/it |
| sized by the patch       |                70.4 MB | 5.2 / 6.1 / 5.7 ms/it |
| ...and cached per evaluation |             70.4 MB | 4.7 / 4.4 / 4.9 ms/it |

Halving the BYTES changed nothing: it replaced five big blocking copies with 74 small ones, and the
cost is the round trip, not the payload. Fetching each field at most once per evaluation -- the
evaluator walks a tree and asks for a name every time it appears -- is what moved the phase, from 5.8
to about 4.7 ms per iteration, and the four phases from 21.0 to 19.9.

THE LAST STEP WAS THE BATCH, and it is the one that shows the rule cleanly. After the cache the patch
still made SIX blocking copies per iteration -- five fetches of 179,200 bytes and the upload of the
result -- at about 500 us each. 179 KB in 500 us would be 0.36 GB/s, so that is not bandwidth; it is
six queue drains. The first request now fills the WHOLE cache: every registered field is gathered (the
scattered internal values) or sliced (the contiguous boundary values) into ONE device buffer, and that
buffer comes back in a single copy. The expression pays for fields it does not name in bytes, which
are cheap, rather than in round trips, which are not.

| squareBendLiq, energy phase | blocking copies/it | their API time | phase wall / busy (nsys) | phase, 3 plain runs |
|-----------------------------|-------------------:|---------------:|-------------------------:|--------------------:|
| whole-field downloads       |                8.0 |       2.75 ms  |        4.92 / 1.93 (39%) | 5.7 / 5.9 / 4.7 ms/it |
| sized by the patch, cached  |                6.0 |       3.03 ms  |        5.17 / 1.98 (38%) | 4.7 / 4.4 / 4.9 |
| ...and batched into one     |                2.0 |       0.79 ms  |        3.02 / 1.96 (65%) | 4.1 / 3.7 / 3.9 |

The energy phase is 5.8 to 3.9 ms per iteration and the four phases 21.0 to 19.0, which puts
squareBendLiq at 1.21x of OpenFOAM on 20 cores per iteration where it was 1.08x. The phase is now 65%
GPU-busy where it was 39%.

Two round trips are left: the batch coming back, and the result going out. The second is a host vector
pushed into the patch's refValue, so making it async needs a pinned staging buffer that outlives the
call -- about 0.47 ms per iteration, and the last thing on this row.

Gates: 15 of 16 pass, including `rho_patch_expression_vs_openfoam` at its 1e-12 walls bound,
`rho_squarebendliq`, `liquid_thermo`, `liquid_inversion`, `energy_bc`, `limit_temperature`,
`rho_tutorials`, `rho_simple_end_to_end` and run-to-run identity. The one failure, `liquid_correct`, is
one of the nine that fail identically at HEAD.

## FP-9: the inlet's reduction, and the first drain removal that moved the clock (2026-09-13)

`flowRateInletVelocity` sets U on the patch to -flowRate/gSum(rho*magSf) times the face normal. The sum
is a reduction, and brae read it to the host to pass avgU as a kernel argument -- one blocking copy per
inlet patch per iteration, measured on squareBend as a 463 us gap between the reduction and the patch
kernel with the GPU idle across it.

It now stays on the device: the reduction writes a device scalar, a one-thread kernel forms avgU beside
it, and the patch kernel reads it. OpenFOAM's `continue` on a non-positive sum -- which leaves the
patch untouched rather than writing zeros -- is carried in a second slot next to avgU, because a host
branch would need the number here and that read is the whole cost. The host-scalar entry point stays
for the incompressible callers.

| squareBend, 200 iterations | before | after |
|----------------------------|-------:|------:|
| wall                       | 4.42 / 4.44 / 4.52 s | 4.11 / 4.06 / 4.18 s |
| blocking copies per iteration |  49.6 |  30.6 |
| kernels per iteration      |  1,334 | 1,336 |
| GPU ms per iteration       |  11.06 | 11.02 |

THIS ONE MOVED THE CLOCK -- about 1.7 ms per iteration, 8% -- where the five drain removals before it
did not, and the difference is worth stating. Those sat where the host had to wait for queued work
anyway. This one sits between a reduction and the kernel that consumes it, in the middle of the
boundary update, so the stall was pure: the GPU had nothing else to run. Note also that the API time of
the remaining copies went UP (7.04 to 8.17 ms per iteration) while their count fell by 19: with fewer
stall points the host waits longer at each, which is what better overlap looks like from the API's
side.

The five `cudaDeviceSynchronize` calls per iteration in the earlier tables were `BRAE_PHASE_TIME`'s own
phase boundaries: this profile, taken without it, shows zero.

Ten gates pass, including `rho_flowrate_inlet_vs_openfoam`, `flowrate_device_solver_vs_openfoam`,
squareBend, squareBendLiq, the tutorials, the coded Function1 and run-to-run identity.

Where the three cases now stand, 200 iterations each, wall including prep:

| case          | before this campaign's last three rows | now |
|---------------|--------------------------------------:|----:|
| squareBend    | 4.44 s | 4.12 s |
| squareBendLiq | 4.50 s | 4.25 s |
| injectorPipe  | 3.75 s | 3.38 s |

## The six tutorials again, after the fast-path campaign (2026-09-13)

Same rules as the table at the top of this file and the same driver: 100 fixed SIMPLE iterations, wall
time of the solver run only, residualControl removed and `functions` stripped on both sides,
everything else the tutorial's own. brae is the CUDA mirror arm timed as a whole process; OpenFOAM is
`mpirun -np 20 rhoSimpleFoam -parallel`, decomposed `hierarchical` with the tutorial's own layout
scaled to 20 and `scotch` for angledDuct. Every arm completed its iterations. The meshes are the ones
the September 12 table used.

| tutorial                     |   cells | model                | brae 100 it | OF-20c 100 it | brae is | brae ms/it | OF-20c ms/it | per iteration |
|------------------------------|--------:|----------------------|------------:|--------------:|--------:|-----------:|-------------:|--------------:|
| aerofoilNACA0012             |  16,000 | kOmegaSST            |       1.1 s |         1.7 s |   1.55x |          6 |            7 |         1.17x |
| angledDuctExplicitFixedCoeff |  28,000 | kEpsilon             |       1.1 s |         1.4 s |   1.27x |          6 |            7 |         1.17x |
| squareBend                   | 112,000 | kEpsilon             |       2.4 s |         3.7 s |   1.54x |         17 |           32 |         1.88x |
| squareBendLiq                | 112,000 | kEpsilon, liquid     |       2.8 s |         3.7 s |   1.32x |         15 |           25 |         1.67x |
| squareBendLiqNoNewtonian     | 112,000 | generalizedNewtonian |       2.0 s |         2.8 s |   1.40x |         12 |           21 |         1.75x |
| injectorPipe                 |  74,650 | kEpsilon             |       2.1 s |         2.8 s |   1.33x |         13 |           20 |         1.54x |

The ms/it columns are iterations 101-200 alone, from a second run at 200 iterations, so start-up is out
of them (brae: mesh load, device set-up, AMG hierarchy; OpenFOAM: MPI start-up and field reads).

BRAE IS NOW FASTER THAN OPENFOAM ON 20 GRACE CORES ON EVERY ONE OF THE SIX, whole-run and per
iteration. On 2026-09-12 it was faster on one.

| tutorial                     | whole-run then | now | per iteration then | now |
|------------------------------|---------------:|----:|-------------------:|----:|
| aerofoilNACA0012             |          0.89x | 1.55x |             0.62x | 1.17x |
| angledDuctExplicitFixedCoeff |          0.93x | 1.27x |             0.78x | 1.17x |
| squareBend                   |          1.33x | 1.54x |             1.52x | 1.88x |
| squareBendLiq                |          0.87x | 1.32x |             0.70x | 1.67x |
| injectorPipe                 |          0.94x | 1.33x |             0.95x | 1.54x |

Per iteration, brae's own numbers: the aerofoil 13 -> 6 ms, angledDuct 9 -> 6, squareBend 21 -> 17,
squareBendLiq 33 -> 15, injectorPipe 21 -> 13. What earned them, in order of what each was worth:
FP-2's DILU walk rule and the preconditioner policy on the aerofoil; FP-12's contiguous-row AMG
operator everywhere; FP-6, FP-7 and FP-9's round trips on the liquid cases and every case with a
flow-rate inlet; FP-1's colour-ordered scalar sweeps and FP-3's cached least-squares tensor.

Trajectory agreement at the last iteration, brae against OpenFOAM, relative L2 (this is an agreement
figure at a fixed iteration, not a convergence statement): at 100 iterations U 5.9e-04 / 3.3e-04 /
8.1e-04 / 1.0e-02 / 1.5e-03 / 1.3e-02 and p 3.1e-04 / 7.3e-05 / 5.0e-04 / 6.7e-05 / 2.2e-05 / 1.7e-06
across the six; at 200 every U figure except the two liquid cases falls below 1e-03 and every p figure
below 6e-05.

## FP-11: the turbulence block at 896,000 cells (2026-09-13)

The campaign has measured at 112,000 cells and below, where the tutorials live, and the rows that paid
there were mostly host round trips and launch counts. At 896,000 the balance is different, and this is
the first look with the current binary. squareBend scaled in all three directions, 100 fixed
iterations:

| phase       | ms/it | ns/cell |
|-------------|------:|--------:|
| UEqn        |  30.1 |      34 |
| EEqn        |  13.9 |      16 |
| pEqn        |  78.8 |      88 |
| turbulence  |  49.0 |      55 |
| four phases | 171.8 |     192 |

The turbulence block is exactly where the campaign left it -- 49.0 ms/it against the 48.9 recorded on
2026-09-08 -- so nothing done since has touched it. Its row asks for 46 ns/cell.

WHERE IT GOES, from nsys: the phase is 44.9 GPU ms per iteration and 22.5 of them are the FP64
matrix-vector product, 50 launches at 450 us each. squareBend relaxes k and epsilon at 0.9, so the
policy's rule picks a degree-22 Neumann series and each application is 21 of those products.

THE DEGREE, swept at 896k (the same run otherwise, residuals at iteration 100):

| degree | turbulence ms/it | four phases | U at 100 | k at 100 |
|--------|-----------------:|------------:|---------:|---------:|
| 22 (the rule's) |       48.9 |       172.0 | 8.92e-03 | 7.11e-03 |
| 16              |       41.8 |       161.5 | 9.09e-03 | 7.04e-03 |
| 12              |       39.2 |       160.0 | 9.72e-03 | 7.19e-03 |
| 8               |       39.1 |       156.6 | 9.59e-03 | 6.93e-03 |
| 4               |       32.2 |       151.5 | 1.26e-02 | 8.56e-03 |
| 2               |       28.7 |    diverged | 0.00e+00 |      nan |

Degree 12 meets the row's target -- 39.2 ms/it is 44 ns/cell -- on a trajectory indistinguishable from
degree 22's at iteration 100. Degree 2 diverges, which is what `turb_precon_vs_openfoam`'s degree-2 arm
already asserts.

BUT THE DEGREE IS NOT A TUNING KNOB HERE, and that is why this row does not close on this measurement.
It is derived: fvMatrix::relax bounds the series' ratio by the relaxation factor, so degree
ceil(ln 0.1 / ln alpha) buys a factor-of-ten residual reduction on ANY case with that factor. Choosing
12 at alpha = 0.9 means choosing 0.28 instead of 0.1, on one case at one size, at one iteration.
Changing the constant changes every case's preconditioner, so it needs the health table the row asks
for -- degrees across cases and sizes, not one trajectory -- and that is a policy change of the same
kind as the DILU-entry rule, with the same obligation to announce it.

THE SpMV WAS THE OTHER WAY IN, AND IT DOES NOT WORK. FP-12's contiguous-row layout took 37% off the
FP32 AMG operator, and this product has the same shape: its neighbour loop reads losort[k], then
lower[f] and psi[owner[f]], three indirections into arrays ordered by face. It was built for the FP64
product too, bit-identical, wired into the BiCGStab solves and their series, and measured: the products
it took over went 450 us to 397 (12%, not 37), the turbulence phase 49.0 to 47.4 ms/it, the iteration's
GPU time 153.0 to 151.0, the wall not at all. Reverted, with the reason in the source: a row layout
stores each off-diagonal TWICE where the LDU form stores it once, which in FP32 traded 10.5 MB of
values for 21 and won on coalescing, and in FP64 trades 21 MB for 42 plus a refill pass. The cost here
is bandwidth, and a layout that doubles the bytes cannot fix bandwidth.

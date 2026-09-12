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


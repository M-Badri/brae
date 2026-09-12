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

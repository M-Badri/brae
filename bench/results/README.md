# bench/results

Two separate benchmark campaigns live here, a solver and two months apart. They were previously in one
flat directory, where the July charts sat beside the September ones with nothing to tell them apart.

| | `simpleFoam/` | `rhoSimpleFoam/` |
|---|---|---|
| when | July 2026 | September 2026 |
| case | motorBike, 354k cells, kOmegaSST, 500 SIMPLE iterations | the six rhoSimpleFoam tutorials, 14k to 24M cells, 100 fixed iterations |
| hardware | one GB10 (20 Grace cores + Blackwell sm_121) | GB10 and GH200 (64 Grace cores + Hopper sm_90) |
| arms | brae, OpenFOAM CPU, AMGX, PETSc, spuma | the same five |
| the top-level README's images | **yes** - all of them | no |

## simpleFoam/ -- the motorBike five-way

The study the project README shows. `BENCHMARK_TOTALWALL.md` is the headline (brae beats a 20-core Grace
node above 15M cells, 1.04x widening to 1.13x at 35.6M); `motorbike_comparison.md` is the same-mesh,
same-scheme cross-check with drag and lift.

    solver_runtime_comparison.png   the README's runtime chart
    motorbike_p_*.png               surface pressure, one per arm, the README's five-way grid
    crossover_chart.png, .csv       where brae overtakes the CPU node
    *_100iter.txt                   the raw 100-iteration walls behind the table
    plot_runtime.py                 regenerates solver_runtime_comparison.png

## rhoSimpleFoam/ -- the OF-mirror campaign

    rhoSimpleFoam_tutorials_gb10.md    all six tutorials on the GB10; the port's running record
    rhoSimpleFoam_squareBend_gb10.md   squareBend in depth, the solver-by-solver work
    rhoSimpleFoam_gh200.md             the GH200: cross-solver ladder, memory ceiling, spuma on Hopper
    gh200_csv/                         177 raw rows behind that file, one csv per run
    brae_benchmark_arms.png, .svg      the two published charts. NOT the README's images -- these are
    brae_benchmark_scaling.png, .svg   the September ones, for the site and the posts.

## OF_GPU_SETUP_GB10.md

Kept at the top because both campaigns need it: how to get OpenFOAM's own pressure solve onto the GPU
through petsc4Foam, the AMGX blocker, and the path that works.

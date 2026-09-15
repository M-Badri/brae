# device_sym_gauss_seidel.cu assessment

## Classification

The steady-state smoother arithmetic is device-resident.  `psi`, diagonal, source, coefficients,
topology, level ordering, and level-ordered CSR operands are `DeviceBuffer` data or device pointers.
The single-block path performs an entire half-sweep in one CUDA kernel, with device block barriers
between dependency levels.  The fused path performs independent vector components in parallel blocks.

Host work remains in three places:

- On the first matrix use, `gsLevelsFor` copies fixed topology to the host, constructs the forward and
  backward DAG schedules and CSR order, copies those structures to the device, and caches them.
- The host selects single-block versus per-level execution and submits kernels.  If a dependency level
  is wider than 1,024 cells, the fallback is a host loop issuing one kernel per level.
- Environment-variable checks and the address-keyed topology cache are host-side control work.

None of those host operations performs the repeated Gauss-Seidel cell update.  On the normal cached,
single-block path, the solve data and numerical sweep remain on the GPU.

## Accepted experiment

The production refresh code issues separate forward and backward permutation kernels.  The factory
copy adds kernels that write both independent outputs in one launch.  Every output element evaluates
the same expression as production, so this changes neither operation order nor floating-point values.

On NVIDIA GB10 (compute capability 12.1), fusion helps below a measured size crossover.  Above it,
larger fused kernels lose enough occupancy/bandwidth efficiency to erase the launch saving, so the
factory copy retains the original two-kernel path when `nEntries > 450000` or `nCells > 110000`.

## Measurements

All figures are medians of seven alternating production/factory batches.  Each identity check ran 25
symmetric sweeps from identical data and compared every FP64 output bit.

| Topology | Production refresh | Factory refresh | Speedup | Result |
|---|---:|---:|---:|---|
| T3A, scalar (26,820 cells) | 0.020 ms | 0.014 ms | 1.425x | bitwise identical |
| T3A, fused 3-component | 0.037 ms | 0.027 ms | 1.380x | bitwise identical |
| duct3d, scalar (72,000 cells) | 0.045 ms | 0.041 ms | 1.083x | bitwise identical |
| duct3d, fused 3-component | 0.118 ms | 0.093 ms | 1.267x | bitwise identical |
| 209,825-cell grid, size fallback | 0.205 ms | 0.199 ms | 1.033x | bitwise identical |

The accepted change accelerates operand preparation, not the recurrence-dominated sweep.  T3A's
unchanged scalar sweep measured 0.839 ms in both implementations.  Consequently, the end-to-end gain
is modest when a solve takes many sweeps, but the change is useful and low risk when many one/few-sweep
solves refresh operands frequently.

## Rejected experiments

- Reducing the 1,024-thread block to the widest-level warp multiple was exact but neutral: 0.998x
  scalar and 1.007x fused on T3A.
- Explicit indirect-`psi` load staging was exact but slower: 0.949x scalar on T3A.
- A cooperative multi-block level walk was exact but about 0.21x--0.48x because grid barriers or queue
  atomics overwhelmed the added parallelism.
- CSR prefetch depths 2, 5, and 6 were all exact but slower than the production depth 4: 0.812x,
  0.965x, and 0.944x respectively on T3A.

Nsight Compute showed the production single-block kernel is latency/synchronization limited, not
bandwidth limited: one block occupies one SM, schedulers had no eligible warp for about 93% of sampled
cycles, and the dominant reported stall was the level barrier.  Exact Gauss-Seidel dependencies make
that bottleneck difficult to remove without replacing the smoother with a numerically different order.

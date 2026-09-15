# symGaussSeidel kernel experiment

This directory is deliberately isolated from the production source.  `device_sym_gauss_seidel.cu` is
a copy of the production implementation with one measured change: forward and backward operand
permutations share a kernel launch below a size crossover, while larger arrays retain the original path.

`benchmark_sym_gauss_seidel.cu` links the unchanged production implementation and compiles the factory
copy under renamed public symbols, so both implementations run in one process against the same matrix,
level schedule, operands, and initial field.  It requires bitwise-identical results before reporting
timings.

Build and run from the repository root:

```sh
cmake --build build --target factory_sym_gauss_seidel_bench -j2
./build/factory_sym_gauss_seidel_bench validation/T3A
```

The benchmark checks the per-solve operand refresh and complete scalar and three-component sweeps for
bitwise identity, then times refresh and steady-state level-ordered symmetric sweeps separately.

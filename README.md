[![Build & Test](https://github.com/Heavypilgrim/MscMolecDynRustProject/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/Heavypilgrim/MscMolecDynRustProject/actions/workflows/build_and_test.yml) [![Benchmark](https://github.com/Heavypilgrim/MscMolecDynRustProject/actions/workflows/benchmark.yml/badge.svg)](https://github.com/Heavypilgrim/MscMolecDynRustProject/actions/workflows/benchmark.yml)
# rsmd
OMG! Molecular Dynamics with Rust!

## Benchmarks
### LJ Direct Summation
- To benchmark the rust LJ direct summation with criterion, just run `cargo bench`.
- To compare the current performance of the Rsmd implementation with the one of yamd, just let the `plot.py` plot the comparison of the previously measured Rust benchmark with the yamd one.
![image](https://github.com/Heavypilgrim/MscMolecDynRustProject/blob/main/docs/LJ_Direct_Summation_Benchmark_Rust_Vs_C++.png?raw=true)

- To run the benchmark on the C++ yamd, please use the repo https://github.com/Heavypilgrim/MscMolecDynCPPBenchmark.

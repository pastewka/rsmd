[![Build & Test](https://github.com/Heavypilgrim/MscMolecDynRustProject/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/Heavypilgrim/MscMolecDynRustProject/actions/workflows/build_and_test.yml) [![Benchmark](https://github.com/Heavypilgrim/MscMolecDynRustProject/actions/workflows/benchmark.yml/badge.svg)](https://github.com/Heavypilgrim/MscMolecDynRustProject/actions/workflows/benchmark.yml)
# rsmd
OMG! Molecular Dynamics with Rust!

## Benchmarks
### LJ Direct Summation
- To benchmark the rust LJ direct summation with criterion, do the following steps:
* set the environment variable to make use of THP and compile in release mode `MALLOC_CONF="thp:always,metadata_thp:always" cargo build --release`
* run the benchmarks with `cargo bench`
- To compare the current performance of the Rsmd implementation with the one of yamd, just let the `plot.py -p` plot the comparison of the previously measured Rust benchmark with the yamd one.
![image](https://github.com/Heavypilgrim/MscMolecDynRustProject/blob/main/docs/LJ_Direct_Summation_Benchmark_Rust_Vs_C++.png?raw=true)

- To run the benchmark on the C++ yamd, please use the repo https://github.com/Heavypilgrim/MscMolecDynCPPBenchmark.

### Verlet loop over ndarray positions and velocities
- To implement the iteration over all positions and velocities simultaneously and with the best performance, the following chart depicts the performance of different loop variations:
![image](https://github.com/Heavypilgrim/MscMolecDynRustProject/blob/main/docs/ndarray_verlet_iteration_benchmark.svg?raw=true)
- The best performing one uses Zip to combine the two iterators for positions and velocities. (named loop_zip in chart)
```Zip::from(&mut self.positions)
            .and(&self.velocities)
            .for_each(|position, &velocity| {
                *position += velocity * 0.0001;
            });
```

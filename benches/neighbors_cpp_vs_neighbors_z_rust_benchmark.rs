use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::process::Command;
use std::time::Duration;

use mimalloc::MiMalloc;
#[global_allocator]
//static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
static GLOBAL: MiMalloc = MiMalloc;

fn benchmark_binary(binary_path: &str) -> Duration {
    // Run the binary and time its execution
    let start = std::time::Instant::now();
    let status = Command::new(binary_path)
        .status()
        .expect("Failed to execute binary");

    if !status.success() {
        panic!("Binary did not run successfully");
    }

    start.elapsed()
}

//Benchmark using the executable "neighbors_z_rust_executable__..." and "neighbors_cpp_executable__..."
fn neighbors_cpp_vs_neighbors_z_rust_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbors_cpp_vs_neighbors_z_rust_benchmark");

    let binary_folder = "target/release/";

    let executables = ["neighbors_cpp_executable__inputfile_lj_cube_1000_timestep_0_0001_nbiterations_50k_cutoff_1_5", "neighbors_z_rust_executable__inputfile_lj_cube_1000_timestep_0_0001_nbiterations_50k_cutoff_1_5"];

    group.sample_size(20);

    for executable in executables {
        let executable_path = binary_folder.to_owned() + executable;

        println!("Benchmark executable: {}", executable_path);

        group.bench_function(BenchmarkId::new(&executable_path, "__Benchmark"), |b| {
            b.iter(|| {
                benchmark_binary(&executable_path);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, neighbors_cpp_vs_neighbors_z_rust_benchmark);
criterion_main!(benches);

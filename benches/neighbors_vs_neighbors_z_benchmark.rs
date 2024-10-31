use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rsmd::md_implementation::{self, neighbors::NeighborList, neighbors_z::NeighborListZ};
use std::fs;

use mimalloc::MiMalloc;
#[global_allocator]
//static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
static GLOBAL: MiMalloc = MiMalloc;

fn neighbors_vs_neighbors_z_benchmark(c: &mut Criterion) {
    const TIMESTEP: f64 = 0.0001;
    const NB_ITERATIONS: u32 = 10_000;
    const CUTOFF_RADIUS: f64 = 1.5;
    const INPUT_PATH: &str = "input_files/lj_cube_1000.xyz";

    if !fs::metadata(INPUT_PATH).is_ok() {
        panic!("input file \"{}\" doesn't exist!", INPUT_PATH);
    }
    println!("input_file path {}", INPUT_PATH.to_string());

    let mut group = c.benchmark_group("neighbors_vs_neighbors_z");

    group.sample_size(20);
    let iterations_string = &("neighbors_".to_owned() + &NB_ITERATIONS.to_string() + "_iters");

    group.bench_function(BenchmarkId::new(iterations_string, INPUT_PATH), |b| {
        let mut atoms = md_implementation::xyz::read_xyz(INPUT_PATH.to_string())
            .expect("Failed to load atoms configuration.");
        let mut neighbor_list: NeighborList = NeighborList::new();
        b.iter(|| {
            for _ in 0..NB_ITERATIONS {
                atoms.verlet_step1(TIMESTEP.into());
                black_box(neighbor_list.update(&mut atoms, CUTOFF_RADIUS));
                atoms.verlet_step2(TIMESTEP.into());
            }
        });
    });

    let iterations_string = &("neighbors_z_".to_owned() + &NB_ITERATIONS.to_string() + "_iters");

    group.bench_function(BenchmarkId::new(iterations_string, INPUT_PATH), |b| {
        let mut atoms = md_implementation::xyz::read_xyz(INPUT_PATH.to_string())
            .expect("Failed to load atoms configuration.");
        let mut neighbor_list: NeighborListZ = NeighborListZ::new();
        b.iter(|| {
            for i in 0..NB_ITERATIONS {
                atoms.verlet_step1(TIMESTEP.into());
                if i % 100 != 0 {
                    black_box(neighbor_list.update(&mut atoms, CUTOFF_RADIUS, false));
                } else {
                    black_box(neighbor_list.update(&mut atoms, CUTOFF_RADIUS, true));
                }
                atoms.verlet_step2(TIMESTEP.into());
            }
        })
    });

    group.finish();
}

criterion_group!(benches, neighbors_vs_neighbors_z_benchmark);
criterion_main!(benches);

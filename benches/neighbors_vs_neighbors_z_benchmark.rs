use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rsmd::md_implementation::{self, neighbors::NeighborList, neighbors_z::NeighborListZ};
use std::fs;

use mimalloc::MiMalloc;
#[global_allocator]
//static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
static GLOBAL: MiMalloc = MiMalloc;

//Benchmark using the milestone
fn neighbors_vs_neighbors_z_benchmark(c: &mut Criterion) {
    const TIMESTEP: f64 = 0.0001;
    const NB_ITERATIONS: u32 = 10_000;
    const CUTOFF_RADIUS:f64 = 1.5;

    let folder = "input_files";
    let json_file_path = folder.to_owned() + "/benchmark_lj_direct_summation.json";
    let content = fs::read_to_string(&json_file_path).expect("JSON file \"benchmark_lj_direct_summation.json\" could not be loaded to benchmark with the specified input files.");
    let json: serde_json::Value =
        serde_json::from_str(&content).expect("JSON was not well-formatted");

    let input_files = json.as_array().expect(
        &("The JSON file \"".to_owned()
            + &json_file_path
            + "\" doesn't contain a valid JSON array with the filenames of the input files."),
    );

    let path = folder.to_owned() + &"/".to_owned() + input_files[5].as_str().unwrap();
    if !fs::metadata(&path).is_ok() {
        panic!("input file \"{}\" doesn't exist!", path);
    }
    println!("input_file path {}", &path.to_string());

    let mut group = c.benchmark_group("neighbors_vs_neighbors_z");

    group.sample_size(20);
    let iterations_string = &("neighbors_".to_owned()+&NB_ITERATIONS.to_string()+"_iters");

    group.bench_function(BenchmarkId::new(iterations_string, &path), |b| {
        let mut atoms = md_implementation::xyz::read_xyz(path.to_string())
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

    let iterations_string = &("neighbors_z_".to_owned()+&NB_ITERATIONS.to_string()+"_iters");

    group.bench_function(BenchmarkId::new(iterations_string, &path), |b| {
        let mut atoms = md_implementation::xyz::read_xyz(path.to_string())
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

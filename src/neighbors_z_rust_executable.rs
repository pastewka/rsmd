use rsmd::md_implementation::{neighbors_z::NeighborListZ, xyz::read_xyz};

use mimalloc::MiMalloc;
#[global_allocator]
//static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
static GLOBAL: MiMalloc = MiMalloc;

const TIMESTEP: f64 = 0.0001;
const NB_ITERATIONS: u32 = 50_000;
const CUTOFF_RADIUS: f64 = 1.5;
const INPUT_PATH: &str = "input_files/lj_cube_1000.xyz";

//Executable for benchmark "neighbors_cpp_vs_neighbors_z_rust_benchmark.rs"
fn main() {
    let mut atoms = read_xyz(INPUT_PATH.to_string()).expect("Failed to load atoms configuration.");

    let mut neighbor_list: NeighborListZ = NeighborListZ::new();

    for i in 0..NB_ITERATIONS {
        atoms.verlet_step1(TIMESTEP.into());
        if i % 100 != 0 {
            neighbor_list.update(&mut atoms, CUTOFF_RADIUS, false);
        } else {
            neighbor_list.update(&mut atoms, CUTOFF_RADIUS, true);
        }
        atoms.verlet_step2(TIMESTEP.into());
    }
}

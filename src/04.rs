use rsmd::md_implementation::{self, xyz};

// use morton_encoding;

use mimalloc::MiMalloc;
#[global_allocator]
//static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
static GLOBAL: MiMalloc = MiMalloc;

const NB_ITERATIONS: u32 = 100;
const SCREEN_INTERVAL: u32 = 1;
const FILE_INTERVAL: u32 = 1000;
const TIMESTEP: f64 = 0.0001;
const INPUT_FOLDER: &str = "input_files/";
const INPUT_FILE: &str = "lj54.xyz";
const OUTPUT_FILE: &str = "traj.xyz";

fn main() {
    let mut atoms =
        md_implementation::xyz::read_xyz_with_velocities(INPUT_FOLDER.to_owned() + INPUT_FILE)
            .unwrap();
    println!(
        "atom configuration loaded with {} atoms",
        atoms.positions.ncols()
    );
    //delete old trajectory if it exists
    _ = std::fs::remove_file(OUTPUT_FILE);

    let ekin: f64 = atoms.kinetic_energy();
    let epot: f64 = atoms.lj_direct_summation(None, None);

    for i in 0..NB_ITERATIONS {
        atoms.verlet_step1(TIMESTEP.into());

        let epot: f64 = atoms.lj_direct_summation(None, None);

        atoms.verlet_step2(TIMESTEP.into());

        if i % SCREEN_INTERVAL == 0 {
            let ekin: f64 = atoms.kinetic_energy();
            println!(
                "{:>20}{:>20}{:>20}{:>20}{:>20}{:>20}",
                i,
                i as f64 * TIMESTEP,
                ekin,
                epot,
                ekin + epot,
                ekin / (1.5 * atoms.positions.shape()[1] as f64)
            );
        }

        if i % FILE_INTERVAL == 0 {
            xyz::write_xyz(OUTPUT_FILE.to_string(), atoms.clone()).unwrap();
        }
    }
}

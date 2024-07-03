use rsmd::md_implementation::{self, atoms::Atoms, xyz::read_xyz};

use mimalloc::MiMalloc;
#[global_allocator]
//static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
static GLOBAL: MiMalloc = MiMalloc;
fn main() {
    const EPSILON: f64 = 0.7;
    const SIGMA: f64 = 0.3;
    const ATOMS_DATA: &str = include_str!("../input_files/lj_cube_4096.xyz");
    let mut a: Atoms = read_xyz(ATOMS_DATA.to_string()).unwrap();
    for _ in 0..1000000 {
        a.lj_direct_summation(Some(EPSILON), Some(SIGMA));
    }
}

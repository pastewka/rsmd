mod atoms;
mod xyz;
use verlet;

fn main() {
    let mut atoms = xyz::read_xyz("cluster_3871.xyz".to_string()).unwrap();
}

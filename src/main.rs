mod atoms;
mod lj_direct_summation;
mod thermo;
mod types;
mod verlet;
mod xyz;

//use ndarray_rand::{RandomExt, rand_dist};
use ndarray::Array;

const nb_iterations: u32 = 1000000;
const screen_interval: u32 = 1000;
const file_interval: u32 = 1000;
const timestep: f64 = 0.0001;

fn main() {
    #![cfg_attr(feature = "dev", feature(plugin))]
    #![cfg_attr(feature = "dev", plugin(clippy))]

    //#![deny(missing_docs,
    //        missing_debug_implementations, missing_copy_implementations,
    //        trivial_casts, trivial_numeric_casts,
    //        unsafe_code,
    //        unstable_features,
    //        unused_import_braces, unused_qualifications)]
    let mut atoms = xyz::read_xyz("cluster_3871.xyz".to_string()).unwrap();
    //let mut atoms = xyz::read_xyz_with_velocities("lj54InclVelocity.xyz".to_string()).unwrap();

    let ekin :f64 = atoms.kinetic_energy();
    println!("kinetic energy: {:?}", ekin);
    // atoms.verlet_step1(timestep);
    // atoms.verlet_step2(timestep);

    //   let a = Array::random((2, 5), Uniform::new(0., 10.));
    //println!("{:8.4}", a);

    // for i in 0..nb_iterations{
    //     atoms.verlet_step1(i);
    // }
    atoms.verlet_step1(timestep);
    atoms.verlet_step2(timestep);
    //atoms.lj_direct_summation(None,None);
    //println!("x coordinates of the:\n 0th atom: {:?},\n 1st: {:?},\n 250th: {:?},\n 2200th: {:?},\n 3800th: {:?},\n last: {:?}", atoms.positions[[0,0]], atoms.positions[[0,1]], atoms.positions[[0,249]], atoms.positions[[0,2199]], atoms.positions[[0,3799]], atoms.positions[[0,atoms.positions.shape()[1]-1]]);

    for i in 0..atoms.positions.shape()[1] {
    //    println!("atom i={}, x={:?}", &i, atoms.positions[[0, i]]);
    //    println!("atom velo i={}, x={:?}", i, atoms.velocities[[0, i]]);
    }
}

mod atoms;
mod types;
mod verlet;
mod xyz;


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

    println!("x coordinates of the:\n 0th atom: {:?},\n 1st: {:?},\n 250th: {:?},\n 2200th: {:?},\n 3800th: {:?},\n last: {:?}", atoms.positions[[0,0]], atoms.positions[[0,1]], atoms.positions[[0,249]], atoms.positions[[0,2199]], atoms.positions[[0,3799]], atoms.positions[[0,atoms.positions.shape()[1]-1]]);

    for i in 3860..atoms.positions.shape()[1] {
        println!("atom i={}, x={:?}", i, atoms.positions[[0, i]]);
    }
}

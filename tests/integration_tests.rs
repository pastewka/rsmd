use ndarray::{array, Array2, Array3, Axis};

use mimalloc::MiMalloc;
#[global_allocator]
//static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
static GLOBAL: MiMalloc = MiMalloc;

const NB_ITERATIONS: u32 = 100;
const SCREEN_INTERVAL: u32 = 1;
const FILE_INTERVAL: u32 = 1000;
const TIMESTEP: f64 = 0.0001;
const INPUT_FOLDER: &str = "input_files/";
const INPUT_FILE: &str = "lj5InclVelocity.xyz";
const OUTPUT_FILE: &str = "traj.xyz";

fn arrays_almost_equal(should: &Array2<f64>, is: &Array2<f64>, tolerance: f64) -> bool {
    if should.shape() != is.shape() {
        return false;
    }

    for i in 0..should.shape()[0] {
        for j in 0..should.shape()[1] {
            let should_val = should[[i, j]];
            let is_val = is[[i, j]];

            if (should_val - is_val).abs() > tolerance {
                println!(
                    "Arrays differ should: {:?};  is: {:?};  index: ({},{})",
                    should_val, is_val, i, j
                );
                return false;
            }
        }
    }
    return true;
}

#[test]

fn test_milestone04_compared_to_yamd() {
    use rsmd::md_implementation::{self, xyz};

    let mut atoms =
        md_implementation::xyz::read_xyz_with_velocities(INPUT_FOLDER.to_owned() + INPUT_FILE)
            .unwrap();
    println!(
        "atom configuration loaded with {} atoms",
        atoms.positions.ncols()
    );
    //delete old trajectory if it exists
    _ = std::fs::remove_file(OUTPUT_FILE);

    let target_velocities_step0_before_after_verl1_after_verl2: Array3<f64> = array![
        [
            [1.77012, -0.572017, 0.956302, 0.500229, 1.19634],
            [0.43038, 1.25828, 0.428697, -0.997027, 0.265998],
            [-0.045422, 0.668012, -0.426285, 0.266474, -1.07314],
        ],
        [
            [1.77014, -0.57202, 0.956294, 0.500226, 1.19634],
            [0.430374, 1.25828, 0.428706, -0.997033, 0.266001],
            [-0.045423, 0.668011, -0.426282, 0.266481, -1.07315],
        ],
        [
            [1.77015, -0.572023, 0.956286, 0.500222, 1.19634],
            [0.430367, 1.25828, 0.428716, -0.997039, 0.266004],
            [-0.0454241, 0.668011, -0.426279, 0.266489, -1.07316],
        ],
    ];

    let target_pos_step0_before_after_verl1_after_verl2: Array3<f64> = array![
        [
            [-0.689712, 3.0363, 0.975778, 1.64322, 0.574133],
            [1.51676, 1.06446, 0.935256, 2.12177, 0.65288],
            [2.02485, 2.9303, 1.35348, -0.0872166, 3.52808],
        ],
        [
            [-0.689535, 3.03624, 0.975874, 1.64327, 0.574253],
            [1.5168, 1.06459, 0.935299, 2.12167, 0.652907],
            [2.02485, 2.93037, 1.35344, -0.08719, 3.52797],
        ],
        [
            [-0.689535, 3.03624, 0.975874, 1.64327, 0.574253],
            [1.5168, 1.06459, 0.935299, 2.12167, 0.652907],
            [2.02485, 2.93037, 1.35344, -0.08719, 3.52797],
        ]
    ];
    let target_force_step0_after_1st_sum_after_verl2: Array3<f64> = array![
        [
            [0.310122, -0.0581501, -0.163971, -0.068866, -0.0191348],
            [-0.126934, -0.00531164, 0.188917, -0.118519, 0.0618481],
            [-0.0209965, -0.0143095, 0.0575335, 0.149386, -0.171614],
        ],
        [
            [0.31019, -0.0581694, -0.163982, -0.0689016, -0.0191366],
            [-0.126971, -0.0053172, 0.188989, -0.118574, 0.0618731],
            [-0.0210027, -0.0143121, 0.0575061, 0.149465, -0.171657],
        ]
    ];
    const PRECISION: f64 = 0.0001;

    // println!("v: {:?}", &atoms.velocities);
    // println!("p: {:?}", &atoms.positions);

    assert!(atoms.forces.iter().all(|&f| f == 0.0));
    // println!(
    // "{:>20}{:>20}{:>20}{:>20}{:>20}{:>20}",
    // "step", "time", "ekin", "epot", "ekin+epot", "temperature"
    // );
    // println!(
    // "{:>20}{:>20}{:>20}{:>20}{:>20}{:>20}",
    // "----", "----", "----", "----", "---------", "-----------"
    // );

    let _ekin: f64 = atoms.kinetic_energy();
    let _epot: f64 = atoms.lj_direct_summation(None, None);

    // println!(
    // "first target force: {:?}",
    // &target_force_step0_after_1st_sum_after_verl2.index_axis(Axis(0), 0)
    // );
    //
    // println!(
    // "{:>20}{:>20}{:>20}{:>20}{:>20}{:>20}",
    // "START",
    // "-",
    // ekin,
    // epot,
    // ekin + epot,
    // ekin / (1.5f64 * atoms.positions.shape()[1] as f64)
    // );

    for i in 0..NB_ITERATIONS {
        if i == 0 {
            assert!(arrays_almost_equal(
                &target_velocities_step0_before_after_verl1_after_verl2
                    .index_axis(Axis(0), 0)
                    .to_owned(),
                &atoms.velocities,
                PRECISION
            ));
            assert!(arrays_almost_equal(
                &target_pos_step0_before_after_verl1_after_verl2
                    .index_axis(Axis(0), 0)
                    .to_owned(),
                &atoms.positions,
                PRECISION
            ));
        }

        atoms.verlet_step1(TIMESTEP.into());

        println!("v: {:?}", &atoms.velocities);
        println!("p: {:?}", &atoms.positions);

        if i == 0 {
            assert!(arrays_almost_equal(
                &target_velocities_step0_before_after_verl1_after_verl2
                    .index_axis(Axis(0), 1)
                    .to_owned(),
                &atoms.velocities,
                PRECISION
            ));
            assert!(arrays_almost_equal(
                &target_pos_step0_before_after_verl1_after_verl2
                    .index_axis(Axis(0), 1)
                    .to_owned(),
                &atoms.positions,
                PRECISION
            ));
        }

        let epot: f64 = atoms.lj_direct_summation(None, None);
        if i == 0 {
            assert!(arrays_almost_equal(
                &target_velocities_step0_before_after_verl1_after_verl2
                    .index_axis(Axis(0), 1)
                    .to_owned(),
                &atoms.velocities,
                PRECISION
            ));
            assert!(arrays_almost_equal(
                &target_pos_step0_before_after_verl1_after_verl2
                    .index_axis(Axis(0), 1)
                    .to_owned(),
                &atoms.positions,
                PRECISION
            ));
            assert!(arrays_almost_equal(
                &target_force_step0_after_1st_sum_after_verl2
                    .index_axis(Axis(0), 1)
                    .to_owned(),
                &atoms.forces,
                PRECISION
            ));
        }

        atoms.verlet_step2(TIMESTEP.into());
        println!("v: {:?}", &atoms.velocities);
        println!("p: {:?}", &atoms.positions);
        println!("f: {:?}", &atoms.forces);
        if i == 0 {
            assert!(arrays_almost_equal(
                &target_velocities_step0_before_after_verl1_after_verl2
                    .index_axis(Axis(0), 2)
                    .to_owned(),
                &atoms.velocities,
                PRECISION
            ));
            assert!(arrays_almost_equal(
                &target_pos_step0_before_after_verl1_after_verl2
                    .index_axis(Axis(0), 2)
                    .to_owned(),
                &atoms.positions,
                PRECISION
            ));
            assert!(arrays_almost_equal(
                &target_force_step0_after_1st_sum_after_verl2
                    .index_axis(Axis(0), 1)
                    .to_owned(),
                &atoms.forces,
                PRECISION
            ));
        }

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

    _ = std::fs::remove_file(OUTPUT_FILE);
}

#[test]
fn test_neighbor_list_z_compared_to_old_neighbor_list() {
    use itertools::assert_equal;
    use ndarray::Array1;
    use rsmd::md_implementation::{atoms, neighbors};

    let mut atoms = atoms::Atoms::new(3);
    let new_positions = vec![0.0, 0.0, 0.0, 7.0, 6.0, 0.0, 0.0, 0.0, 0.0];

    let new_positions_arr = Array2::from_shape_vec((3, 3), new_positions)
        .expect("Failed to create new positions array");
    atoms.positions.assign(&new_positions_arr);

    let mut neighbor_list: neighbors::NeighborList = neighbors::NeighborList::new();
    let (seed, neighbors) = neighbor_list.update(&mut atoms, 5.0);

    assert_eq!(neighbor_list.nb_total_neighbors(), 2);
    assert_eq!(neighbor_list.nb_neighbors_of_atom(0), 1);
    assert_eq!(neighbor_list.nb_neighbors_of_atom(1), 1);
    assert_eq!(neighbor_list.nb_neighbors_of_atom(2), 0);
    assert_equal(seed.clone(), Array1::<i32>::from_vec(vec![0, 1, 2, 2]));
    assert_equal(neighbors.clone(), Array1::<i32>::from_vec(vec![1, 0]));
}

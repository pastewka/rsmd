use crate::{atoms::Atoms, lj_direct_summation};
use ndarray::{array, Array1, ArrayView1};
use std::ops::{AddAssign, SubAssign};
//use ndarray_linalg::norm;

impl Atoms {

//    fn l2_norm(x: ArrayView1<f64>) -> f64 {
//    return x.dot(&x).sqrt();
//}

    pub fn lj_direct_summation(&mut self, epsilon_opt: Option<f64>, sigma_opt: Option<f64>) -> f64 {
        let epsilon: f64 = epsilon_opt.unwrap_or(1.0);
        let sigma: f64 = sigma_opt.unwrap_or(1.0);
        //if sigma == Null {sigma = Some(1.0);}
        let mut potential_energy = 0f64;

        for i in 0..self.positions.shape()[1] {
            for j in i + 1..self.positions.shape()[1] {
                println!("i={}, j={}", i, j);
                ////let distance_vector:Array1<f64> = &self.positions.column(i) - &self.positions.column(j);
                let mut distance_vector: Array1<f64> = self.positions.column(i).to_owned();
                //println!("distance_vector: {:?}",distance_vector);
                //println!("self.pos[j={}]: {:?}",j, self.positions.column(j));
                distance_vector.sub_assign(&self.positions.column(j));
                //println!("distance_vector after minus: {:?}",distance_vector);
                let distance: f64 = (&distance_vector * &distance_vector).sum().sqrt();
                //println!("distance: {}",distance);
                //println!("\n");

                //let a:ndarray::Array1<f64> = array![-3.72601200000000032375169212173,
                //    0.452300000000000146371803566581,
                //    -0.905450000000000088107299234252];
                //let dist: f64 = (&a * &a).sum().sqrt();
                //println!("a: {:?}",a);
                //println!("distance: {}",dist);
                let (pair_energy, pair_force) = lj_pair(distance, epsilon, sigma); //(distance, epsilon, sigma);
                potential_energy += pair_energy;
                //println!("epot: {}",&potential_energy);
                let force_vector: Array1<f64> = pair_force * &distance_vector / distance;
                //println!("force_vec:[{},{},{}]",force_vector[0],force_vector[1],force_vector[2]);
                //atoms.forces.slice(s![[.., i]]) += &force_vector;
                //atoms.forces.slice(s![[.., j]]) -= &force_vector;
                self.forces.column_mut(i).add_assign(&force_vector);
                self.forces.column_mut(j).sub_assign(&force_vector);
            }
        }
        println!("epot: {}", &potential_energy);
        return potential_energy;
    }
}
#[inline]
pub fn lj_pair(distance: f64, epsilon: f64, sigma: f64) -> (f64, f64) {
    let sd = &sigma / &distance;
    let sd2 = &sd * &sd;
    let sd6 = &sd2 * &sd2 * &sd2;
    let sd12 = &sd6 * &sd6;
    return (
        4.0 * &epsilon * (&sd12 - &sd6),
        24.0 * &epsilon * (2.0 * sd12 - sd6) / &distance,
    );
}

#[cfg(test)]
mod tests {
    use crate::{atoms::Atoms, xyz};
    use googletest::{matchers::near, verify_that};
    use ndarray::Array;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    #[test]
    fn test_lj_direct_summation() {
        const NB_ATOMS: usize = 10;
        const EPSILON: f64 = 0.7;
        const SIGMA: f64 = 0.3;
        const DELTA: f64 = 0.0001;
        let mut atoms = Atoms::new(usize::try_from(NB_ATOMS).unwrap()); //xyz::read_xyz_with_velocities("lj5InclVelocity.xyz".to_string()).unwrap();

        atoms.positions = Array::random((3, NB_ATOMS), Uniform::new(-1.0, 1.0)); //TODO:uncomment

        //compute and store original energy of the indisturbed configuration
        //atoms.lj_direct_summation(Some(EPSILON), Some(SIGMA));//TODO: make sigma and epsilon like

        let positions_original = atoms.positions.clone();
        let velocities_original = atoms.velocities.clone();
        atoms.lj_direct_summation(Some(EPSILON), Some(SIGMA));
        assert_eq!(atoms.positions, positions_original);
        let forces_original = atoms.forces.clone();
        println!("atoms.forces.shape(): {:?}", atoms.forces.shape());
        println!("atoms.forces (ORIGINAL ones): {:?}\n", atoms.forces);

        for j in 0..NB_ATOMS {
            for dim in 0..atoms.positions.shape()[0] {
                println!("atom manipulation on atom {}", j);
                //move atom to the right of the original position
                atoms.positions[[dim, j]] += DELTA;
                println!("atoms.positions += DELTA: {:.64}", atoms.positions);
                println!(
                    "atoms.forces (BEFORE 1st lj_direct_sum): {:?}\n",
                    atoms.forces
                );
                assert_eq!(
                    atoms.positions[[dim, j]] - DELTA,
                    positions_original[[dim, j]]
                );
                assert_eq!(
                    atoms.positions[[dim, j]],
                    positions_original[[dim, j]] + DELTA
                );

                let eplus = atoms.lj_direct_summation(Some(EPSILON), Some(SIGMA));
                println!(
                    "atoms.forces (AFTER 1st lj_direct_sum): {:?}\n",
                    atoms.forces
                );

                //move atom to the left of the original position
                atoms.positions[[dim, j]] -= 2.0 * DELTA;
                println!(
                    "atoms.positions[[{}, {}]] -= 2*DELTA: {}",
                    dim,
                    j,
                    atoms.positions[[dim, j]]
                );
                let eminus = atoms.lj_direct_summation(Some(EPSILON), Some(SIGMA));
                println!(
                    "atoms.forces (AFTER 2nd lj_direct_sum): {:?}\n",
                    atoms.forces
                );
                //move atom back to original position
                atoms.positions[[dim, j]] += DELTA;
                //assert_eq!(atoms.positions,positions_original);

                //finite-difference forces
                let fd_force = -(eplus - eminus) / (2.0 * DELTA);
                println!(
                    "atom {} ; eplus {} eminus {} fd_force {}",
                    j, eplus, eminus, fd_force
                );
                if &forces_original[[dim, j]].abs() > &1e-10 {
                    println!("NEAR abs(fd_force - forces_original[[dim,j]])={} / forces_original[[dim,j]]={}",f64::abs(fd_force - forces_original[[dim, j]]),forces_original[[dim,j]]);
                    verify_that!(
                        f64::abs(fd_force - forces_original[[dim, j]]) / &forces_original[[dim, j]],
                        near(0.0, 1e-5)
                    )
                    .unwrap_or_else(|e| panic!("{}", e));
                } else {
                    verify_that!(fd_force, near(forces_original[[dim, j]], 1e-10))
                        .unwrap_or_else(|e| panic!("{}", e));
                }
            }
        }
    }
}

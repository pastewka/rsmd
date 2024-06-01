use crate::md_implementation::atoms::Atoms;
use ndarray::{Array1,Array2};

impl Atoms {
    pub fn lj_direct_summation(&mut self, epsilon_opt: Option<f64>, sigma_opt: Option<f64>) -> f64 {
        let epsilon: f64 = epsilon_opt.unwrap_or(1.0);
        let sigma: f64 = sigma_opt.unwrap_or(1.0);
        let mut potential_energy = 0f64;

        for i in 0..self.positions.shape()[1] {
            for j in i + 1..self.positions.shape()[1] {
                let distance_vector = distance_vector(&self.positions, i, j);
                let distance: f64 = distance_vector.dot(&distance_vector).sqrt();
                
                let (pair_energy, pair_force) = lj_pair(distance, epsilon, sigma);
                potential_energy += pair_energy;

                //add force vector to ith and subtract it from jth force column
                self.forces.column_mut(i).scaled_add(pair_force / distance, &distance_vector);
                self.forces.column_mut(j).scaled_add(-pair_force / distance, &distance_vector);
            }
        }
        return potential_energy;
    }
}

#[inline]
fn distance_vector(positions: &Array2<f64>, i: usize, j: usize) -> Array1<f64> {
    return &positions.column(i) - &positions.column(j);
}

#[inline]
fn lj_pair(distance: f64, epsilon: f64, sigma: f64) -> (f64, f64) {
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
    use crate::md_implementation::atoms::Atoms;
    use googletest::{matchers::near, verify_that};
    use ndarray::Array;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use rand_isaac::isaac64::Isaac64Rng;

    #[test]
    fn test_lj_direct_summation() {
        const NB_ATOMS: usize = 10;
        const EPSILON: f64 = 0.7;
        const SIGMA: f64 = 0.3;
        const DELTA: f64 = 0.0001;
        const SEED: u64 = 42;
        let mut rng = Isaac64Rng::seed_from_u64(SEED);
        let mut atoms = Atoms::new(usize::try_from(NB_ATOMS).unwrap());

        atoms.positions = Array::random_using((3, NB_ATOMS), Uniform::new(-1.0, 1.0), &mut rng);
        println!("position[[0,0]]: {}", atoms.positions[[0, 0]]);

        //compute and store original energy of the indisturbed configuration
        let positions_original = atoms.positions.clone();

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
                    .unwrap_or_else(|e| panic!("Comparison of (|fd_force - force_original| / force_original) to 0.0 failed:\n{}", e));
                } else {
                    verify_that!(fd_force, near(forces_original[[dim, j]], 1e-10)).unwrap_or_else(
                        |e| panic!("Comparison of fd_force to force_original failed:\n{}", e),
                    );
                }
            }
        }
    }
}

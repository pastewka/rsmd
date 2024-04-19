use crate::atoms::Atoms;
use ndarray::Axis;

impl Atoms {
    pub fn verlet_step1(&mut self, timestep: f64) {
        self.verlet_velo_update(timestep);
        for i in 0..self.positions.shape()[0] {
            for j in 0..self.positions.shape()[1] {
                self.positions[[i, j]] += self.velocities[[i, j]] * timestep;
            }
        }
    }

    pub fn verlet_step2(&mut self, timestep: f64) {
        //println!("\n\n------------------- Verlet2 ---------------\n\n");
        //println!("forces: {:?}", &self.forces);
        //println!("Masses untransposed: {:?}", &self.masses);
        self.verlet_velo_update(timestep);
    }

    fn verlet_velo_update(&mut self, timestep: f64) {
        let velo_update = 0.5 * &timestep * &self.forces;

        let mass_trans = self.masses.t(); //TODO:maybe transpose not even necessary..
        println!("Masses transp.: {:?}", &mass_trans);
        let masses_broadcast = mass_trans.broadcast(velo_update.shape()).unwrap();
        println!("Masses transp. & broadcasted: {:?}", masses_broadcast);
        let velo_update_divided = velo_update / &masses_broadcast;
        println!("velo_upd / masses.T: {:?}", &velo_update_divided);
        for i in 0..3 {
            for j in 0..5 {
                self.velocities[[i, j]] += velo_update_divided[[i, j]];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{atoms::Atoms, xyz};
    use googletest::{matchers::near, verify_that};
    use ndarray::s;
    use ndarray::{array, Array};
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    #[test]
    fn test_verlet() {
        let nb_atoms = 5;
        let mut atoms = Atoms::new(usize::try_from(nb_atoms).unwrap());

        atoms.forces = Array::random((3, nb_atoms), Uniform::new(-1.0, 1.0));
        println!("before -- forces: {:?}", &atoms.forces);
        atoms.positions = Array::random((3, nb_atoms), Uniform::new(-1.0, 1.0));
        println!("before -- positions: {:?}", &atoms.positions);
        atoms.velocities = Array::random((3, nb_atoms), Uniform::new(-1.0, 1.0));
        println!("before -- velocities: {:?}", &atoms.velocities);
        atoms.masses = Array::ones(nb_atoms);

        let mut timestep: f64 = 1e-6;
        for _ in 0.. {
            if timestep >= 1e-3 {
                break;
            }
            let init_positions = atoms.positions.clone();
            let init_velocities = atoms.velocities.clone();

            for step in 0..100 {
                atoms.verlet_step1(timestep);
                atoms.verlet_step2(timestep);

                let f_t = &atoms.forces * (step as f64 + 1.0) * timestep;
                let analytical_velocities = &init_velocities + f_t;

                for j in 0..nb_atoms {
                    for dim in 0..atoms.positions.shape()[0] {
                        verify_that!(
                            atoms.velocities[[dim, j]], //near(atoms.velocities[[dim, j]], 1e-100)
                            near(analytical_velocities[[dim, j]], 1e-10)
                        )
                        .unwrap_or_else(|e| panic!("{}", e));
                    }
                }
            }
            timestep *= 10.0;
        }
    }
}

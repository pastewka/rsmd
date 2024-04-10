use crate::atoms::Atoms;
use ndarray::Zip;

impl Atoms {
    pub fn verlet_step1(&mut self, timestep: f64) {
        // let mut velo_cpy = self.velocities.clone();
        // let mut velo_cpy2 = self.velocities.clone();
        let velo_update = 0.5 * &timestep * &self.forces;

        println!("forces: {:?}", &self.forces);
        println!("Masses untransposed: {:?}", &self.masses);

        let mass_trans = self.masses.t(); //TODO:maybe transpose not even necessary..
        println!("Masses transp.: {:?}", &mass_trans);
        let masses_broadcast = mass_trans.broadcast(velo_update.shape()).unwrap();
        println!("Masses transp. & broadcasted: {:?}", masses_broadcast);
        let velo_update_divided = velo_update / &masses_broadcast;
        println!("velo_upd / masses.T: {:?}", &velo_update_divided);

        //println!("Velocities shape: {:?}",self.velocities.shape());
        //println!("velocity change shape: {:?}",velo_update_divided.shape());

        //self.velocities += velo_update_divided;

        /*
            //Other method with iterators
            type M = ndarray::Array2<f64>;

            // Create four 2d arrays of the same size
            let mut a = M::zeros((64, 32));
            let b = M::from_elem(a.dim(), 1.);
            let c = M::from_elem(a.dim(), 2.);
            let d = M::from_elem(a.dim(), 3.);
            println!("dim of d: {:?}", d.shape());
            Zip::from(&mut a)
                .and(&b)
                .and(&c)
                .and(&d)
                .for_each(|w, &x, &y, &z| {
                    *w += x + y * z;
                });

            let mut velo_update_val = 0.5 * timestep * forces;
            /*Zip::from(velo_cpy.rows())
                        .and(&velo_update_val.rows())
                        .and(&self.masses.broadcast(1))
                        .map_collect(| new_velo, &velo, &mass| {
                            *new_velo += velo / mass;
                        });
            */
            use itertools::izip;

            for (v_new, v_upd, m) in izip!(
                velo_cpy2.iter_mut(),
                velo_update_val.iter(),
                velo_cpy.iter()
            ) {
                *v_new += v_upd * m;
            }

            /*for mass_col in self.masses.axis_iter_mut(Axis(1)){

             for velo_row in velo_update_val.axis_iter_mut(Axis(0)){
                 velo_row /= mass_col;
             }
            }
            velo_cpy += velo_update_val;*/
            //assert!(velo_update_val == self.velocities);

            //        self.velocities += (0.5 * timestep * forces).axis_iter(Axis(0)).map(|row| {
            //
            //            row / self.masses.t()
            //        });
            //self.positions += self.velocities * timestep;
        */
    }

    pub fn verlet_step2(&mut self, timestep: f64) {
        self.velocities += 1.0; //(0.5 * timestep * forces).rowwise() / self.masses.t();
    }
}

#[cfg(test)]
mod tests {
    use crate::{atoms::Atoms, xyz};
    use googletest::{matchers::near, verify_that};
    use ndarray::s;
    use ndarray::{Array, Array2};
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    /*use ndarray::Array;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use crate::verlet::Atoms;*/

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
        atoms.masses *= 1.3;
        println!("before -- masses: {:?}", &atoms.masses);

        // let mut atoms = xyz::read_xyz("cluster_3871.xyz".to_string()).unwrap();
        // let nb_atoms = 5;
        // atoms.forces = atoms.forces.slice(s![..,..nb_atoms]).to_owned();
        // println!("before -- forces: {:?}",&atoms.forces);
        // atoms.positions = atoms.positions.slice(s![..,..nb_atoms]).to_owned();
        // println!("before -- positions: {:?}",&atoms.positions);
        // atoms.velocities = atoms.velocities.slice(s![..,..nb_atoms]).to_owned();
        // println!("before -- velocities: {:?}",&atoms.velocities);
        // atoms.masses = atoms.masses.slice(s![..nb_atoms]).to_owned();
        // println!("before -- masses: {:?}",&atoms.masses);
        let mut timestep: f64 = 1e-6;
        for i in 1i8..=3 {
            for j in 0..100 {
                let init_positions = &mut atoms.positions;
                let init_velocities = &mut atoms.velocities;
                atoms.verlet_step1(timestep);
                atoms.verlet_step2(timestep);
                //let analytical_velocities = *init_velocities + atoms.forces * (j as f64 + 1f64) * timestep;
                //let analytical_positions = *init_positions + init_velocities * (j as f64 + 1f64) * timestep + 0.5* atoms.forces * ((j as f64 + 1f64) * timestep);
                //

                //for j in 0..nb_atoms{
                //    for dim in 0..atoms.positions.shape()[0]{
                //        verify_that!(atoms.velocities[[dim,j]], near(analytical_velocities[[dim,j]], 1e-10));
                //    }
                //}
            }

            timestep *= 10.0;
        }

        /*TODO: maybe randomize atoms with seed
         let nb_atoms = 3;
         let mut atoms = Atoms::new(usize::try_from(nb_atoms).unwrap());
        atoms.positions = Array::<f64, _>::random((3, 2), Standard);//Array::random((3,nb_atoms), Standard);*/
    }
}

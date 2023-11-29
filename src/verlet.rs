use ndarray::Array2;

use crate::atoms::Atoms;

impl Atoms {
    pub fn verlet_step1(&self, &forces: ndarray::Array2<f64>, timestep: f64) {
        velocities += (0.5 * timestep * forces).rowwise() / masses.transpose();
        positions += velocities * timestep;
    }

    pub fn verlet_step2(&self, &forces: ndarray::Array2<f64>, timestep: f64) {
        velocities += (0.5 * timestep * forces).rowwise() / masses.transpose();
    }    
}

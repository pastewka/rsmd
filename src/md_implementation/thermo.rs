use crate::md_implementation::atoms::Atoms;
use ndarray::s;

impl Atoms {
    pub fn kinetic_energy(&self) -> f64 {
        let m_v_squared_sum = (&self.velocities.slice(s![0, ..])
            * &self.velocities.slice(s![0, ..])
            + &self.velocities.slice(s![1, ..]) * &self.velocities.slice(s![1, ..])
            + &self.velocities.slice(s![2, ..]) * &self.velocities.slice(s![2, ..]))
            .sum();
        return 0.5 * m_v_squared_sum;
    }
}
#[cfg(test)]

mod tests {
    use crate::md_implementation::{atoms::Atoms, xyz};
    use googletest::{matchers::near, verify_that};

    const FOLDER: &str = "input_files/";

    #[test]
    fn test_kinetic_energy() {
        let atoms: Atoms =
            xyz::read_xyz_with_velocities(FOLDER.to_owned() + "lj54InclVelocity.xyz").unwrap();
        assert_eq!(atoms.positions.nrows(), 3);
        assert_eq!(atoms.positions.ncols(), 54);

        let ekin: f64 = atoms.kinetic_energy();
        verify_that!(ekin, near(108.21239652853088842, 1e-10)).unwrap_or_else(|e| panic!("{}", e));
    }
}

use crate::atoms::Atoms;
use ndarray::{s, Array2};

impl Atoms {
    pub fn kinetic_energy(&self) -> f64 {
        let arr = Array2::from_shape_vec((3, 2), (1..=6).collect()).unwrap();
        println!("arr[1,:]: {:?}", arr.slice(s![1, ..]));
        println!("velo[0,:]: {:?}", &self.velocities.slice(s![0, ..]));

        let m_v_squared_sum = (&self.velocities.slice(s![0, ..])
            * &self.velocities.slice(s![0, ..])
            + &self.velocities.slice(s![1, ..]) * &self.velocities.slice(s![1, ..])
            + &self.velocities.slice(s![2, ..]) * &self.velocities.slice(s![2, ..]))
            .sum();
        println!("m_v_squared: {:?}", m_v_squared_sum);
        return 0.5 * m_v_squared_sum;
    }
}
#[cfg(test)]

mod tests {
    use crate::{atoms::Atoms, xyz};
    use googletest::{matchers::near, verify_that};
    #[test]
    fn test_kinetic_energy() {
        let mut atoms: Atoms =
            xyz::read_xyz_with_velocities("lj54InclVelocity.xyz".to_string()).unwrap();
        assert_eq!(atoms.positions.nrows(), 3);
        assert_eq!(atoms.positions.ncols(), 54);

        let ekin: f64 = atoms.kinetic_energy();
        verify_that!(ekin, near(108.21239652853088842, 1e-10)).unwrap_or_else(|e| panic!("{}", e));
    }
}

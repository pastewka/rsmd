use ndarray::{Array1, Array2};
#[allow(warnings)]

pub struct Atoms {
    pub masses: Array1<f64>,
    pub positions: Array2<f64>,
    pub velocities: Array2<f64>,
    pub forces: Array2<f64>,
}

impl Atoms {
    pub fn new(nb_atoms: usize) -> Self {
        let mut masses = Array1::ones(nb_atoms);
        let mut positions = Array2::zeros((3, nb_atoms));
        let mut velocities = Array2::zeros((3, nb_atoms));
        let mut forces = Array2::zeros((3, nb_atoms));
        Self {
            masses,
            positions,
            velocities,
            forces,
        }
    }

    pub fn push_vec(
        &mut self,
        m_vec: Vec<f64>,
        x_vec: Vec<f64>,
        y_vec: Vec<f64>,
        z_vec: Vec<f64>,
        vx_vec: Vec<f64>,
        vy_vec: Vec<f64>,
        vz_vec: Vec<f64>,
    ) {
        self.masses = Array1::from_shape_vec(m_vec.len(), m_vec).unwrap();
        for i in 0..x_vec.len() {
            self.positions[[0, i]] = x_vec[i];
            self.positions[[1, i]] = y_vec[i];
            self.positions[[2, i]] = z_vec[i];
        }
    }
    fn print_type_of<T>(_: &T) {
        println!("{}", std::any::type_name::<T>())
    }
}

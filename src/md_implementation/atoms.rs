use ndarray::{Array1, Array2};
#[allow(warnings)]
#[derive(Clone)]

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

    pub fn push_pos_vec(&mut self, x_vec: Vec<f64>, y_vec: Vec<f64>, z_vec: Vec<f64>) {
        for i in 0..x_vec.len() {
            self.positions[[0, i]] = x_vec[i];
            self.positions[[1, i]] = y_vec[i];
            self.positions[[2, i]] = z_vec[i];
        }
    }

    pub fn push_pos_velo_vec(
        &mut self,
        x_vec: Vec<f64>,
        y_vec: Vec<f64>,
        z_vec: Vec<f64>,
        vx_vec: Vec<f64>,
        vy_vec: Vec<f64>,
        vz_vec: Vec<f64>,
    ) {
        let nb_atoms = &x_vec.len();
        Self::print_type_of(nb_atoms);
        self.push_pos_vec(x_vec, y_vec, z_vec);
        for i in 0..*nb_atoms as usize {
            self.velocities[[0, i]] = vx_vec[i];
            self.velocities[[1, i]] = vy_vec[i];
            self.velocities[[2, i]] = vz_vec[i];
        }
    }

    fn print_type_of<T>(_: &T) {
        println!("type printing: {}", std::any::type_name::<T>())
    }
}

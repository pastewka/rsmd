use ndarray::{Array1, Array2};

#[derive(Clone)]

pub struct Atoms {
    pub masses: Array1<f64>,
    pub positions: Array2<f64>,
    pub velocities: Array2<f64>,
    pub forces: Array2<f64>,
}

impl Atoms {
    pub fn new(nb_atoms: usize) -> Self {
        let masses = Array1::ones(nb_atoms);
        let positions = Array2::zeros((3, nb_atoms));
        let velocities = Array2::zeros((3, nb_atoms));
        let forces = Array2::zeros((3, nb_atoms));
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
        self.push_pos_vec(x_vec, y_vec, z_vec);
        for i in 0..*nb_atoms as usize {
            self.velocities[[0, i]] = vx_vec[i];
            self.velocities[[1, i]] = vy_vec[i];
            self.velocities[[2, i]] = vz_vec[i];
        }
    }
}

use ndarray::{ArrayBase, DataMut, Dim, Ix1, OwnedRepr};

pub trait ArrayExt<D> {
    fn conservative_resize(&mut self, new_shape: D);
}

impl<S> ArrayExt<Ix1> for ArrayBase<S, Ix1>
where
    S: DataMut<Elem = i32>,
    ArrayBase<S, Dim<[usize; 1]>>: From<ArrayBase<OwnedRepr<i32>, Dim<[usize; 1]>>>,
{
    fn conservative_resize(&mut self, new_shape: Ix1) {
        let new_len = new_shape[0];
        let old_len = self.len();

        let mut new_array = Array1::zeros(new_len);

        let min_len = old_len.min(new_len);

        for i in 0..min_len {
            new_array[i] = self[i] as i32;
        }

        *self = new_array.to_owned().into();
    }
}

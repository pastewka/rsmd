use ndarray::{Array1, Array2};

//use ndarray::{ArrayBase, DataMut, Dimension, Ix1, Ix2};
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

    //pub fn resize(&mut self,new_num_elements:usize) {
    //
    //}

    fn print_type_of<T>(_: &T) {
        println!("type printing: {}", std::any::type_name::<T>())
    }
}

use ndarray::{ArrayBase, DataMut, Dim, Ix1, Ix2, OwnedRepr};

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

        println!("old_len: {}", old_len);
        println!("new_len: {}", new_len);
        let mut new_array = Array1::zeros(new_len);

        let min_len = old_len.min(new_len);

        for i in 0..min_len {
            new_array[i] = self[i] as i32;
        }

        // println!("Before resize:\n{:?}", self);
        *self = new_array.to_owned().into();
        // println!("After resize:\n{:?}", self);
    }
}

impl<S> ArrayExt<Ix2> for ArrayBase<S, Ix2>
where
    S: DataMut<Elem = i32>,
    ArrayBase<S, Dim<[usize; 2]>>: From<ArrayBase<OwnedRepr<i32>, Dim<[usize; 2]>>>,
{
    fn conservative_resize(&mut self, new_shape: Ix2) {
        let (new_rows, new_cols) = (new_shape[0], new_shape[1]);
        let (old_rows, old_cols) = self.dim();

        let mut new_array = Array2::zeros((new_rows, new_cols));

        let min_rows = old_rows.min(new_rows);
        let min_cols = old_cols.min(new_cols);

        for i in 0..min_rows {
            for j in 0..min_cols {
                new_array[(i, j)] = self[(i, j)] as i32;
            }
        }

        println!("Before resize:\n{:?}", self);
        *self = ArrayBase::from(new_array);
        println!("After resize:\n{:?}", self);
    }
}

/*

trait ArrayExt
{
    fn conservativeResize(&mut self,  new_shape: D);
}
// impl ArrayExt for Array2<f64> {
//     fn conservativeResize(&mut self, new_rows_cols: (usize, usize)){
//         if self.ncols()==new_rows_cols.0 && self.nrows() == new_rows_cols.1 {
//             return;
//         }
//         let mut vec = Array2::zeros(new_rows_cols);
//         vec.assign(self);

impl<S, D> ArrayExt for ArrayBase<S, D>
where
    S: DataMut<Elem = f64>,
    D: Dimension,
{
    fn conservativeResize(&mut self, new_shape: D) {



    }
}
*/
// impl<T: Zero> ArrayExt for ndarray::ArrayBase<S,D>
//     {
//     fn conservativeResize(&mut self, new_rows_cols: (usize, usize)){
//         if self.ncols()==new_rows_cols.0 && self.nrows() == new_rows_cols.1 {
//             return;
//         }
//         let mut vec = Array2::zeros(new_rows_cols);
//         vec.assign(self);
//     }

// }

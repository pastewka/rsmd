use crate::md_implementation::atoms::Atoms;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, Axis, Ix2};
use ndarray_stats::QuantileExt;

use super::atoms::ArrayExt;

pub struct NeighborList {
    seed: Array1<i32>,
    neighbors: Array1<i32>,
}

impl NeighborList {
    pub fn new() -> Self {
        let seed = Array1::from_elem(1, 1);
        let neighbors = Array1::from_elem(1, 1);
        return Self { seed, neighbors };
    }
    pub fn update(&mut self, atoms: Atoms, cutoff: f64) -> (Array1<i32>, Array1<i32>) {
        println!("positions init: {:?}", atoms.positions);
        if atoms.positions.is_empty() {
            //  self.seed.resize(0);
            //  self.neighbors.resize(0);
            //  return (self.seed,self.neighbors);
        }

        let mut origin = Array1::<f64>::from_elem(3, 3.0);
        let mut lengths = Array1::<f64>::from_elem(3, 3.0);
        let mut padding_lengths = Array1::<f64>::from_elem(3, 3.0);

        let mut nb_grid_points = Array1::<i64>::from_elem(3, 3);

        for (index_of_row, row) in atoms.positions.axis_iter(Axis(0)).enumerate() {
            let mut current_min = f64::INFINITY;
            for &val in row.iter() {
                if val < current_min {
                    current_min = val;
                }
            }
            origin[index_of_row] = current_min;
        }

        println!("origin {:?}", origin);

        for (index_of_row, row) in atoms.positions.axis_iter(Axis(0)).enumerate() {
            let mut current_max = -f64::INFINITY;
            for &val in row.iter() {
                if val > current_max {
                    current_max = val;
                }
            }
            lengths[index_of_row] = current_max - origin[index_of_row];
        }
        println!("lengths {:?}", lengths);
        for (index_of_i, i) in (&lengths / cutoff).iter().enumerate() {
            nb_grid_points[index_of_i] = i.ceil() as i64;

            //set to 1 if all atoms are in-plane
            println!("before: {}", nb_grid_points[index_of_i]);
            if nb_grid_points[index_of_i] <= 0 {
                nb_grid_points[index_of_i] = 1;
            }
            println!("after: {}", nb_grid_points[index_of_i]);
        }
        println!("nb_grid_points: {}", nb_grid_points);

        for (index_of_i, i) in nb_grid_points.iter().enumerate() {
            padding_lengths[index_of_i] = *i as f64 * cutoff - &lengths[index_of_i];
            origin[index_of_i] -= padding_lengths[index_of_i] / 2.0;
            lengths[index_of_i] += padding_lengths[index_of_i];
        }
        println!("padding_lengths: {:?}", padding_lengths);

        let mut r = atoms.positions.clone();
        for (mut col, &origin) in r.axis_iter_mut(Axis(0)).zip(origin.iter()) {
            col -= origin;
        }

        for (mut col, (grid_pts_per_len, &length)) in r
            .axis_iter_mut(Axis(1))
            .zip(nb_grid_points.iter().zip(lengths.iter()))
        {
            col *= *grid_pts_per_len as f64 / length;
        }

        let mut r = r.mapv(|i| i.floor() as i64);
        println!("r input to function atom_to_cell: {:?}", r);

        let atom_to_cell: Array1<i64> = r
            .axis_iter(Axis(1))
            .map(|col| Self::coordinate_to_index(col[0], col[1], col[2], nb_grid_points.view()))
            .collect();
        println!("atom_to_cell: {:?}", atom_to_cell);

        let mut sorted_atom_indices = Array1::from_vec((0..atom_to_cell.len()).collect()).into_raw_vec();
        println!("sorted_atom_indices with ingredients: {:?}",sorted_atom_indices);

        //sort indices according to cell membership
        sorted_atom_indices.sort_by(|&i, &j| atom_to_cell[i].cmp(&atom_to_cell[j]));
        println!("sorted_atom_indices actually sorted: {:?}",sorted_atom_indices);

        let cell_index:i64 = atom_to_cell[sorted_atom_indices[0]];
        let entry_index:i64 = 0;
        let binned_atoms: Vec<(i64,i64)> = vec![(cell_index,entry_index)];

        println!("binned_atoms at initialization: {:?}",binned_atoms);





        return (Array1::from_elem(5, 0), Array1::from_elem(5, 0));
    }

    fn coordinate_to_index_from_array(
        c: Array1<i64>,
        nb_grid_points: ArrayView1<i64>,
    ) -> Array1<i64> {
        println!("nb_grid_pts(0): {}", nb_grid_points[0]);
        println!("nb_grid_pts: {}", nb_grid_points);
        Self::coordinate_to_index(c[0], c[1], c[2], nb_grid_points);
        // c.row_mut(2).mapv_inplace(|z| z*nb_grid_points[1]);
        // c.row_mut(2).zip().mapv_inplace(|grid_pts_times_z| grid_pts_times_z +c.row(1));

        // c.row(0) += nb_grid_points[0];
        return Array1::from_elem(4, 2); // * (c.row(1) + nb_grid_points[1] * c.row(2));
    }

    fn coordinate_to_index(x: i64, y: i64, z: i64, nb_grid_points: ArrayView1<i64>) -> i64 {
        return x + nb_grid_points[0] * (y + nb_grid_points[1] * z);
    }
}

#[cfg(test)]
mod tests {
    use crate::md_implementation::atoms::Atoms;
    use googletest::{matchers::near, verify_that, assert_that};
    use itertools::assert_equal;
    use ndarray::Array2;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use rand_isaac::isaac64::Isaac64Rng;

    use super::NeighborList;

    #[test]
    fn test_neighbor_list() {
        let mut atoms = Atoms::new(4);
        let new_positions = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0];

        let new_positions_arr = Array2::from_shape_vec((3, 4), new_positions)
            .expect("Failed to create new positions array");
        atoms.positions.assign(&new_positions_arr);

        let mut neighbor_list: NeighborList = NeighborList::new();
        let (seed, neighbors) = neighbor_list.update(atoms, 1.5);
        assert_eq!(0,1);
    }
}

//update neighbor list from the particle positions stored in the 'atoms' argument

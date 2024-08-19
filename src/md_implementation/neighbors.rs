use crate::md_implementation::atoms::Atoms;
use itertools::iproduct;
use ndarray::{Array1, Array2, ArrayView1, Axis, Dim};
use ndarray_linalg::norm::Norm;
use std::cmp::Ordering;

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

//update neighbor list from the particle positions stored in the 'atoms' argument
    pub fn update(&mut self, atoms: Atoms, cutoff: f64) -> (Array1<i32>, Array1<i32>) {
        if atoms.positions.is_empty() {
            self.seed.conservative_resize(Dim(0));
            self.neighbors.conservative_resize(Dim(0));
            return (self.seed.clone(), self.neighbors.clone());
        }

        let mut origin = Array1::<f64>::from_elem(3, 3.0);
        let mut lengths = Array1::<f64>::from_elem(3, 3.0);
        let mut padding_lengths = Array1::<f64>::from_elem(3, 3.0);

        let mut nb_grid_points = Array1::<i32>::from_elem(3, 3);

        for (index_of_row, row) in atoms.positions.axis_iter(Axis(0)).enumerate() {
            let mut current_min = f64::INFINITY;
            for &val in row.iter() {
                if val < current_min {
                    current_min = val;
                }
            }
            origin[index_of_row] = current_min;
        }

        for (index_of_row, row) in atoms.positions.axis_iter(Axis(0)).enumerate() {
            let mut current_max = -f64::INFINITY;
            for &val in row.iter() {
                if val > current_max {
                    current_max = val;
                }
            }
            lengths[index_of_row] = current_max - origin[index_of_row];
        }

        for (index_of_i, i) in (&lengths / cutoff).iter().enumerate() {
            nb_grid_points[index_of_i] = i.ceil() as i32;

            //set to 1 if all atoms are in-plane
            if nb_grid_points[index_of_i] <= 0 {
                nb_grid_points[index_of_i] = 1;
            }
        }

        for (index_of_i, i) in nb_grid_points.iter().enumerate() {
            padding_lengths[index_of_i] = *i as f64 * cutoff - &lengths[index_of_i];
            origin[index_of_i] -= padding_lengths[index_of_i] / 2.0;
            lengths[index_of_i] += padding_lengths[index_of_i];
        }

        let mut r = atoms.positions.clone();
        for mut col in r.axis_iter_mut(Axis(1)) {
            col -= &origin;
        }

        for mut col in r.axis_iter_mut(Axis(1)) {
            col *= &(nb_grid_points.mapv(|nb| nb as f64) / &lengths);
        }

        let r = r.mapv(|i| i.floor() as i32);

        let atom_to_cell: Array1<i32> = r
            .axis_iter(Axis(1))
            .map(|col| Self::coordinate_to_index(col[0], col[1], col[2], nb_grid_points.view()))
            .collect();

        let mut sorted_atom_indices =
            Array1::from_vec((0..atom_to_cell.len()).collect()).into_raw_vec();

        //sort indices according to cell membership
        sorted_atom_indices.sort_by(|&i, &j| atom_to_cell[i].cmp(&atom_to_cell[j]));

        let mut cell_index: i32 = atom_to_cell[sorted_atom_indices[0]];
        let mut entry_index: i32 = 0;
        let mut binned_atoms: Vec<(i32, i32)> = vec![(cell_index, entry_index)];

        for i in 1..sorted_atom_indices.len() {
            if atom_to_cell[sorted_atom_indices[i]] != cell_index {
                cell_index = atom_to_cell[sorted_atom_indices[i]];
                entry_index = i as i32;
                binned_atoms.push((cell_index, entry_index));
            }
        }

        self.seed
            .conservative_resize(Dim(atoms.positions.shape()[1] + 1));

        let mut neighborhood = Array2::<i32>::zeros((3, 27));

        // Fill the array with combinations of x, y, z in the range -1 to 1
        for (i, (x, y, z)) in iproduct!(-1..=1, -1..=1, -1..=1).enumerate() {
            neighborhood[(0, i)] = x;
            neighborhood[(1, i)] = y;
            neighborhood[(2, i)] = z;
        }

        let mut n: usize = 0;
        let cutoff_sq = cutoff * cutoff;
        for i in 0..atoms.positions.shape()[1] {
            self.seed[i] = n as i32;

            let cell_coord = (&nb_grid_points.mapv(|nb_grid_pt| nb_grid_pt as f64)
                * (&atoms.positions.column(i) - &origin.view())
                / &lengths.view())
                .mapv(|coord_raw| coord_raw.floor() as i32);

            for shift in neighborhood.axis_iter(Axis(1)) {
                let neigh_cell_coord: Array1<i32> = &cell_coord.view() + &shift;

                //skip if cell is out of bounds
                if neigh_cell_coord.iter().position(|&x| x < 0) != None
                    || neigh_cell_coord
                        .iter()
                        .zip(&nb_grid_points)
                        .position(|(&neigh, &nb)| neigh >= nb)
                        != None
                {
                    continue;
                }
                let cell_index: i32 = Self::coordinate_to_index(
                    neigh_cell_coord[0],
                    neigh_cell_coord[1],
                    neigh_cell_coord[2],
                    nb_grid_points.view(),
                );

                //Find first entry within the cell neighbor list
                let res = binned_atoms.binary_search_by(|&(x, _)| {
                    if x == cell_index {
                        return Ordering::Equal;
                    } else if x < cell_index {
                        return Ordering::Less;
                    } else {
                        return Ordering::Greater;
                    }
                });

                let cell: (i32, i32) = match res {
                    Ok(val) => binned_atoms[val],
                    Err(_) => continue,
                };

                if cell.0 != cell_index {
                    continue;
                }

                let mut j = cell.1 as usize;
                while j < atom_to_cell.len() && atom_to_cell[sorted_atom_indices[j]] == cell_index {
                    let neighi = sorted_atom_indices[j];
                    if neighi == i {
                        j += 1;
                        continue;
                    }
                    let distance_vector =
                        &atoms.positions.column(i).view() - &atoms.positions.column(neighi).view();
                    let distance_sq = distance_vector.norm_l2().powi(2);

                    if distance_sq <= cutoff_sq {
                        if n >= self.neighbors.len() {
                            self.neighbors
                                .conservative_resize(Dim(2 * self.neighbors.len()));
                        }
                        self.neighbors[n] = neighi as i32;
                        n += 1;
                    }

                    j += 1;
                }
            }
        }
        self.seed[atoms.positions.shape()[1]] = n as i32;
        self.neighbors.conservative_resize(Dim(n));
        return (self.seed.clone(), self.neighbors.clone());
    }

    pub fn nb_total_neighbors(&self) -> i32 {
        return self.seed[self.seed.len() - 1];
    }

    pub fn nb_neighbors_of_atom(&self, i: usize) -> i32 {
        assert!(i < self.seed.len());
        return self.seed[i + 1] - self.seed[i];
    }

    fn coordinate_to_index(x: i32, y: i32, z: i32, nb_grid_points: ArrayView1<i32>) -> i32 {
        return x + nb_grid_points[0] * (y + nb_grid_points[1] * z);
    }
}

#[cfg(test)]
mod tests {
    use crate::md_implementation::atoms::Atoms;
    use itertools::assert_equal;
    use ndarray::{s, Array1, Array2};

    use super::NeighborList;

    #[test]
    fn test_neighbor_list_4_atoms() {
        let mut atoms = Atoms::new(4);
        let new_positions = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0];

        let new_positions_arr = Array2::from_shape_vec((3, 4), new_positions)
            .expect("Failed to create new positions array");
        atoms.positions.assign(&new_positions_arr);

        let mut neighbor_list: NeighborList = NeighborList::new();
        let (seed, neighbors) = neighbor_list.update(atoms, 1.5);

        assert_eq!(neighbor_list.nb_total_neighbors(), 10);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(0), 3);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(1), 3);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(2), 2);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(3), 2);

        assert_equal(
            neighbors.slice(s![seed[0]..seed[1]]).into_owned(),
            Array1::<i32>::from_vec(vec![3, 1, 2]),
        );
        assert_equal(
            neighbors.slice(s![seed[1]..seed[2]]).into_owned(),
            Array1::<i32>::from_vec(vec![3, 0, 2]),
        );
        assert_equal(
            neighbors.slice(s![seed[2]..seed[3]]).into_owned(),
            Array1::<i32>::from_vec(vec![0, 1]),
        );
        assert_equal(
            neighbors.slice(s![seed[3]..seed[4]]).into_owned(),
            Array1::<i32>::from_vec(vec![0, 1]),
        );
    }

    #[test]
    fn test_neighbor_list_4_atoms_first_atom_has_no_neighbor() {
        let mut atoms = Atoms::new(4);
        let new_positions = vec![0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 7.0, 6.0, 0.0, 0.0, 0.0, 0.0];

        let new_positions_arr = Array2::from_shape_vec((3, 4), new_positions)
            .expect("Failed to create new positions array");
        atoms.positions.assign(&new_positions_arr);

        let mut neighbor_list: NeighborList = NeighborList::new();
        let (seed, neighbors) = neighbor_list.update(atoms, 5.0);

        println!("neighbors: {:?}", neighbors);

        assert_eq!(neighbor_list.nb_total_neighbors(), 2);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(0), 0);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(1), 0);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(2), 1);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(3), 1);

        assert_equal(seed.clone(), Array1::<i32>::from_vec(vec![0, 0, 0, 1, 2]));
        assert_equal(neighbors.clone(), Array1::<i32>::from_vec(vec![3, 2]));
    }

    #[test]
    fn test_neighbor_list_3_atoms_last_atom_has_no_neighbor() {
        let mut atoms = Atoms::new(3);
        let new_positions = vec![0.0, 0.0, 0.0, 7.0, 6.0, 0.0, 0.0, 0.0, 0.0];

        let new_positions_arr = Array2::from_shape_vec((3, 3), new_positions)
            .expect("Failed to create new positions array");
        atoms.positions.assign(&new_positions_arr);

        let mut neighbor_list: NeighborList = NeighborList::new();
        let (seed, neighbors) = neighbor_list.update(atoms, 5.0);

        assert_eq!(neighbor_list.nb_total_neighbors(), 2);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(0), 1);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(1), 1);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(2), 0);
        assert_equal(seed.clone(), Array1::<i32>::from_vec(vec![0,1,2,2]));
        assert_equal(neighbors.clone(), Array1::<i32>::from_vec(vec![1, 0]));

    }

    #[test]
    fn test_neighbor_list_4_atoms_first_atoms_have_no_neighbor() {
        let mut atoms = Atoms::new(4);
        let new_positions = vec![0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 7.0, 6.0, 0.0, 0.0, 0.0, 0.0];

        let new_positions_arr = Array2::from_shape_vec((3, 4), new_positions)
            .expect("Failed to create new positions array");
        atoms.positions.assign(&new_positions_arr);

        let mut neighbor_list: NeighborList = NeighborList::new();
        let (seed, neighbors) = neighbor_list.update(atoms, 0.5);

        println!("neighbors: {:?}", neighbors);

        assert_eq!(neighbor_list.nb_total_neighbors(), 0);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(0), 0);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(1), 0);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(2), 0);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(3), 0);
    }
}

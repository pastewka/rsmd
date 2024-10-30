use crate::md_implementation::atoms::Atoms;
use itertools::iproduct;
use ndarray::{Array1, Array2, ArrayView1, Axis, Dim};
use ndarray_linalg::norm::Norm;

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
    pub fn update(&mut self, atoms: &mut Atoms, cutoff: f64) -> (Array1<i32>, Array1<i32>) {
        if atoms.positions.is_empty() {
            self.seed.conservative_resize(Dim(0));
            self.neighbors.conservative_resize(Dim(0));
            return (self.seed.clone(), self.neighbors.clone());
        }

        // Origin stores the bottom left corner of the enclosing rectangles and
        // lengths the three Cartesian lengths.

        let mut origin = Array1::<f64>::from_elem(3, 3.0);
        let mut lengths = Array1::<f64>::from_elem(3, 3.0);

        // This is the number of cells/grid points that fit into the enclosing
        // rectangle. The grid is such that a sphere of diameter *cutoff* fits into
        // each cell.
        let mut nb_grid_points = Array1::<i32>::from_elem(3, 3);

        // Compute box that encloses all atomic positions. Make sure that box
        // lengths are exactly divisible by the interaction range. Also compute the
        // number of cells in each Cartesian direction.
        for (pos_row, (origin_element, lengths_element)) in atoms
            .positions
            .axis_iter(Axis(0))
            .zip(origin.iter_mut().zip(lengths.iter_mut()))
        {
            //minimum of each positions row
            *origin_element = pos_row.iter().fold(f64::INFINITY, |accumulator, &element| {
                f64::min(accumulator, element)
            });
            *lengths_element = pos_row
                .iter()
                .fold(-f64::INFINITY, |accumulator, &element| {
                    f64::max(accumulator, element)
                })
                - *origin_element;
        }

        let l_by_cutoffs: Array1<i32> = (&lengths / cutoff)
            .iter()
            .map(|&l_by_cutoff| l_by_cutoff.ceil() as i32)
            .collect();

        for (l_by_cutoff, nb_grid_point) in l_by_cutoffs.iter().zip(nb_grid_points.iter_mut()) {
            *nb_grid_point = std::cmp::max(*l_by_cutoff, 1);
        }

        for (index, &nb_grid_point) in nb_grid_points.iter().enumerate() {
            let padding_length = nb_grid_point as f64 * cutoff - lengths[index];
            origin[index] -= padding_length / 2.0;
            lengths[index] += padding_length;
        }

        let mut r = atoms.positions.clone();
        let transform_factor = nb_grid_points.mapv(|nb_grid_pt| nb_grid_pt as f64) / &lengths;

        for mut r_col in r.axis_iter_mut(Axis(1)) {
            r_col -= &origin;
            r_col *= &transform_factor;
        }

        let r = r.mapv(|i| i.floor() as i32);

        let atom_to_cell: Array1<i32> = r
            .axis_iter(Axis(1))
            .map(|r_col| {
                Self::coordinate_to_index(r_col[0], r_col[1], r_col[2], nb_grid_points.view())
            })
            .collect();

        let mut sorted_atom_indices =
            Array1::from_vec((0..atom_to_cell.len()).collect()).into_raw_vec();

        //sort indices according to cell membership
        sorted_atom_indices.sort_by_key(|&i| atom_to_cell[i]);

        let mut previous_cell_index: i32 = atom_to_cell[sorted_atom_indices[0]];
        let mut entry_index: i32 = 0;
        let mut binned_atoms: Vec<(i32, i32)> = vec![(previous_cell_index, entry_index)];

        let nb_atoms = atoms.positions.shape()[1];

        for i in 1..sorted_atom_indices.len() {
            let current_cell_index = atom_to_cell[sorted_atom_indices[i]];

            if current_cell_index != previous_cell_index {
                previous_cell_index = atom_to_cell[sorted_atom_indices[i]];
                entry_index = i as i32;
                binned_atoms.push((current_cell_index, entry_index));
            }
        }

        self.seed.conservative_resize(Dim(nb_atoms + 1));

        let mut neighborhood = Array2::<i32>::zeros((3, 27));

        // Fill the array with combinations of x, y, z in the range -1 to 1
        for (i, (x, y, z)) in iproduct!(-1..=1, -1..=1, -1..=1).enumerate() {
            neighborhood[(0, i)] = x;
            neighborhood[(1, i)] = y;
            neighborhood[(2, i)] = z;
        }

        let mut neighbors_index: usize = 0;
        let cutoff_sq = cutoff * cutoff;
        for atom_index in 0..nb_atoms {
            self.seed[atom_index] = neighbors_index as i32;

            let cell_coord = (&nb_grid_points.mapv(|nb_grid_pt| nb_grid_pt as f64)
                * (&atoms.positions.column(atom_index) - &origin.view())
                / &lengths.view())
                .mapv(|coord_float| coord_float.floor() as i32);

            for shift in neighborhood.axis_iter(Axis(1)) {
                let neigh_cell_coord: Array1<i32> = &cell_coord.view() + &shift;

                //skip if cell is out of bounds
                if neigh_cell_coord
                    .iter()
                    .zip(&nb_grid_points)
                    .any(|(&neigh, &nb_grid_pt)| neigh < 0 || neigh >= nb_grid_pt)
                {
                    continue;
                }

                let current_cell_index: i32 = Self::coordinate_to_index(
                    neigh_cell_coord[0],
                    neigh_cell_coord[1],
                    neigh_cell_coord[2],
                    nb_grid_points.view(),
                );

                let find_first_entry_of_cell_result = binned_atoms
                    .binary_search_by_key(&current_cell_index, |&(binned_atoms_cell_index, _)| {
                        binned_atoms_cell_index
                    });

                let mut entry_index: usize = match find_first_entry_of_cell_result {
                    Ok(val) => binned_atoms[val].1 as usize,
                    Err(_no_entry_found) => continue,
                };

                while entry_index < atom_to_cell.len()
                    && atom_to_cell[sorted_atom_indices[entry_index]] == current_cell_index
                {
                    let potential_neighbor_index = sorted_atom_indices[entry_index];

                    let atom_is_own_neighbor = potential_neighbor_index == atom_index;
                    if atom_is_own_neighbor {
                        entry_index += 1;
                        continue;
                    }

                    let distance_vector = &atoms.positions.column(atom_index)
                        - &atoms.positions.column(potential_neighbor_index);
                    let distance_sq = distance_vector.norm_l2().powi(2);

                    if distance_sq <= cutoff_sq {
                        if neighbors_index >= self.neighbors.len() {
                            self.neighbors
                                .conservative_resize(Dim(2 * self.neighbors.len()));
                        }
                        self.neighbors[neighbors_index] = potential_neighbor_index as i32;
                        neighbors_index += 1;
                    }

                    entry_index += 1;
                }
            }
        }
        self.seed[nb_atoms] = neighbors_index as i32;
        self.neighbors.conservative_resize(Dim(neighbors_index));
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
        let (seed, neighbors) = neighbor_list.update(&mut atoms, 1.5);

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
        let (seed, neighbors) = neighbor_list.update(&mut atoms, 5.0);

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
        let (seed, neighbors) = neighbor_list.update(&mut atoms, 5.0);

        assert_eq!(neighbor_list.nb_total_neighbors(), 2);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(0), 1);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(1), 1);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(2), 0);
        assert_equal(seed.clone(), Array1::<i32>::from_vec(vec![0, 1, 2, 2]));
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
        let (_seed, neighbors) = neighbor_list.update(&mut atoms, 0.5);

        println!("neighbors: {:?}", neighbors);

        assert_eq!(neighbor_list.nb_total_neighbors(), 0);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(0), 0);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(1), 0);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(2), 0);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(3), 0);
    }
}

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
    pub fn update(&mut self, atoms: Atoms, cutoff: f64) -> (Array1<i32>, Array1<i32>) {
        println!("positions init: {:?}", atoms.positions);
        if atoms.positions.is_empty() {
            self.seed.conservative_resize(Dim(0));
            self.neighbors.conservative_resize(Dim(0));
            return (self.seed.clone(),self.neighbors.clone());
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
            nb_grid_points[index_of_i] = i.ceil() as i32;

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

        let r = r.mapv(|i| i.floor() as i32);
        println!("r input to function atom_to_cell: {:?}", r);

        let atom_to_cell: Array1<i32> = r
            .axis_iter(Axis(1))
            .map(|col| Self::coordinate_to_index(col[0], col[1], col[2], nb_grid_points.view()))
            .collect();
        println!("atom_to_cell: {:?}", atom_to_cell);

        let mut sorted_atom_indices =
            Array1::from_vec((0..atom_to_cell.len()).collect()).into_raw_vec();
        println!(
            "sorted_atom_indices with ingredients: {:?}",
            sorted_atom_indices
        );

        //sort indices according to cell membership
        sorted_atom_indices.sort_by(|&i, &j| atom_to_cell[i].cmp(&atom_to_cell[j]));
        println!(
            "sorted_atom_indices actually sorted: {:?}",
            sorted_atom_indices
        );

        let mut cell_index: i32 = atom_to_cell[sorted_atom_indices[0]];
        let mut entry_index: i32 = 0;
        let mut binned_atoms: Vec<(i32, i32)> = vec![(cell_index, entry_index)];

        println!("binned_atoms at initialization: {:?}", binned_atoms);

        for i in 1..sorted_atom_indices.len() {
            if atom_to_cell[sorted_atom_indices[i]] != cell_index {
                cell_index = atom_to_cell[sorted_atom_indices[i]];
                entry_index = i as i32;
                binned_atoms.push((cell_index, entry_index));
            }
        }

        println!("binned_atoms: {:?}", binned_atoms);

        println!("atoms.positions.len(): {}", atoms.positions.len());
        self.seed
            .conservative_resize(Dim(atoms.positions.shape()[1] + 1));

        let mut neighborhood = Array2::<i32>::zeros((3, 27));

        // Fill the array with combinations of x, y, z in the range -1 to 1
        for (i, (x, y, z)) in iproduct!(-1..=1, -1..=1, -1..=1).enumerate() {
            neighborhood[(0, i)] = x;
            neighborhood[(1, i)] = y;
            neighborhood[(2, i)] = z;
        }

        println!("Neighborhood: {:?}", neighborhood);

        let mut n: usize = 0;
        let cutoff_sq = cutoff * cutoff;
                    let mut cou: i32 = 0;
        for i in 0..atoms.positions.shape()[1] {
            self.seed[i] = n as i32;

            println!("=================\ncurrent seed: {:?}\ncurrent neighbors: {:?}\n======================",self.seed,self.neighbors);



            println!(
                "column: {:?};\norigin: {:?};\nlengths: {:?}",
                &atoms.positions.column(i),
                &origin.view(),
                &lengths.view()
            );
            let cell_coord = (&nb_grid_points.mapv(|nb_grid_pt| nb_grid_pt as f64)
                * (&atoms.positions.column(i) - &origin.view())
                / &lengths.view())
                .mapv(|coord_raw| coord_raw.floor() as i32);
            println!("cell_coord: {:?}", cell_coord);
            //let c = cell_coord.mapv(|c| c.floor() as i32);
            //println!("cell_coord after flooring: {:?}",c);

            for shift in neighborhood.axis_iter(Axis(1)) {
                println!("new iteration in for neighboring cells .....");
                let neigh_cell_coord: Array1<i32> = &cell_coord.view() + &shift;

                //skip if cell is out of bounds
                if neigh_cell_coord.iter().position(|&x| x < 0) != None
                    || neigh_cell_coord
                        .iter()
                        .zip(&nb_grid_points)
                        .position(|(&neigh, &nb)| neigh >= nb)
                        != None
                {
                    cou+=1;
                    if cou == 12{
                        println!("HERE...");
                    }
                    println!("CONTINUE --- Cell out of bounds; neigh cell coords: {:?}",neigh_cell_coord);
                    continue;
                }
                let cell_index: i32 = Self::coordinate_to_index(
                    neigh_cell_coord[0],
                    neigh_cell_coord[1],
                    neigh_cell_coord[2],
                    nb_grid_points.view(),
                );
                println!("cell_index: {}", cell_index);

                println!("binned_atoms: {:?}", binned_atoms);

let cell_test = binned_atoms[binned_atoms
                .binary_search_by((|&(x, _)| {
                    if x < cell_index {
                        return Ordering::Less;
                    } else {
                        return Ordering::Greater;
                    }
                })).unwrap_or_else(|x| x)];

                println!("cell_test: {:?}",cell_test);

                //Find first entry within the cell neighbor list
                let res = 
                // skip_if_no_lower_bound!(
                    binned_atoms
                    .binary_search_by(|&(x, _)| {
                         if x == cell_index{
                            return Ordering::Equal;
                        }else if x < cell_index {
                            return Ordering::Less;
                        } else{
                            return Ordering::Greater;
                        }
                    });

                    let cell:(i32,i32) = match res{
                        Ok(val) => binned_atoms[val],
                        Err(_) => continue,
                    };
                // );

                // let cell: (i32, i32) = (5,20);//binned_atoms[res];

                println!("lower_bound cell: {:?}", cell);
                println!("binned_atoms.end() quasi: {:?}",binned_atoms[0]);

                // if cell == binned_atoms[0] || 
                if cell.0 != cell_index {
                    println!("CONTINUE --- cell == *binned_atoms.last().unwrap() || cell.0 != cell_index");
                    continue;
                }
                let mut count = 0;
                    println!("sorted_atom_indices: {:?}",sorted_atom_indices);
                    let mut j = cell.1 as usize;
                        println!("j before while : {}",j);
                    while j < atom_to_cell.len()
                        && atom_to_cell[sorted_atom_indices[j]] == cell_index
                    {
                        println!("j: {}",j);
                        let neighi = sorted_atom_indices[j];
                        println!("inside while loop: index j: {}; neighi=sorted_atom_indices: {}",j, neighi);
                        if neighi == i {
                            println!("CONTINUE --- neighi==i (inside while)");
                            count +=1;
                            // if count == 2{
                                // println!("RETURN at neighi continue... ------------------------------------------------------------------------------------------");
                            // return (self.seed.clone(),self.neighbors.clone());
                            // }
                            j+=1;
                            continue;
                        }
                        let distance_vector = &atoms.positions.column(i).view()
                            - &atoms.positions.column(neighi).view();
                        println!("distance_vector: {:?}",distance_vector);
                        let distance_sq = distance_vector.norm_l2().powi(2);
                        println!("distance_sq: {}; cutoff_sq: {}",distance_sq,cutoff_sq);

                        if distance_sq <= cutoff_sq {
                            if n >= self.neighbors.len() {
                                self.neighbors.conservative_resize(Dim(2 * self.neighbors.len()));
                            }
                            println!("neighbors[{}]={}",n,neighi);
                            self.neighbors[n] = neighi as i32;
                            n+=1;
                        }

                        j += 1;
                    }
                    println!("Outside while loop-----");
            }
        }
        self.seed[atoms.positions.shape()[1]] = n as i32;
        self.neighbors.conservative_resize(Dim(n));
        return (self.seed.clone(),self.neighbors.clone());
        // return (Array1::from_elem(5, 0), Array1::from_elem(5, 0));
    }

    pub fn nb_total_neighbors(&self)-> i32{
        return self.seed[self.seed.len()-1];
    }

    pub fn nb_neighbors_of_atom(&self, i: usize)-> i32{
        assert!(i<self.seed.len());
        return self.seed[i+1]-self.seed[i];
    }

    fn coordinate_to_index_from_array(
        c: Array1<i32>,
        nb_grid_points: ArrayView1<i32>,
    ) -> Array1<i32> {
        println!("nb_grid_pts(0): {}", nb_grid_points[0]);
        println!("nb_grid_pts: {}", nb_grid_points);
        Self::coordinate_to_index(c[0], c[1], c[2], nb_grid_points);
        // c.row_mut(2).mapv_inplace(|z| z*nb_grid_points[1]);
        // c.row_mut(2).zip().mapv_inplace(|grid_pts_times_z| grid_pts_times_z +c.row(1));

        // c.row(0) += nb_grid_points[0];
        return Array1::from_elem(4, 2); // * (c.row(1) + nb_grid_points[1] * c.row(2));
    }

    fn coordinate_to_index(x: i32, y: i32, z: i32, nb_grid_points: ArrayView1<i32>) -> i32 {
        return x + nb_grid_points[0] * (y + nb_grid_points[1] * z);
    }
}

#[cfg(test)]
mod tests {
    use crate::md_implementation::atoms::Atoms;
    use googletest::{assert_that, matchers::near, verify_that};
    use itertools::assert_equal;
    use ndarray::{Array1,Array2, s};
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use rand_isaac::isaac64::Isaac64Rng;

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

        assert_eq!(neighbor_list.nb_total_neighbors(),10);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(0),3);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(1),3);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(2),2);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(3),2);

        // println!("neighbors: {:?}; seed[0]: {}, seed[1]: {}\nslice: {:?}",neighbors, seed[0],seed[1], neighbors.slice(s![seed[0]..seed[1]]));
        assert_equal(neighbors.slice(s![seed[0]..seed[1]]).into_owned(), Array1::<i32>::from_vec(vec![3,1,2]));
        assert_equal(neighbors.slice(s![seed[1]..seed[2]]).into_owned(), Array1::<i32>::from_vec(vec![3,0,2]));
        assert_equal(neighbors.slice(s![seed[2]..seed[3]]).into_owned(), Array1::<i32>::from_vec(vec![0,1]));
        assert_equal(neighbors.slice(s![seed[3]..seed[4]]).into_owned(), Array1::<i32>::from_vec(vec![0,1]));
    }

    #[test]
    fn test_neighbor_list_3_atoms_first_atom_has_no_neighbor() {
        let mut atoms = Atoms::new(4);
        let new_positions = vec![0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 7.0, 6.0, 0.0, 0.0, 0.0, 0.0];

        let new_positions_arr = Array2::from_shape_vec((3, 4), new_positions)
            .expect("Failed to create new positions array");
        atoms.positions.assign(&new_positions_arr);

        let mut neighbor_list: NeighborList = NeighborList::new();
        let (seed, neighbors) = neighbor_list.update(atoms, 5.0);

        println!("neighbors: {:?}",neighbors);

        assert_eq!(neighbor_list.nb_total_neighbors(),2);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(0),0);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(1),0);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(2),1);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(3),1);
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

        assert_eq!(neighbor_list.nb_total_neighbors(),2);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(0),1);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(1),1);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(2),0);
    }
}

//update neighbor list from the particle positions stored in the 'atoms' argument

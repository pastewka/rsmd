use crate::md_implementation::atoms::Atoms;
use bigint::U256;
use core::arch::x86_64::_pdep_u64;
use itertools::iproduct;
use ndarray::s;
use ndarray::{Array1, Array2, ArrayView1, Axis, Dim};
use ndarray_linalg::norm::Norm;
use std::cmp::Ordering;

use super::atoms::ArrayExt;

pub struct NeighborListZ {
    seed: Array1<i32>,
    neighbors: Array1<i32>,
}

impl NeighborListZ {
    pub fn new() -> Self {
        let seed = Array1::from_elem(1, 1);
        let neighbors = Array1::from_elem(1, 1);
        return Self { seed, neighbors };
    }

    //update neighbor list from the particle positions stored in the 'atoms' argument
    pub fn update(
        &mut self,
        atoms: &mut Atoms,
        cutoff: f64,
        sort_atoms_array: bool,
    ) -> (Array1<i32>, Array1<i32>) {
        if atoms.positions.is_empty() {
            self.seed.conservative_resize(Dim(0));
            self.neighbors.conservative_resize(Dim(0));
            return (self.seed.clone(), self.neighbors.clone());
        }

        // Origin stores the bottom left corner of the enclosing rectangles and
        // lengths the three Cartesian lengths.

        let mut origin = Array1::<f64>::from_elem(3, 3.0);
        let mut lengths = Array1::<f64>::from_elem(3, 3.0);
        let mut padding_lengths = Array1::<f64>::from_elem(3, 3.0);

        // This is the number of cells/grid points that fit into the enclosing
        // rectangle. The grid is such that a sphere of diameter *cutoff* fits into
        // each cell.
        let mut nb_grid_points = Array1::<i32>::from_elem(3, 3);

        // Compute box that encloses all atomic positions. Make sure that box
        // lengths are exactly divisible by the interaction range. Also compute the
        // number of cells in each Cartesian direction.
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

        //compute cell index to store in e.g. atom_to_cell[0] the cell_index of atom 0
        let atom_to_cell: Array1<i32> = r
            .axis_iter(Axis(1))
            .map(|col| Self::coordinate_to_index(col[0], col[1], col[2], nb_grid_points.view()))
            .collect();

        //create handle that stores the key-value pairs: (key=morton-code(cell), value=atom_index)
        let mut handles: Vec<(U256, usize)> = Vec::new();
        let nb_atoms = atoms.positions.shape()[1];

        for i in 0..nb_atoms {
            handles.push((
                U256::from(morton_encode_cell(Self::i32_to_u64_order_preserving(
                    atom_to_cell[i],
                ))),
                i,
            ));
        }

        insertion_sort(&mut handles);

        //store atoms in memory according to handle order
        if sort_atoms_array {
            let mut positions = Array2::zeros((3, nb_atoms));
            println!("old handles {:?}", handles);
            println!("old atoms.positions: {:?}", atoms.positions);
            let h = handles.clone();
            for (new_index, (_, original_index)) in h.iter().enumerate() {
                positions
                    .slice_mut(s![.., new_index])
                    .assign(&atoms.positions.slice(s![.., *original_index]));
                handles[new_index].1 = new_index;
            }
            atoms.positions.assign(&positions);
            println!("after reordering atoms.positions: {:?}", atoms.positions);
            println!("new handles {:?}", handles);
        }

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

        self.seed.conservative_resize(Dim(nb_atoms + 1));

        // Constructing index shift vectors to look for neighboring cells
        let mut neighborhood = Array2::<i32>::zeros((3, 27));

        // Fill the array with combinations of x, y, z in the range -1 to 1
        for (i, (x, y, z)) in iproduct!(-1..=1, -1..=1, -1..=1).enumerate() {
            neighborhood[(0, i)] = x;
            neighborhood[(1, i)] = y;
            neighborhood[(2, i)] = z;
        }

        let mut n: usize = 0;
        let cutoff_sq = cutoff * cutoff;
        for i in 0..nb_atoms {
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
        self.seed[nb_atoms] = n as i32;
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

    fn i32_to_u64_order_preserving(value: i32) -> u64 {
        return value.wrapping_sub(i32::MIN) as u64; // For negative numbers: invert all bits
    }
}

pub fn f64_to_u128_order_preserving(value: f64) -> u128 {
    let bits = value.to_bits(); // Get the raw bit pattern of the f64 (as u128)

    // If the number is negative, we flip all the bits to invert the order.
    // If the number is positive, we flip only the sign bit to maintain order.
    if bits & 0x8000000000000000 != 0 {
        return !bits as u128; // For negative numbers: invert all bits
    } else {
        return bits as u128 + 0x8000000000000000; // For positive numbers: flip the sign bit
    }
}

fn spread(v: u64) -> U256 {
    let original_amount_of_ones = v.count_ones();
    let mut value = U256::from(v);

    let mask_low = 0x9249249249249249_u64;
    let mask_middle = 0x4924924924924924_u64;
    let mask_high = 0x2492492492492492_u64;

    let low_64 = unsafe { _pdep_u64(v as u64, mask_low) };
    let middle_64 = unsafe { _pdep_u64((v >> 22) as u64, mask_middle) };
    let high_64 = unsafe { _pdep_u64((v >> 43) as u64, mask_high) };

    value = (U256::from(high_64) << 128) | (U256::from(middle_64) << 64) | U256::from(low_64);

    return value;
}

pub fn morton_encode_cell(cell_index: u64) -> U256 {
    return spread(cell_index);
}

pub fn insertion_sort(data: &mut Vec<(U256, usize)>) {
    for i in 1..data.len() {
        let mut j = i;
        let current = data[i].clone();

        // Compare current key with the keys in the sorted portion (left of index i)
        // Shift elements to the right until correct position for `current`
        while j > 0 && data[j - 1].0 > current.0 {
            data[j] = data[j - 1].clone(); // Shift element to the right
            j -= 1;
        }

        // Insert `current` at its correct position
        data[j] = current.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bigint::{U128, U256};
    use itertools::assert_equal;
    use ndarray::Array2;
    use rand::Rng;

    #[test]
    fn test_neighbor_list_4_atoms() {
        let mut atoms = Atoms::new(4);
        let new_positions = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0];

        let new_positions_arr = Array2::from_shape_vec((3, 4), new_positions)
            .expect("Failed to create new positions array");
        atoms.positions.assign(&new_positions_arr);

        let mut neighbor_list: NeighborListZ = NeighborListZ::new();
        let (seed, neighbors) = neighbor_list.update(&mut atoms, 1.5, false);

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

        let mut neighbor_list: NeighborListZ = NeighborListZ::new();
        let (seed, neighbors) = neighbor_list.update(&mut atoms, 5.0, false);

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
        let original_positions = atoms.positions.clone();

        let mut neighbor_list: NeighborListZ = NeighborListZ::new();

        //test without resorting the original atoms array in memory
        let (seed, neighbors) = neighbor_list.update(&mut atoms, 5.0, false);

        assert_eq!(atoms.positions, original_positions);
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
        let atoms_before = atoms.clone();

        let mut neighbor_list: NeighborListZ = NeighborListZ::new();
        let (seed, neighbors) = neighbor_list.update(&mut atoms, 0.5, false);
        assert_equal(atoms.positions, atoms_before.positions);

        println!("neighbors: {:?}", neighbors);

        assert_eq!(neighbor_list.nb_total_neighbors(), 0);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(0), 0);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(1), 0);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(2), 0);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(3), 0);
    }

    fn count_ones(value: U256) -> u64 {
        let mut count = 0u64;
        for i in 0..value.bits() {
            if value.bit(i) {
                count += 1;
            }
        }
        return count;
    }

    fn check_interleaved_by_two(spread_value: U256) -> bool {
        let amount_of_ones = count_ones(spread_value);
        let mut count_checked_ones: u64 = 0;

        for i in 0..spread_value.bits() {
            if i % 3 == 0 && i + 2 != spread_value.bits() {
                if spread_value.bit(i + 1) || spread_value.bit(i + 2) {
                    return false;
                }
                if spread_value.bit(i) {
                    count_checked_ones += 1;
                }
            }

            if count_checked_ones == amount_of_ones {
                return true;
            }
        }
        return false;
    }

    fn morton_decode(morton_code: U256) -> U256 {
        let mut result = U256::zero();
        let mut amount_to_shift_for_endianness_swap = 0u32;

        let mut modified_morton = morton_code.clone();

        while modified_morton.bits() >= 4 {
            let before = result.clone();
            println!(
                "result before: {:b}; morton code: {:b}",
                before.bits(),
                modified_morton.bits()
            );
            if modified_morton.bit(0) {
                result = (result << 1) | U256::one();
            } else {
                result = result << 1;
            }
            if result == before {
                // need of shift when swapping endianness, because leading zeros get truncated
                amount_to_shift_for_endianness_swap += 1;
            }

            modified_morton = modified_morton >> 3;

            println!(
                "result after: {:b}; morton code: {:b}",
                result.bits(),
                modified_morton.bits()
            );
            println!("modified_morton.bits(): {}", modified_morton.bits());
        }
        if modified_morton.bits() > 0 {
            if modified_morton.bit(0) {
                result = (result << 1) | U256::one();
            } else {
                result = result << 1;
            }
        }
        println!(
            "morton decode result before swapping endianness: \n{:b}; morton code: \n{:b}",
            result.bits(),
            modified_morton.bits()
        );

        let mut reversed_result = U256::zero();
        while result.bits() > 0 {
            if result.bit(0) {
                reversed_result = (reversed_result << 1) | U256::one();
            } else {
                reversed_result = reversed_result << 1;
            }
            result = result >> 1;
        }
        result = reversed_result << usize::try_from(amount_to_shift_for_endianness_swap).unwrap();

        println!("final demortanized result: \n{:b}", result.bits());

        return result;
    }

    #[test]
    fn test_check_interleaved_by_two() {
        //check_interleaved_by_two() should check, if the provided number is interleaved by 2, no matter if the bit of the original value was 0 or 1.
        let a = U256::from(0b1001001001001001u32);
        assert!(a.bit(15));
        assert!(check_interleaved_by_two(U256::from(0b1001001001001u32))); // original number: 0b..000011111
        assert!(check_interleaved_by_two(U256::from(0b1001001001001000u32))); // original number: 0b..0000111110
                                                                              // assert!(check_interleaved_by_two(U256::from(
                                                                              //     0b0000000000001001001001001000u128
                                                                              // ))); // original number: 0b..0000111110
        assert!(check_interleaved_by_two(U256::from(0b0001001001u32))); // original number: 0b..0000111
                                                                        // assert!(check_interleaved_by_two(U256::from(
                                                                        //     0x200200200200200200200200u128
                                                                        // ))); // original number: 0x88888888

        assert!(!check_interleaved_by_two(U256::from(0b100100100100100u32)));
        assert!(!check_interleaved_by_two(U256::from(0b101001001001u32)));
        assert!(!check_interleaved_by_two(U256::from(0b000100001u32)));
        assert!(!check_interleaved_by_two(U256::from(0b000100101u32)));
        assert!(!check_interleaved_by_two(U256::from(0b0011001001u32)));
    }

    #[test]
    fn test_spread_with_check_interleaved_by_two() {
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let x = rng.gen::<u64>();
            let spread = spread(x);
            println!("x: {:#0x}; spread(x): {:#0x}", x, spread);
            assert!(check_interleaved_by_two(spread));
        }
    }

    #[test]
    fn test_i32_to_u64_order_preserving() {
        let testvalues = [
            (i32::MIN, i32::MIN + 1),
            (i32::MIN + 1, i32::MIN + 2),
            (0i32 - 1, 0i32),
            (0i32, 0i32 + 1),
            (-1i32, 1i32),
            (-10002323i32, -10002323i32 + 1),
            (10002323i32, 10002323i32 + 1),
            (i32::MAX - 1, i32::MAX),
            (i32::MIN, i32::MAX),
        ];

        for (i, j) in testvalues {
            assert!(
                ((i >= j)
                    && (NeighborListZ::i32_to_u64_order_preserving(i)
                        >= NeighborListZ::i32_to_u64_order_preserving(j)))
                    || ((i < j)
                        && (NeighborListZ::i32_to_u64_order_preserving(i)
                            < NeighborListZ::i32_to_u64_order_preserving(j)))
            );
        }
    }

    #[test]
    fn test_insertion_sort() {
        let mut map = vec![
            (U256::from(U128::MAX).pow(U256::from(2)), 0),
            (U256::from(U128::MAX).pow(U256::from(2)) - U256::one(), 1),
            (U256::from(1u8), 2),
            (U256::from(200u8), 3),
            (U256::from(U128::MAX), 4),
        ];
        insertion_sort(&mut map);

        assert_eq!(map[0], (U256::from(1u8), 2));
        assert_eq!(map[1], (U256::from(200u8), 3));
        assert_eq!(map[2], (U256::from(U128::MAX), 4));
        assert_eq!(
            map[3],
            (U256::from(U128::MAX).pow(U256::from(2)) - U256::one(), 1)
        );
        assert_eq!(map[4], (U256::from(U128::MAX).pow(U256::from(2)), 0));
    }

    #[test]
    fn test_morton_decode() {
        let morton_code = U256::from(0b1_001_001_000_000_001u64);
        let decoded_result = morton_decode(morton_code);

        assert_eq!(decoded_result, U256::from(0b111001u32));
    }

    #[test]
    fn test_morton_encode_decode_cell() {
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let x = rng.gen::<i32>();
            let spread = spread(NeighborListZ::i32_to_u64_order_preserving(x));
            println!("x: {:#0x}; spread(x): {:#0x}", x, spread);
            assert_eq!(
                NeighborListZ::i32_to_u64_order_preserving(x),
                morton_decode(spread).as_u64()
            );
        }
    }
}

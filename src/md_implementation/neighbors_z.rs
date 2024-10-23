use crate::md_implementation::atoms::Atoms;
use itertools::iproduct;
use ndarray::array;
use ndarray::s;
use ndarray::{Array1, Array2, ArrayView1, Axis, Dim};
use ndarray_linalg::norm::Norm;
use num::BigUint;
use num::ToPrimitive;
use num::Zero;
use std::cmp::Ordering;
use core::arch::x86_64::_pdep_u64;

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
        mut atoms: Atoms,
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

        println!("atom_to_cell: {:?}", atom_to_cell);

        //create handle that stores the key-value pairs: (key=morton-code(cell), value=atom_index)
        let mut handles: Vec<(BigUint, usize)> = Vec::new();
        let nb_atoms = atoms.positions.shape()[1];

        for i in 0..nb_atoms {
            handles.push((
                BigUint::from(morton_encode_cell(Self::i32_to_u64_order_preserving(
                    atom_to_cell[i],
                ))),
                i,
            ));
        }

        insertion_sort(&mut handles);

        //store atoms in memory according to handle order
        if sort_atoms_array {
            let mut positions = Array2::zeros((3, nb_atoms));
            for (new_index, (_, original_index)) in handles.iter().enumerate() {
                positions
                    .slice_mut(s![.., new_index])
                    .assign(&atoms.positions.slice(s![.., *original_index]));
            }
            atoms.positions.assign(&positions);
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

fn spread(v: u64) -> BigUint {

    println!(" value before masking: \n{:#0x}", v);

    let original_amount_of_ones = v.count_ones();

        let mut value = BigUint::from(v);
        
        println!("value before pdep: {:#0x}",v);
        let mask_low = 0x9249249249249249_u64;
        println!("mask: {:#0x}",mask_low);
        let mask_middle = 0x4924924924924924_u64;
        println!("mask: {:#0x}",mask_middle);
        let mask_high = 0x2492492492492492_u64;
        println!("mask: {:#0x}",mask_high);

        let low_64 = unsafe { _pdep_u64(v as u64, mask_low) };
        println!("low_64: {:#0x}",low_64);
        println!("(v>>22): {:#0x}",(v>>22));
        let middle_64 = unsafe { _pdep_u64((v>>22) as u64, mask_middle) };
        println!("middle_64: {:#0x}",middle_64);
        println!("(v>>42): {:#0x}",(v>>42));
        let high_64 = unsafe { _pdep_u64((v>>43) as u64, mask_high) };
        println!("high_64: {:#0x}",high_64);

        value = (BigUint::from(high_64) << 128) | (BigUint::from(middle_64) << 64) | BigUint::from(low_64);

        println!("value: {:#0x}",value);

        assert_eq!(value.count_ones() as u32, original_amount_of_ones);
        return value;
}

pub fn combine_spread(x_spread: BigUint, y_spread: BigUint, z_spread: BigUint) -> BigUint {
    let result = x_spread | y_spread << 1 | z_spread << 2;

    return result;
}

pub fn morton_encode_cell(cell_index: u64) -> BigUint {
    //TODO: make all parameters only u64 not with into()
    println!("cell_index: {:b}",cell_index);
    return spread(cell_index);
}

pub fn insertion_sort(data: &mut Vec<(BigUint, usize)>) {
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
    use itertools::assert_equal;
    use ndarray::{array, Array2};
    use num::BigUint;
    use num::One;
    use num::Zero;
    use rand::Rng;
    use std::mem;

    #[test]
    fn test_neighbor_list_4_atoms() {
        let mut atoms = Atoms::new(4);
        let new_positions = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0];

        let new_positions_arr = Array2::from_shape_vec((3, 4), new_positions)
            .expect("Failed to create new positions array");
        atoms.positions.assign(&new_positions_arr);

        let mut neighbor_list: NeighborListZ = NeighborListZ::new();
        let (seed, neighbors) = neighbor_list.update(atoms, 1.5, false);

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
        let (seed, neighbors) = neighbor_list.update(atoms, 5.0, false);

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

        let mut neighbor_list: NeighborListZ = NeighborListZ::new();
        let (seed, neighbors) = neighbor_list.update(atoms, 5.0, false);

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

        let mut neighbor_list: NeighborListZ = NeighborListZ::new();
        let (seed, neighbors) = neighbor_list.update(atoms, 0.5, false);

        println!("neighbors: {:?}", neighbors);

        assert_eq!(neighbor_list.nb_total_neighbors(), 0);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(0), 0);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(1), 0);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(2), 0);
        assert_eq!(neighbor_list.nb_neighbors_of_atom(3), 0);
    }

    fn check_interleaved_by_two(spread_value: BigUint) -> bool {
        let amount_of_ones = spread_value.count_ones();
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

    fn morton_decode_3d(morton_code: BigUint) -> (BigUint, BigUint, BigUint) {
        let mut b = BigUint::zero();
        let mut result_array = array![BigUint::zero(), BigUint::zero(), BigUint::zero()];
        let mut amount_to_shift_for_endianness_swap = array![0u32, 0u32, 0u32];
        for dimension in 0..3 {
            println!("dimension: {}", dimension);

            let mut modified_morton = morton_code.clone();
            modified_morton >>= dimension;

            while modified_morton.bits() >= 4 {
                let before = result_array[dimension].clone();
                println!(
                    "result before: {:b}; morton code: {:b}",
                    before, modified_morton
                );
                if modified_morton.bit(0) {
                    result_array[dimension] = (&result_array[dimension] << 1) | BigUint::one();
                } else {
                    result_array[dimension] = &result_array[dimension] << 1;
                }
                if result_array[dimension] == before {
                    // need of shift when swapping endianness, because leading zeros get truncated
                    amount_to_shift_for_endianness_swap[dimension] += 1;
                }

                modified_morton >>= 3;

                println!(
                    "result after: {:b}; morton code: {:b}",
                    result_array[dimension], modified_morton
                );
                println!("modified_morton.bits(): {}", modified_morton.bits());
            }
            if modified_morton.bits() > 0 {
                if modified_morton.bit(0) {
                    result_array[dimension] = (&result_array[dimension] << 1) | BigUint::one();
                } else {
                    result_array[dimension] = &result_array[dimension] << 1;
                }
            }
            println!(
                "morton decode result before swapping endianness: \n{:b}; morton code: \n{:b}",
                result_array[dimension], modified_morton
            );
        }

        for i in 0..result_array.len() {
            let mut reversed_result = BigUint::zero();
            while result_array[i].bits() > 0 {
                if result_array[i].bit(0) {
                    reversed_result = (&reversed_result << 1) | BigUint::one();
                } else {
                    reversed_result = &reversed_result << 1;
                }
                result_array[i] >>= 1;
            }
            result_array[i] = reversed_result << amount_to_shift_for_endianness_swap[i];
        }
        println!(
            "final demortanized results: \n{:b},\n{:b},\n{:b}",
            result_array[0], result_array[1], result_array[2]
        );

        return (
            result_array[0].clone(),
            result_array[1].clone(),
            result_array[2].clone(),
        );
    }

    #[test]
    fn test_check_interleaved_by_two() {
        //check_interleaved_by_two() should check, if the provided number is interleaved by 2, no matter if the bit of the original value was 0 or 1.
        let a = BigUint::from(0b1001001001001001u32);
        assert!(a.bit(15));
        assert!(check_interleaved_by_two(BigUint::from(0b1001001001001u32))); // original number: 0b..000011111
        assert!(check_interleaved_by_two(BigUint::from(
            0b1001001001001000u32
        ))); // original number: 0b..0000111110
        assert!(check_interleaved_by_two(BigUint::from(
            0b0000000000001001001001001000u128
        ))); // original number: 0b..0000111110
        assert!(check_interleaved_by_two(BigUint::from(0b0001001001u32))); // original number: 0b..0000111
        assert!(check_interleaved_by_two(BigUint::from(
            0x200200200200200200200200u128
        ))); // original number: 0x88888888

        assert!(!check_interleaved_by_two(BigUint::from(
            0b100100100100100u32
        )));
        assert!(!check_interleaved_by_two(BigUint::from(0b101001001001u32)));
        assert!(!check_interleaved_by_two(BigUint::from(0b000100001u32)));
        assert!(!check_interleaved_by_two(BigUint::from(0b000100101u32)));
        assert!(!check_interleaved_by_two(BigUint::from(0b0011001001u32)));
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
    fn test_f64_to_u128_order_preserving() {
        let testvalues = [
            (f64::MIN, f64::MIN + f64::EPSILON),
            (f64::MIN + 1.0, f64::MIN + 1.0 + f64::EPSILON),
            (0.0f64 - f64::EPSILON, 0.0f64),
            (0.0f64, 0.0f64 + f64::EPSILON),
            (10002323f64, 10002323f64 + f64::EPSILON),
            (f64::MAX - f64::EPSILON, f64::MAX),
            (f64::MIN, f64::MAX),
        ];

        for (i, j) in testvalues {
            assert!(
                ((i >= j) && (f64_to_u128_order_preserving(i) >= f64_to_u128_order_preserving(j)))
                    || ((i < j)
                        && (f64_to_u128_order_preserving(i) < f64_to_u128_order_preserving(j)))
            );
        }
    }

    #[test]
    fn test_insertion_sort() {
        let mut map = vec![
            (BigUint::from(u128::MAX).pow(2), 0),
            (BigUint::from(u128::MAX).pow(2) - 1u8, 1),
            (BigUint::from(1u8), 2),
            (BigUint::from(200u8), 3),
            (BigUint::from(u128::MAX), 4),
        ];
        insertion_sort(&mut map);

        assert_eq!(map[0], (BigUint::from(1u8), 2));
        assert_eq!(map[1], (BigUint::from(200u8), 3));
        assert_eq!(map[2], (BigUint::from(u128::MAX), 4));
        assert_eq!(map[3], (BigUint::from(u128::MAX).pow(2) - 1u8, 1));
        assert_eq!(map[4], (BigUint::from(u128::MAX).pow(2), 0));
    }

    #[test]
    fn test_morton_decode_3d() {
        let morton_code = BigUint::from(0b10011011u128);
        let decoded_result = morton_decode_3d(morton_code);

        assert_eq!(decoded_result.0, BigUint::from(0b011u128));
        assert_eq!(decoded_result.1, BigUint::from(0b111u8));
        assert_eq!(decoded_result.2, BigUint::from(0b0u8));

        let morton_code = BigUint::from(0b101101010111010111u128);
        let decoded_result = morton_decode_3d(morton_code);
        assert_eq!(decoded_result.0, BigUint::from(0b110101u8));
        assert_eq!(decoded_result.1, BigUint::from(0b1111u8));
        assert_eq!(decoded_result.2, BigUint::from(0b110101u8));
    }

    #[test]
    fn test_morton_encode_decode_3d_u32_input() {
        let x = 0x2222u32;
        let y = 0x222222u32;
        let z = 0x88888888u32;

        println!("\nSPREAD U32...");
        let x_spread = spread(x.into());
        let y_spread = spread(y.into());
        let z_spread = spread(z.into());

        let morton_code = combine_spread(x_spread, y_spread, z_spread);
        println!(
            "combine_spread(x_spread, y_spread, z_spread): \n{:#0x}",
            morton_code
        );
        let decoded = morton_decode_3d(morton_code);
        assert_eq!(decoded.0, x.into());
        assert_eq!(decoded.1, y.into());
        assert_eq!(decoded.2, z.into());
    }

    #[test]
    fn test_morton_encode_decode_3d_u64_input() {
        let x = 0x2222222222222222u64;
        let y = 0x4444444444444444u64;
        let z = 0x8888888888888888u64;

        println!("\nSPREAD U64 ...");
        let x_spread = spread(x.into());
        let y_spread = spread(y.into());
        let z_spread = spread(z.into());

        let morton_code = combine_spread(x_spread, y_spread, z_spread);
        println!(
            "combine_spread(x_spread, y_spread, z_spread): \n{:#0x}",
            morton_code
        );
        let decoded = morton_decode_3d(morton_code);
        assert_eq!(decoded.0, x.into());
        assert_eq!(decoded.1, y.into());
        assert_eq!(decoded.2, z.into());

        let x = 0xFFFFFFFFFFFFFFFFu64;
        let y = u64::MAX - 0x4444444444444444u64;
        let z = u64::MAX - 0x8888888888888888u64;

        println!("\nSPREAD U64 big numbers ...");
        let x_spread = spread(x.into());
        let y_spread = spread(y.into());
        let z_spread = spread(z.into());

        let morton_code = combine_spread(x_spread, y_spread, z_spread);
        println!(
            "combine_spread(x_spread, y_spread, z_spread): \n{:#0x}",
            morton_code
        );
        let decoded = morton_decode_3d(morton_code);
        assert_eq!(decoded.0, x.into());
        assert_eq!(decoded.1, y.into());
        assert_eq!(decoded.2, z.into());
    }
    // #[test]
    // fn test_neighbors_z_morton_encode_application() {
    /*TODO: test and benchmark with BTreeMap
        use std::collections::BTreeMap;
    use itertools::Itertools;

        let mut atoms = Atoms::new(4);
        let new_positions = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0];
        let mut handles:BTreeMap<(u128,u128,u128),usize> = BTreeMap::new(); //(key=morton_code, value=index_of_atom)

        let new_positions_arr = Array2::from_shape_vec((3, 4), new_positions)
            .expect("Failed to create new positions array");
        atoms.positions.assign(&new_positions_arr);
        // let mut morton_codes: Vec<(u128, u128, u128)> = Vec::new();
        for (i,pos) in atoms.positions.axis_iter(Axis(1)).enumerate() {
            println!("Pos: {:?}", pos);
            println!(
                "morton code: {:?}",
                morton_encode_position(pos[0], pos[1], pos[2])
            );
            handles.insert(morton_encode_position(pos[0], pos[1], pos[2]),i);
        }
        println!("sorted handles: {:?}",handles);
        */
}

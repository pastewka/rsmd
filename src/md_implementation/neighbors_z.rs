// pub fn norm_and_scale(value: f32)-> u64{
//     println!("value in norm_and_scale: {}",value);
//     const MASK_NO_SIGN: u32= 0x7FFFFFFF;
//     // assert!(value.is_finite() && value<INT_MAX as f32 && value > INT_MIN as f32);

//     let unsigned_interpretation:u32 = value.to_bits();
//     println!("unsigned_interpretation: {}",unsigned_interpretation);
//     println!("ix: {:#032b}",unsigned_interpretation);

//     //extract the sign bit, which is -1i32 if the signed integer value is negative
//     let integer_sign= unsigned_interpretation as i32>>31;

//     //Treat integer value as positive value.
//     //If original value was negative, perform 2's complement negation. (flip the bits via xor and add +1)
//     //Then shift the range to make all values fit the u64 range.
//     //=> Preserves the order of the original floatingpoint value with f32::MIN at 0u64 aka u64::MIN, 0.0f32 at the middle of the u64 range and f32::MAX at the u64::MAX
//     let value_without_sign_bit = (unsigned_interpretation & MASK_NO_SIGN);
//     println!("value_without_sign_bit: {}",value_without_sign_bit);
//     let xored = (value_without_sign_bit as i32 ^integer_sign);
//     println!("xored: {}; {}",xored,xored as u32);
//     let mut scaled_positive = (xored - integer_sign)as u64;
//     println!("scaled_positive without add of MASK: {}",scaled_positive);
//     println!("MASK_NO_SIGN: {}",MASK_NO_SIGN);
//     // scaled_positive+= MASK_NO_SIGN as u64;
//     scaled_positive <<=1;
//     println!("scaled_positive: {}",scaled_positive);
//     return scaled_positive;
// }
use num::BigUint;
use num::One;
pub fn f64_to_u128_order_preserving(value: f64) -> u128 {
    let bits = value.to_bits(); // Get the raw bit pattern of the f32 (as u32)

    // If the number is negative, we flip all the bits to invert the order.
    // If the number is positive, we flip only the sign bit to maintain order.
    if bits & 0x8000000000000000 != 0 {
        return !bits as u128; // For negative numbers: invert all bits
    } else {
        return bits as u128 + 0x8000000000000000; // For positive numbers: flip the sign bit by adding 2^63
    }
}

fn spread(v: u128) -> BigUint {
    //bitmask ... 32 ones 64 zeros 32 ones
    let mut mask_1 = BigUint::from(0x0000000000000000ffffffff00000000_u128);
    mask_1 <<= 128;
    mask_1 += 0x00000000ffffffff0000000000000000_u128;
    mask_1 <<= 128;
    mask_1 += 0xffffffff0000000000000000ffffffff_u128;

    //bitmask 0b...11111111 11111111 00000000 00000000 00000000 00000000 11111111 11111111
    let mut mask_2 = BigUint::from(0x00000000ffff00000000ffff00000000_u128);
    mask_2 <<= 128;
    mask_2 += 0xffff00000000ffff00000000ffff0000_u128;
    mask_2 <<= 128;
    mask_2 += 0x0000ffff00000000ffff00000000ffff_u128;

    //bitmask 0b...11111111 00000000 00000000 11111111
    let mut mask_3 = BigUint::from(0x0ff0000ff00000ff0000ff0000ff0000_u128);
    mask_3 <<= 128;
    mask_3 += 0xff0000ff00000ff0000ff0000ff0000f_u128;
    mask_3 <<= 128;
    mask_3 += 0xf0000ff0000ff00000ff0000ff0000ff_u128;

    //bitmask 0b...1111000000001111000000001111
    let mut mask_4 = BigUint::from(0x00f00f00f00f00f00f00f00f00f00f00_u128);
    mask_4 <<= 128;
    mask_4 += 0xf00f00f00f00f00f00f00f00f00f00f0_u128;
    mask_4 <<= 128;
    mask_4 += 0x0f00f00f00f00f00f00f00f00f00f00f_u128;

    //bitmask 0b...11000011000011
    let mut mask_5 = BigUint::from(0x0c30c30c30c30c30c30c30c30c30c30c_u128);
    mask_5 <<= 128;
    mask_5 += 0x30c30c30c30c30c30c30c30c30c30c30_u128;
    mask_5 <<= 128;
    mask_5 += 0xc30c30c30c30c30c30c30c30c30c30c3_u128;

    //bitmask 0b...1001001001
    let mut mask_6 = BigUint::from(0x14924924924924924924924924924924_u128);
    mask_6 <<= 128;
    mask_6 += 0x92492492492492492492492492492492_u128;
    mask_6 <<= 128;
    mask_6 += 0x49249249249249249249249249249249_u128;

    println!(
        "Generated masks:\n1: {:#0x}\n2: {:#0x}\n3: {:#0x}\n4: {:#0x}\n5: {:#0x}\n6: {:#0x}",
        mask_1, mask_2, mask_3, mask_4, mask_5, mask_6
    );
    // let mut val
    let mut value = BigUint::from(v);
    println!(" value before masking: \n{:#0x}", value);
    let a = value.clone();
    println!(" value shifted: \n{:#0x}", a.clone() | (a << 64));
    let a = value.clone();
    value = (value | (a << 64));
    value &= mask_1;

    println!("value after mask 1: \n{:#0x}", value);
    println!("amount of bits of value: {}", value.bits());
    value = (value.clone() | (value << 32)) & mask_2;
    println!("value after mask 2: \n{:#0x}", value);
    println!("amount of bits of value: {}", value.bits());
    value = (value.clone() | (value << 16)) & mask_3;
    println!("value after mask 3: \n{:#0x}", value);
    value = (value.clone() | (value << 8)) & mask_4;
    println!("value after mask 4: \n{:#0x}", value);
    value = (value.clone() | (value << 4)) & mask_5;
    println!("value after mask 5: \n{:#0x}", value);
    value = (value.clone() | (value << 2)) & mask_6;
    println!("value after mask 6: \n{:#0x}", value);
    return value;
}

pub fn combine_spread(x_spread: BigUint, y_spread: BigUint, z_spread: BigUint) -> BigUint {
    // let mut morton_code = x_spread;
    // println!("morton code: 'spread(x)':\n {:#0x}", morton_code);

    // let mut m1 = BigUint::from(y_spread);
    // m1 <<= 1;
    // morton_code |= m1;
    // println!(
    //     "morton code: 'spread(x) | spread(y)<<1':\n {:#0x}",
    //     morton_code
    // );

    // let mut m2 = BigUint::from(z_spread);
    // m2 <<= 2;
    // morton_code |= m2;

    // println!(
    //     "morton code 'spread(x) | spread(y)<<1 | spread(z) << 2':\n {:#0x}",
    //     morton_code
    // );
    // assert_eq!(x_spread.bits(),190);
    // let x_y_spread:BigUint = x_spread.clone() | y_spread.clone() <<1;
    // assert_eq!(x_y_spread.bits(), x_spread.bits()+y_spread.bits());

    return x_spread | y_spread << 1 | z_spread << 2;
}

// pub fn morton_encode_position(x: f64, y: f64, z: f64) -> (u128, u128, u128) {
//     let a = spread(f64_to_u128_order_preserving(x));
//     let mut b = (spread(f64_to_u128_order_preserving(y)));
//     b = b << 1;
//     let mut c = (spread(f64_to_u128_order_preserving(z)));
//     c = c << 1;
//     c = c << 1;
//     return (a, b, c);
// }

pub fn insertion_sort(data: &mut Vec<((u128, u128, u128), usize)>) {
    for i in 1..data.len() {
        let mut j = i;
        let current = data[i];

        // Compare current key with the keys in the sorted portion (left of index i)
        // Shift elements to the right until correct position for `current`
        while j > 0 && data[j - 1].0 > current.0 {
            data[j] = data[j - 1]; // Shift element to the right
            j -= 1;
        }

        // Insert `current` at its correct position
        data[j] = current;
    }
}

#[cfg(test)]
mod tests {
    use super::{combine_spread, f64_to_u128_order_preserving, insertion_sort, spread};
    use num::BigUint;
    use num::One;
    use num::Zero;

    fn check_interleaved_by_two(spread_value: BigUint) -> bool {
        let amount_of_ones = spread_value.count_ones();
        let mut count_checked_ones: u64 = 0;

        let mut first_one_found = false;
        println!("bits in spread_value: {}", spread_value.bits());

        for i in 0..spread_value.bits() {
            if !first_one_found && spread_value.bit(i) {
                first_one_found = true;
                count_checked_ones += 1;
            }
            if first_one_found {
                if spread_value.bit(i) && i + 2 != spread_value.bits() {
                    count_checked_ones += 1;
                    if spread_value.bit(i + 1) || spread_value.bit(i + 2) {
                        return false;
                    }
                }
            }

            if count_checked_ones == amount_of_ones {
                return true;
            }
        }
        return false;
    }

    fn morton_decode_3d(morton_code: BigUint) -> (BigUint, BigUint, BigUint) {
        use ndarray::array;
        // bitmask 0b...1001001001
        let mut mask_6 = BigUint::from(0x14924924924924924924924924924924_u128);
        mask_6 <<= 128;
        mask_6 += 0x92492492492492492492492492492492_u128;
        mask_6 <<= 128;
        mask_6 += 0x49249249249249249249249249249249_u128;

        let mut b = BigUint::zero();
        let mut result_array = array![BigUint::zero(), BigUint::zero(), BigUint::zero()];
        for dimension in 0..3 {
            println!("dimension: {}", dimension);

            let mut modified_morton = morton_code.clone();
            modified_morton >>= dimension;

            while modified_morton.bits() >= 4 {
                println!(
                    "result before: {:b}; morton code: {:b}",
                    result_array[dimension], modified_morton
                );
                if modified_morton.bit(0) {
                    result_array[dimension] = (&result_array[dimension] << 1) | BigUint::one();
                } else {
                    result_array[dimension] = &result_array[dimension] << 1;
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
                "result after WHILE: {:b}; morton code: {:b}",
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
            result_array[i] = reversed_result;
        }
        return (
            result_array[0].clone(),
            result_array[1].clone(),
            result_array[2].clone(),
        );
    }

    #[test]
    fn test_check_interleaved_by_two() {
        let a = BigUint::from(0b100100100100100u32);
        assert!(a.bit(14));
        assert!(check_interleaved_by_two(BigUint::from(
            0b100100100100100u32
        )));
    }

    #[test]
    fn test_neighbors_z_norm_and_scale() {
        // assert_eq!(f32_to_u64_order_preserved(f32::MIN),0u64);
        // assert_eq!(f32_to_u64_order_preserved(f32::MAX), u64::MAX);
        println!("epsilon f64: {}", f64::EPSILON);
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
            //   assert!( ( (f1 >= j) && (norm_and_scale(f1) >= norm_and_scale(j)) ) ||
            //           ( (f1 <  j) && (norm_and_scale(f1) < norm_and_scale(j)) ));
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
            ((300, 2, 1), 0),
            ((300, 2, 0), 1),
            ((100, 111111, 223), 2),
            ((0, 2, 45), 3),
            ((0, 0, u128::MAX), 4),
        ];
        insertion_sort(&mut map);

        assert_eq!(map[0], ((0, 0, u128::MAX), 4));
        assert_eq!(map[1], ((0, 2, 45), 3));
        assert_eq!(map[2], ((100, 111111, 223), 2));
        assert_eq!(map[3], ((300, 2, 0), 1));
        assert_eq!(map[4], ((300, 2, 1), 0));
    }

    #[test]
    fn test_neighbors_z_morton_code_encode_distinct_positions() {
        //TODO: random values for positions and check if morton code is different, given different positions..
        //TODO: check if near positions have near morton code compared to more distant positions
    }

    fn morton_encode_21bits(data: u64) -> u64 {
        let mut x = data & 0x1fffff; //only first 21 bits
        println!(" 21bits.. before masking: \n{:#0x}", x);
        x = (x | x << 32) & 0x1f00000000ffff; // shift left 32 bits, OR with self, and         00011111000000000000000000000000000000001111111111111111
        println!(" 21bits.. x after mask 1: \n{:#0x}", x);
        x = (x | x << 16) & 0x1f0000ff0000ff; // shift left 32 bits, OR with self, and         00011111000000000000000011111111000000000000000011111111
        println!(" 21bits.. x after mask 2: \n{:#0x}", x);
        x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self,and 0001000000001111000000001111000000001111000000001111000000000000
        println!(" 21bits.. x after mask 3: \n{:#0x}", x);
        x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self,and 0001000011000011000011000011000011000011000011000011000100000000
        println!(" 21bits.. x after mask 4: \n{:#0x}", x);
        x = (x | x << 2) & 0x1249249249249249;
        println!(" 21bits.. x after mask 5: \n{:#0x}", x);
        return x;
    }

    #[test]
    fn test_neighbors_z_morton_code_demonstrator_3x_21bits() {
        println!(
            "--------------morton code demonstrator with 3x21bit uint values encoded to one 63 bit morton code--------------\n
            binary 0x10000                                 {:b}",
            0x10000
        );
        println!(
            "binary 0x10000 | 0x10000 <<1:                 {:b}",
            (0x10000 | 0x10000 << 1)
        );

        println!(
            "binary 0x10000 | 0x10000 <<1 | 0x10000 <<2:  {:b}",
            (0x10000 | 0x10000 << 1 | 0x10000 << 2)
        );
        /*
                output
        binary 0x10000                                 10000000000000000
        binary 0x10000 | 0x10000 <<1:                 110000000000000000
        binary 0x10000 | 0x10000 <<1 | 0x10000 <<2:  1110000000000000000

                 */
        let a = morton_encode_21bits(0x1fffff);
        assert!(false);

        println!(
            "morton code separately from 0x1fffff for x, y and z: \n x: {:b};\n y: {:b};\n z: {:b}",
            morton_encode_21bits(0x1fffff),
            morton_encode_21bits(0x1fffff),
            morton_encode_21bits(0x1fffff)
        );
        println!(
            "morton code combined: morton(x):                                        {:b}",
            morton_encode_21bits(0x1fffff)
        );
        println!(
            "morton code combined: morton(x)| morton(y) <<1:                        {:b}",
            morton_encode_21bits(0x1fffff) | morton_encode_21bits(0x1fffff) << 1
        );
        println!(
            "morton code combined: morton(x)| morton(y) <<1 | morton(z) << 2:      {:b}",
            morton_encode_21bits(0x1fffff)
                | morton_encode_21bits(0x1fffff) << 1
                | morton_encode_21bits(0x1fffff) << 2
        );

        assert_eq!(
            morton_encode_21bits(0x1fffff)
                | morton_encode_21bits(0x1fffff) << 1
                | morton_encode_21bits(0x1fffff) << 2,
            (2u64.pow(63) - 1)
        );

        /*output
        morton code separately from 0x1fffff for x, y and z:
         x: 1001001001001001001001001001001001001001001001001001001001001;
         y: 1001001001001001001001001001001001001001001001001001001001001;
         z: 1001001001001001001001001001001001001001001001001001001001001
        morton code combined: morton(x):                                        1001001001001001001001001001001001001001001001001001001001001
        morton code combined: morton(x)| morton(y) <<1:                        11011011011011011011011011011011011011011011011011011011011011
        morton code combined: morton(x)| morton(y) <<1 | morton(z) << 2:      111111111111111111111111111111111111111111111111111111111111111 */
    }

    #[test]
    fn test_combine_spread() {
        //test shifting with result of morton code being 0x8000...u128

        let x: u128 = 2u128.pow(127);
        println!("test shifting with result of morton code being 0x8000...u128");

        let morton_code = combine_spread(x.into(), x.into(), x.into());

        assert_eq!(
            morton_code,
            BigUint::from(2u8).pow(127) + BigUint::from(2u8).pow(128) + BigUint::from(2u8).pow(129)
        );
    }

    // fn revert_3d_combined_morton(morton_code: BigUint) -> (BigUint,BigUint,BigUint){
    //     let result = (BigUint::zero(),BigUint::zero(),BigUint::zero());

    //     for bit_index in 0..570{
    //         for dimension 0..3{
    //         result.dimension =
    //         }
    //     }

    // }

    #[test]
    fn test_neighbors_z_morton_code_demonstrator_3x_128bits() {
        //test with x,y,z being 2^127+2^126+2^125
        let x: u128 = 2u128.pow(127) + 2u128.pow(126) + 2u128.pow(125);

        let spread_x = spread(x);
        println!(
            "Spreaded 1D morton code from 2^127+2^126+2^125 is:\n{:#0x}\n\n",
            spread_x
        );
        const SHIFT_AMOUNT: u64 = 62;
        const X_SIZE: u64 = 128;
        assert_eq!(spread_x.bits(), X_SIZE + SHIFT_AMOUNT);

        assert!(check_interleaved_by_two(spread_x.clone()));

        let spread_z = BigUint::one();

        let morton_code = combine_spread(spread_x.clone(), spread_x.clone(), spread_z.clone());
        assert_eq!(
            morton_code.bits(),
            spread_x.bits() + spread_x.bits() + spread_z.bits() + SHIFT_AMOUNT * 3
        );
        println!("morton_code: {}", morton_code);

        // let (reverted_spread_x,reverted_spred_y,reverted_spread_z) = revert_3d_combined_morton(morton_code);
        // assert_eq!(morton_decode(morton_code),(x.into(),x.into(),BigUint::one()));

        //test with x,y,z being the maximum value of u128
        let x: u128 = u128::MAX; // 2^128-1
        let spread_x = spread(x);
        println!(
            "Spreaded 1D morton code from 2^128-1 for x, y and z: \n x: {:b};\n y: {:b};\n z: {:b}",
            spread_x, spread_x, spread_x
        );
        let morton_code = combine_spread(spread_x.clone(), spread_x.clone(), spread_x.clone());

        let target_morton = BigUint::from(2u8).pow(384) - BigUint::one();
        println!("target_morton: {}", target_morton);
        assert_eq!(morton_code, target_morton);
        assert!(false);
    }

    #[test]
    fn test_morton_decode_3d() {
        let morton_code = BigUint::from(0b10011011u128);
        let decoded_result = morton_decode_3d(morton_code);
        //     let mut bit_string_x = decoded_result.0.to_str_radix(2);

        // // Step 2: Reverse the binary string
        // bit_string_x = bit_string_x.chars().rev().collect();

        // // Step 3: Convert the reversed binary string back into a BigUint
        // let result_x = &bit_string_x.parse::<BigUint>().unwrap();
        // println!("RESULT x: {:b}; x_reversed: {:b}",decoded_result.0,result_x);
        // let decoded_x_reversed:u128 = decoded_result.0;
        assert_eq!(decoded_result.0, BigUint::from(0b011u128));
        assert_eq!(decoded_result.1, BigUint::from(0b111u8));
        assert_eq!(decoded_result.2, BigUint::from(0b0u8));

        let morton_code = BigUint::from(0b101101010111010111u128);
        let decoded_result = morton_decode_3d(morton_code);
        assert_eq!(decoded_result.0, BigUint::from(0b110101u8));
        assert_eq!(decoded_result.1, BigUint::from(0b1111u8));
        assert_eq!(decoded_result.2, BigUint::from(0b110101u8));
    }

    /*#[test]
    fn test_neighbors_z_morton_encode_application() {
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

        let mut atoms = Atoms::new(4);
        let new_positions = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0];
        let new_positions_arr = Array2::from_shape_vec((3, 4), new_positions)
            .expect("Failed to create new positions array");
        atoms.positions.assign(&new_positions_arr);

        let mut handles: Vec<((u128, u128, u128), usize)> = Vec::new();

        for (i, pos) in atoms.positions.axis_iter(Axis(1)).enumerate() {
            handles.push((morton_encode_position(pos[0], pos[1], pos[2]), i));
        }

        println!("handles unsorted: {:?}", handles);
        insertion_sort(&mut handles);
        println!("handles sorted: {:?}", handles);

        assert!(false);
    }*/
}

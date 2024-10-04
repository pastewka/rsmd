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

fn spread(v: u128) -> u128 {
    // let mut val
    let mut value = v;
    value = (value | (value << 64)) & 0x0000000000000000FFFFFFFFFFFFFFFF;
    value = (value | (value << 32)) & 0x00000000FFFFFFFF00000000FFFFFFFF;
    value = (value | (value << 16)) & 0x0000FFFF0000FFFF0000FFFF0000FFFF;
    value = (value | (value << 8)) & 0x00FF00FF00FF00FF00FF00FF00FF00FF;
    value = (value | (value << 4)) & 0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F;
    value = (value | (value << 2)) & 0x33333333333333333333333333333333;
    value = (value | (value << 1)) & 0x55555555555555555555555555555555;
    return value;
}

pub fn morton_encode_position(x: f64, y: f64, z: f64) -> (u128, (u8, u128), (u8, u128)) {
    let mut y_low_bytes: u128 = spread(f64_to_u128_order_preserving(y));
    let y_high_byte: u8 = (y_low_bytes & 0x80000000000000000000000000000000 >> 127)
        .try_into()
        .unwrap();
    println!("{:b}", y_low_bytes);

    y_low_bytes <<= 1;
    println!("{:b}", y_low_bytes);

    let mut z_low_bytes: u128 = spread(f64_to_u128_order_preserving(z));
    println!("z_low_bytes: {:b}", z_low_bytes);
    let z_high_byte: u8 = (z_low_bytes & 0xC0000000000000000000000000000000u128 >> 126)
        .try_into()
        .unwrap();

    println!("z_low_bytes: {:b}", z_low_bytes);
    z_low_bytes <<= 2;
    println!("z_low_bytes: {:b}", z_low_bytes);

    return (
        spread(f64_to_u128_order_preserving(x)),
        (y_high_byte, y_low_bytes),
        (z_high_byte, z_low_bytes),
    );
}

#[cfg(test)]
mod tests {
    use crate::md_implementation::atoms::Atoms;
    use crate::md_implementation::neighbors_z::{
        f64_to_u128_order_preserving, morton_encode_position,
    };
    use itertools::Itertools;
    use ndarray::{Array2, Axis};
    use std::vec::Vec;

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
    fn test_neighbors_z_morton_encode() {
        let mut atoms = Atoms::new(4);
        let new_positions = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0];

        let new_positions_arr = Array2::from_shape_vec((3, 4), new_positions)
            .expect("Failed to create new positions array");
        atoms.positions.assign(&new_positions_arr);
        let mut morton_codes: Vec<(u128, (u8, u128), (u8, u128))> = Vec::new();
        for pos in atoms.positions.axis_iter(Axis(1)) {
            println!("Pos: {:?}", pos);
            morton_codes.push(morton_encode_position(pos[0], pos[1], pos[2]));
        }

        println!("morton codes of the 4 atoms: {:?}", morton_codes);

        assert!(false);
    }
}

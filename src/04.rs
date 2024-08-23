use itertools::Interleave;
use ndarray::{s, Array, Array2, Array3, Axis, Array1};
use rsmd::md_implementation::{self, xyz};


use std::ops::BitXor;
use morton_encoding;

use mimalloc::MiMalloc;
use num::{Float, integer};
#[global_allocator]
//static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
static GLOBAL: MiMalloc = MiMalloc;

const NB_ITERATIONS: u32 = 100;
const SCREEN_INTERVAL: u32 = 1;
const FILE_INTERVAL: u32 = 1000;
const TIMESTEP: f64 = 0.0001;
const INPUT_FOLDER: &str = "input_files/";
const INPUT_FILE: &str = "lj54.xyz";
const OUTPUT_FILE: &str = "traj.xyz";

const CELL_SIZE: i32 = 10;
const INT_MAX: i32 = 512;
const INT_MIN: i32 = -512;

const ABSOLUTE_MAXIMUM_VALUE: f64 = 5000.0;



fn norm_and_scale(value: f32)-> i64{
    const MASK_NO_SIGN: u32= 0x7FFFFFFF;
    assert!(value.is_finite());

    let unsigned_interpretation:u32 = value.to_bits();

    //extract the sign bit, which is -1i32 if the signed integer value is negative
    let integer_sign= (unsigned_interpretation as i32>>31);
    println!("value in norm_and_scale: {}",value);
    println!("unsigned_interpretation: {}",unsigned_interpretation);
    println!("ix: {:032b}",unsigned_interpretation);
println!("integer_sign bit: {:032b}; i32: {}",integer_sign,integer_sign);

    let value_without_sign_bit = (unsigned_interpretation & MASK_NO_SIGN);
    println!("value_without_sign_bit: {:032b}",value_without_sign_bit);
    let mut xored = (value_without_sign_bit as i32 ^integer_sign);
    println!("xored: {}; as bits {:032b}",xored,xored);
    if integer_sign !=0{
        xored+= 1;
    }
    // let mut scaled_positive = (xored + integer_sign);
    println!("scaled_positive without add of MASK: {}; {:032b}",xored,xored);
    println!("MASK_NO_SIGN: {}; bits: {:032b}",MASK_NO_SIGN,MASK_NO_SIGN);
    println!("MASK_NO_SIGN as i32: {:064b}\nxored: {:064b}",MASK_NO_SIGN as i32,xored);
    let scaled_positive = xored as i64 + MASK_NO_SIGN as i64;
    // let scaled_positive = xored + MASK_NO_SIGN as i32;
    // scaled_positive <<=1;
    println!("scaled_positive: {}; bits: {:064b}",scaled_positive,scaled_positive);

    //Treat integer value as positive value.
    //If original value was negative, perform 2's complement negation. (flip the bits via xor and add +1)
    //Then shift the range to make all values fit the u64 range.
    //=> Preserves the order of the original floatingpoint value with f32::MIN at 0u64 aka u64::MIN, 0.0f32 at the middle of the u64 range and f32::MAX at the u64::MAX 
    return ((unsigned_interpretation&MASK_NO_SIGN) as i32 ^ integer_sign) as i64 + MASK_NO_SIGN as i64;
}

use ndarray::ArrayView1;

fn convert_f64_to_f32(value:f64)-> f32{
    if value > f32::MAX as f64{
        return f32::MAX;
    }else if (value< f32::MIN as f64){
        return f32::MIN;
    }else{
        return value as f32;
    }
}

fn interleave_bits(mut uint_value:u32) -> u32{
    println!("(uint_value | (uint_value<<16)): {}",(uint_value | (uint_value<<16)));
    let v = (uint_value | (uint_value<<16)) as u64 & 0xff00ffu64;
    println!("v: {}",v);
    return 0;
}
 fn morton_encode_pos(position: ArrayView1<f64>)-> u32{
    let x_norm = norm_and_scale(f32::MIN);
    let y_norm = norm_and_scale(f32::MAX);
    // let x_norm = norm_and_scale(position[0] as f32);
    //let y_norm= norm_and_scale(convert_f64_to_f32(position[1]));
    //let z_norm= norm_and_scale(convert_f64_to_f32(position[2]));

    println!("x_norm: {}",x_norm );
    println!("x_norm | (x_norm << 16): {}",x_norm | (x_norm << 16));
    

return 0;



// fn hash(x:f32,y:f32,z:f32)-> u64{
// const HASH_TABLE_SIZE: i32 = 4000;
//     const PRIME_NUMBER_1: i32 = 73856093;
//     const PRIME_NUMBER_2: i32 = 19349663;
//     const PRIME_NUMBER_3: i32 = 83492791;
//     return (((x/CELL_SIZE as f32).floor() as i32*PRIME_NUMBER_1).bitxor((y/CELL_SIZE as f32).floor() as i32*PRIME_NUMBER_2).bitxor((z/CELL_SIZE as f32).floor() as i32*PRIME_NUMBER_3)%HASH_TABLE_SIZE) as u64;
// }

}



fn main() {
    // let mut img = Array3::<u8>::zeros((10, 10, 2));
    // println!("before img: {:?}", img);
    // let arr = Array::ones(2);
    // 
    // println!("before arr: {:?}", arr);
    // img.slice_mut(s![4..6usize, .., ..]).assign(&arr);
    // 
    // println!("after img: {:?}", img);
    // 


    let mut atoms = md_implementation::xyz::read_xyz_with_velocities(INPUT_FOLDER.to_owned() + INPUT_FILE)
            .unwrap();
    println!(
        "atom configuration loaded with {} atoms",
        atoms.positions.ncols()
    );
    for (p_index,p) in atoms.positions.axis_iter(Axis(1)).enumerate(){

    let code: u32 = morton_encode_pos(p);


    println!("position {}: {:?} => code {}",p_index,p,code);
    return;
    }
    //delete old trajectory if it exists
    _ = std::fs::remove_file(OUTPUT_FILE);

    let ekin: f64 = atoms.kinetic_energy();
    let epot: f64 = atoms.lj_direct_summation(None, None);

    for i in 0..NB_ITERATIONS {
        atoms.verlet_step1(TIMESTEP.into());

        let epot: f64 = atoms.lj_direct_summation(None, None);

        atoms.verlet_step2(TIMESTEP.into());

        if i % SCREEN_INTERVAL == 0 {
            let ekin: f64 = atoms.kinetic_energy();
            println!(
                "{:>20}{:>20}{:>20}{:>20}{:>20}{:>20}",
                i,
                i as f64 * TIMESTEP,
                ekin,
                epot,
                ekin + epot,
                ekin / (1.5 * atoms.positions.shape()[1] as f64)
            );
        }

        if i % FILE_INTERVAL == 0 {
            xyz::write_xyz(OUTPUT_FILE.to_string(), atoms.clone()).unwrap();
        }
    }
}

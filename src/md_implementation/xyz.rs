use super::atoms::Atoms;
use std::fs;
use std::io;
use std::io::prelude::*;
use std::io::BufWriter;

pub fn read_xyz(file_path: String) -> Result<Atoms, io::Error> {
    let file = fs::File::open(file_path)?;
    let mut reader = io::BufReader::new(file);

    // First line has number of atoms
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let nb_atoms: u32 = line.strip_suffix("\n").unwrap().parse().unwrap();

    // In extended XYZ this contains cell information, we ignore it for now
    reader.read_line(&mut line)?;

    // Create empty atoms container
    let mut atoms_arr = Atoms::new(usize::try_from(nb_atoms).unwrap());

    let mut x_vec = Vec::new();
    let mut y_vec = Vec::new();
    let mut z_vec = Vec::new();
    // Loop over all atoms and parse
    for _ in 0..nb_atoms {
        line.clear();
        reader.read_line(&mut line)?;
        let s: Vec<&str> = line.split_whitespace().collect();
        let mut x: f64 = s.get(1).unwrap().parse().unwrap();
        let mut y: f64 = s.get(2).unwrap().parse().unwrap();
        let mut z: f64 = s.get(3).unwrap().parse().unwrap();

        x_vec.push(x);
        y_vec.push(y);
        z_vec.push(z);
    }
    atoms_arr.push_pos_vec(x_vec, y_vec, z_vec);

    Ok(atoms_arr)
}

pub fn write_xyz(file_path: String, atoms: Atoms) -> Result<Atoms, io::Error> {
    let file = fs::File::create(file_path)?;
    let mut file = BufWriter::new(&file);
    let nb_atoms: usize = atoms.positions.shape()[1];
    file.write(nb_atoms.to_string().as_bytes())?;
    file.write("\n".to_string().as_bytes())?;

    for i in 0..nb_atoms {
        file.write("\nAu ".as_bytes())?;
        for j in 0..atoms.positions.shape()[0] {
            write!(&mut file, "{:10} ", atoms.positions[[j, i]])?;
        }
        for j in 0..atoms.velocities.shape()[0] {
            write!(&mut file, "{:10} ", atoms.velocities[[j, i]])?;
        }
    }
    return Ok(atoms);
}

pub fn read_xyz_with_velocities(file_path: String) -> Result<Atoms, io::Error> {
    let file = fs::File::open(file_path)?;
    let mut reader = io::BufReader::new(file);

    // First line has number of atoms
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let nb_atoms: u32 = line.strip_suffix("\n").unwrap().parse().unwrap();

    // In extended XYZ this contains cell information, we ignore it for now
    reader.read_line(&mut line)?;

    // Create empty atoms container
    let mut atoms_arr = Atoms::new(usize::try_from(nb_atoms).unwrap());

    let mut x_vec = Vec::new();
    let mut y_vec = Vec::new();
    let mut z_vec = Vec::new();
    let mut vx_vec = Vec::new();
    let mut vy_vec = Vec::new();
    let mut vz_vec = Vec::new();

    // Loop over all atoms and parse
    for _ in 0..nb_atoms {
        line.clear();
        reader.read_line(&mut line)?;
        let s: Vec<&str> = line.split_whitespace().collect();
        let mut x: f64 = s.get(1).unwrap().parse().unwrap();
        let mut y: f64 = s.get(2).unwrap().parse().unwrap();
        let mut z: f64 = s.get(3).unwrap().parse().unwrap();
        let mut vx: f64 = s.get(4).unwrap().parse().unwrap();
        let mut vy: f64 = s.get(5).unwrap().parse().unwrap();
        let mut vz: f64 = s.get(6).unwrap().parse().unwrap();

        x_vec.push(x);
        y_vec.push(y);
        z_vec.push(z);
        vx_vec.push(vx);
        vy_vec.push(vy);
        vz_vec.push(vz);
    }
    atoms_arr.push_pos_velo_vec(x_vec, y_vec, z_vec, vx_vec, vy_vec, vz_vec);

    Ok(atoms_arr)
}
#[cfg(test)]
mod tests {
    use crate::md_implementation::{xyz, xyz::Atoms};
    use ndarray::Array;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use std::fs;

    #[test]
    fn test_read_xyz() {
        let mut atoms = xyz::read_xyz("cluster_3871.xyz".to_string()).unwrap();

        assert_eq!(3871, atoms.positions.shape()[1]);
        assert_eq!(3871, atoms.velocities.shape()[1]);
        assert_eq!(3871, atoms.forces.shape()[1]);
        assert_eq!(3871, atoms.masses.len());

        assert_eq!(3, atoms.positions.shape()[0]);
        assert_eq!(3, atoms.velocities.shape()[0]);
        assert_eq!(3, atoms.forces.shape()[0]);
        assert!(
            23.3401 == atoms.positions[[0, 0]]
                && 23.3401 == atoms.positions[[1, 0]]
                && 23.3401 == atoms.positions[[2, 0]]
        );
        assert_eq!(25.67411, atoms.positions[[0, 1]]);
        assert_eq!(23.3401, atoms.positions[[1, 1]]);
        assert_eq!(24.7826, atoms.positions[[2, 1]]);
        //Au       9.33600000      19.01261000      38.99710000     2200        1
        assert_eq!(9.336, atoms.positions[[0, 2199]]);
        assert_eq!(19.01261, atoms.positions[[1, 2199]]);
        assert_eq!(38.9971, atoms.positions[[2, 2199]]);
        //Au      23.34010000       7.13210000       4.66800000     3871        1
        assert_eq!(23.3401, atoms.positions[[0, 3870]]);
        assert_eq!(7.1321, atoms.positions[[1, 3870]]);
        assert_eq!(4.668, atoms.positions[[2, 3870]]);
    }
    #[test]
    fn test_read_xyz_with_velocities() {
        let mut atoms = xyz::read_xyz_with_velocities("lj54InclVelocity.xyz".to_string()).unwrap();

        assert_eq!(54, atoms.positions.shape()[1]);
        assert_eq!(54, atoms.velocities.shape()[1]);
        assert_eq!(54, atoms.forces.shape()[1]);
        assert_eq!(54, atoms.masses.len());

        assert_eq!(3, atoms.positions.shape()[0]);
        assert_eq!(3, atoms.velocities.shape()[0]);
        assert_eq!(3, atoms.forces.shape()[0]);
        assert!(
            -0.689712 == atoms.positions[[0, 0]]
                && 1.51676 == atoms.positions[[1, 0]]
                && 2.02485 == atoms.positions[[2, 0]]
        );

        assert_eq!(1.77012, atoms.velocities[[0, 0]]);
        assert_eq!(0.43038, atoms.velocities[[1, 0]]);
        assert_eq!(-0.045422, atoms.velocities[[2, 0]]);

        assert_eq!(3.0363, atoms.positions[[0, 1]]);
        assert_eq!(1.06446, atoms.positions[[1, 1]]);
        assert_eq!(2.9303, atoms.positions[[2, 1]]);

        assert_eq!(-0.572017, atoms.velocities[[0, 1]]);
        assert_eq!(1.25828, atoms.velocities[[1, 1]]);
        assert_eq!(0.668012, atoms.velocities[[2, 1]]);

        assert_eq!(-0.0238165, atoms.positions[[0, 15]]);
        assert_eq!(1.51979, atoms.positions[[1, 15]]);
        assert_eq!(3.7445, atoms.positions[[2, 15]]);

        assert_eq!(-0.289378, atoms.velocities[[0, 15]]);
        assert_eq!(2.01253, atoms.velocities[[1, 15]]);
        assert_eq!(1.53809, atoms.velocities[[2, 15]]);

        assert_eq!(0.399391, atoms.positions[[0, 53]]);
        assert_eq!(-0.106678, atoms.positions[[1, 53]]);
        assert_eq!(1.96548, atoms.positions[[2, 53]]);

        assert_eq!(1.01327, atoms.velocities[[0, 53]]);
        assert_eq!(-1.0644, atoms.velocities[[1, 53]]);
        assert_eq!(-0.811473, atoms.velocities[[2, 53]]);
    }
    fn get_type_of<T>(_: &T) -> &'static str {
        return std::any::type_name::<T>();
    }

    #[test]
    fn test_write_read_xyz() {
        let nb_atoms = 300;
        let mut atoms = Atoms::new(usize::try_from(nb_atoms).unwrap());

        atoms.forces = Array::random((3, nb_atoms), Uniform::new(-1.0, 1.0));
        atoms.positions = Array::random((3, nb_atoms), Uniform::new(-1.0, 1.0));
        atoms.velocities = Array::random((3, nb_atoms), Uniform::new(-1.0, 1.0));
        atoms.masses = Array::ones(nb_atoms);

        let traj_path = "tmp_rand_traj_test_write_read_xyz.xyz".to_string();
        xyz::write_xyz(traj_path.clone(), atoms.clone()).unwrap();
        let atoms_read = xyz::read_xyz_with_velocities(traj_path.clone()).unwrap();

        for i in 0..atoms.positions.shape()[1] {
            for j in 0..atoms.positions.shape()[0] {
                assert_eq!(
                    atoms.positions[[j, i]].clone(),
                    atoms_read.positions[[j, i]].clone()
                );
                assert_eq!(
                    atoms.velocities[[j, i]].clone(),
                    atoms_read.velocities[[j, i]].clone()
                );
            }
        }
        _ = fs::remove_file(traj_path);
    }
}

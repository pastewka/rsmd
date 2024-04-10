use crate::atoms::Atoms;
use std::fs;
use std::io;
use std::io::prelude::*;

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
    let mut m_vec: Vec<f64> = vec![1.0; nb_atoms.try_into().unwrap()];
    let mut vx_vec: Vec<f64> = Vec::with_capacity(nb_atoms.try_into().unwrap());
    for _ in 0..m_vec.len() {
        m_vec.push(1.0);
        vx_vec.push(0.0);
    }
    let vy_vec = vx_vec.clone();
    let vz_vec = vx_vec.clone();
    atoms_arr.push_vec(m_vec, x_vec, y_vec, z_vec, vx_vec, vy_vec, vz_vec);

    Ok(atoms_arr)
}

#[cfg(test)]
mod tests {
    use crate::xyz;
    #[test]
    fn test_read_xyz() {
        let mut atoms = xyz::read_xyz("cluster_3871.xyz".to_string()).unwrap();

        assert_eq!(3871, atoms.positions.shape()[1]);
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
}

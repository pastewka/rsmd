use std::io::prelude::*;
use std::io;
use std::fs;

use crate::atoms::Atoms;

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
    let mut atoms = Atoms::empty();

    // Loop over all atoms and parse
    for _ in 1..nb_atoms {
        line.clear();
        reader.read_line(&mut line)?;
        let s: Vec<&str> = line.split_whitespace().collect();
        //let element = &s[0];
        let x: f64 = s.get(1).unwrap().parse().unwrap();
        let y: f64 = s.get(2).unwrap().parse().unwrap();
        let z: f64 = s.get(3).unwrap().parse().unwrap();
        atoms.push(1.0, x, y, z, 0.0, 0.0, 0.0);
    }

    Ok(atoms)
}

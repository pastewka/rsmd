use ndarray::{Array1, Array2, ArrayView, Axis};

pub struct Atoms {
    masses: Array1<f64>,
    positions: Array2<f64>,
    velocities: Array2<f64>,
}

impl Atoms {
    pub fn empty() -> Atoms {
        Atoms {
            masses: Array1::<f64>::zeros(0),
            positions: Array2::<f64>::zeros((0, 3)),
            velocities: Array2::<f64>::zeros((0, 3))
        }
    }

    pub fn push(&mut self, m: f64, x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64) {
        self.masses.append(Axis(0), ArrayView::from(&[m])).unwrap();
        self.positions.push_row(ArrayView::from(&[x, y, z])).unwrap();
        self.velocities.push_row(ArrayView::from(&[vx, vy, vz])).unwrap();
    }
}

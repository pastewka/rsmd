use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use ndarray::Zip;
use ndarray::{iter::Axes, Array, Array1, Axis};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use rsmd::md_implementation::{self, atoms::Atoms};

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn ndarray_iterations(c: &mut Criterion) {
    let mut group_verlet_loop = c.benchmark_group("ndarray_iteration_verlet_loop");

    let start = 5;
    let step = 50;
    let num_arr_elements = 10;

    let nb_atoms_arr: Vec<usize> = (0..num_arr_elements).map(|i| start + i * step).collect();

    for i in nb_atoms_arr.iter() {
        group_verlet_loop.bench_function(BenchmarkId::new("loop_elementwise", i), |b| {
            b.iter_batched_ref(
                || -> Atoms { init_atoms(*i) },
                |v| black_box(loop_over_ndarrays_elementwise(v)),
                BatchSize::SmallInput,
            )
        });

        group_verlet_loop.bench_function(BenchmarkId::new("loop_rowwise_scaled_add", i), |b| {
            b.iter_batched_ref(
                || -> Atoms { init_atoms(*i) },
                |v| black_box(loop_over_ndarrays_rowwise_scaled_add(v)),
                BatchSize::SmallInput,
            )
        });

        group_verlet_loop.bench_function(BenchmarkId::new("loop_zip_rowwise", i), |b| {
            b.iter_batched_ref(
                || -> Atoms { init_atoms(*i) },
                |v| black_box(loop_zip_over_ndarrays_rowwise(v)),
                BatchSize::SmallInput,
            )
        });
    }
    group_verlet_loop.finish();
}

fn init_atoms(nb_atoms: usize) -> Atoms {
    let mut atoms = Atoms::new(usize::try_from(nb_atoms).unwrap());
    atoms.forces = Array::random((3, nb_atoms), Uniform::new(-1.0, 1.0));
    atoms.positions = Array::random((3, nb_atoms), Uniform::new(-1.0, 1.0));
    atoms.velocities = Array::random((3, nb_atoms), Uniform::new(-1.0, 1.0));
    atoms.masses = Array::ones(nb_atoms);
    return atoms;
}

fn loop_over_ndarrays_elementwise(atoms: &mut Atoms) {
    for i in 0..atoms.positions.shape()[0] {
        for j in 0..atoms.positions.shape()[1] {
            atoms.positions[[i, j]] += atoms.velocities[[i, j]] * 0.0001;
        }
    }
}

fn loop_over_ndarrays_rowwise_scaled_add(atoms: &mut Atoms) {
    for (i, mut row) in atoms.positions.rows_mut().into_iter().enumerate() {
        row.scaled_add(0.0001, &atoms.velocities.row(i));
    }
}

fn loop_zip_over_ndarrays_rowwise(atoms: &mut Atoms) {
    Zip::from(atoms.positions.rows_mut())
        .and(atoms.velocities.rows())
        .for_each(|mut position, velocity| {
            position.scaled_add(0.0001, &velocity);
        });
}

criterion_group!(
    benches,
    ndarray_iterations //, colwise_iteration,rowwise_iteration
);
criterion_main!(benches);

#[cfg(test)]
mod tests {
    use crate::md_implementation::atoms::Atoms;
    use googletest::{matchers::near, verify_that};

    #[test]
    fn test_iteration_validity() {}
}

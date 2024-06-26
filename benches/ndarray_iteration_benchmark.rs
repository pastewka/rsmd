use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use ndarray::Array;
use ndarray::Zip;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use rsmd::md_implementation::atoms::Atoms;

use mimalloc::MiMalloc;
#[global_allocator]
//static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
static GLOBAL: MiMalloc = MiMalloc;

fn iterations_valid() -> bool {
    let mut atoms_vec: Vec<Atoms> = vec![init_atoms(10); 5];
    loop_over_ndarrays_elementwise(&mut atoms_vec[0]);
    loop_over_ndarrays_rowwise_scaled_add(&mut atoms_vec[1]);
    loop_over_ndarrays_rowwise(&mut atoms_vec[2]);
    loop_zip_over_ndarrays_rowwise(&mut atoms_vec[3]);
    loop_zip(&mut atoms_vec[4]);
    for a in atoms_vec.iter() {
        for i in 0..a.positions.shape()[0] {
            for j in 0..a.positions.shape()[1] {
                if a.positions[[i, j]] != atoms_vec[0].positions[[i, j]]
                    || a.velocities[[i, j]] != atoms_vec[0].velocities[[i, j]]
                    || a.forces[[i, j]] != atoms_vec[0].forces[[i, j]]
                    || a.masses[i] != atoms_vec[0].masses[i]
                {
                    return false;
                }
            }
        }
    }
    return true;
}

fn ndarray_iterations(c: &mut Criterion) {
    let mut group_verlet_loop = c.benchmark_group("ndarray_iteration_verlet_loop");

    let start = 5;
    let step = 50;
    let num_arr_elements = 10;

    let nb_atoms_arr: Vec<usize> = (0..num_arr_elements).map(|i| start + i * step).collect();

    assert!(iterations_valid());

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

        group_verlet_loop.bench_function(BenchmarkId::new("loop_rowwise", i), |b| {
            b.iter_batched_ref(
                || -> Atoms { init_atoms(*i) },
                |v| black_box(loop_over_ndarrays_rowwise(v)),
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

        group_verlet_loop.bench_function(BenchmarkId::new("loop_zip", i), |b| {
            b.iter_batched_ref(
                || -> Atoms { init_atoms(*i) },
                |v| black_box(loop_zip(v)),
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
fn loop_over_ndarrays_rowwise(atoms: &mut Atoms) {
    for (i, mut row) in atoms.positions.rows_mut().into_iter().enumerate() {
        for (position, &velocity) in row.iter_mut().zip(atoms.velocities.row(i).iter()) {
            *position += velocity * 0.0001;
        }
    }
}

fn loop_zip_over_ndarrays_rowwise(atoms: &mut Atoms) {
    Zip::from(atoms.positions.rows_mut())
        .and(atoms.velocities.rows())
        .for_each(|mut position, velocity| {
            position.scaled_add(0.0001, &velocity);
        });
}
fn loop_zip(atoms: &mut Atoms) {
    Zip::from(&mut atoms.positions)
        .and(&atoms.velocities)
        .for_each(|position, &velocity| {
            *position += velocity * 0.0001;
        });
}

criterion_group!(
    benches,
    ndarray_iterations //, colwise_iteration,rowwise_iteration
);
criterion_main!(benches);

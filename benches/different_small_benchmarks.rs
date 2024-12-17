use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Zip;
use ndarray::{Array, Array1, Array2, Axis};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

use mimalloc::MiMalloc;
#[global_allocator]
//static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
static GLOBAL: MiMalloc = MiMalloc;

//Result: mapv is definately better than map and collect
//        'zip and forech max' is a little bit better than 'for iter zip std::max'
//        atom_to_cell is both roughly the same

fn map_benchmarks(c: &mut Criterion) {
    let mut group_map_benchmarks = c.benchmark_group("compare_different_mappings");

    //generate random input
    let lengths = Array::random(3, Uniform::new(-1.0, 1.0));

    group_map_benchmarks.bench_with_input(
        BenchmarkId::new("normal_map_collect", &lengths[0]),
        &lengths,
        |b, l| {
            b.iter(|| normal_map_collect(&l));
        },
    );

    group_map_benchmarks.bench_with_input(
        BenchmarkId::new("mapv", &lengths[0]),
        &lengths,
        |b, l| {
            b.iter(|| mapv_function(&l));
        },
    );

    group_map_benchmarks.finish();
}

fn zip_benchmarks(c: &mut Criterion) {
    let mut group_zip_benchmarks = c.benchmark_group("compare_different_zip_and_max");

    //generate random input
    let l_by_cutoffs = Array1::from(vec![1, 2, 3, 4, 5]);
    let mut nb_grid_points = Array1::from(vec![10, 20, 30, 40, 50]);

    let nb_grid_points_backup = nb_grid_points.clone();

    group_zip_benchmarks.bench_with_input(
        BenchmarkId::new("zip_and_for_each_max", &l_by_cutoffs[0]),
        &l_by_cutoffs,
        |b, l| {
            b.iter(|| {
                //reset nb_grid_points
                nb_grid_points.assign(&nb_grid_points_backup);

                zip_and_for_each_max(l, &mut nb_grid_points);
            });
        },
    );

    group_zip_benchmarks.bench_with_input(
        BenchmarkId::new("for_iter_zip_max_std_max", &l_by_cutoffs[0]),
        &l_by_cutoffs,
        |b, l| {
            b.iter(|| {
                //reset nb_grid_points
                nb_grid_points.assign(&nb_grid_points_backup);

                for_iter_zip_max_std_max(l, &mut nb_grid_points);
            });
        },
    );

    group_zip_benchmarks.finish();
}

fn atom_to_cell_benchmarks(c: &mut Criterion) {
    let mut group_atom_to_cell_benchmarks = c.benchmark_group("compare_atom_to_cell_calculations");

    //generate random input
    let r = Array2::random((3, 1000), Uniform::new(-1.0, 1.0));
    let nb_grid_points = Array1::from(vec![10, 20, 30, 40, 50]);

    group_atom_to_cell_benchmarks.bench_with_input(
        BenchmarkId::new("atom_to_cell_status_quo", &nb_grid_points[0]),
        &nb_grid_points,
        |b, nb| {
            b.iter(|| {
                atom_to_cell_status_quo(&r, nb);
            });
        },
    );

    group_atom_to_cell_benchmarks.bench_with_input(
        BenchmarkId::new("atom_to_cell_axis_iter", &nb_grid_points[0]),
        &nb_grid_points,
        |b, nb| {
            b.iter(|| {
                atom_to_cell_axis_iter(&r, nb);
            });
        },
    );

    group_atom_to_cell_benchmarks.finish();
}

fn normal_map_collect(lengths: &Array1<f64>) {
    let cutoff = 2.0;
    let l_by_cutoffs: Array1<i32> = (lengths / cutoff)
        .iter()
        .map(|&l_by_cutoff| l_by_cutoff.ceil() as i32)
        .collect();
}

fn mapv_function(lengths: &Array1<f64>) {
    let cutoff = 2.0;
    let l_by_cutoffs: Array1<i32> = lengths.mapv(|l| (l / cutoff).ceil() as i32);
}

fn for_iter_zip_max_std_max(l_by_cutoffs: &Array1<i32>, nb_grid_points: &mut Array1<i32>) {
    for (l_by_cutoff, nb_grid_point) in l_by_cutoffs.iter().zip(nb_grid_points.iter_mut()) {
        *nb_grid_point = std::cmp::max(*l_by_cutoff, 1);
    }
}

fn zip_and_for_each_max(l_by_cutoffs: &Array1<i32>, nb_grid_points: &mut Array1<i32>) {
    Zip::from(nb_grid_points)
        .and(l_by_cutoffs)
        .for_each(|nb_grid_point, &l_by_cutoff| {
            *nb_grid_point = l_by_cutoff.max(1);
        });
}

fn coordinate_to_index(x: i32, y: i32, z: i32, nb_grid_points: &Array1<i32>) -> i32 {
    return x + nb_grid_points[0] * (y + nb_grid_points[1] * z);
}

fn atom_to_cell_status_quo(r: &Array2<f64>, nb_grid_points: &Array1<i32>) {
    let r = r.mapv(|i| i.floor() as i32);
    let atom_to_cell: Array1<i32> = r
        .axis_iter(Axis(1))
        .map(|r_col| coordinate_to_index(r_col[0], r_col[1], r_col[2], &nb_grid_points))
        .collect();
}

fn atom_to_cell_axis_iter(r: &Array2<f64>, nb_grid_points: &Array1<i32>) {
    let r = r.mapv(|i| i.floor() as i32);

    let atom_to_cell: Array1<i32> = Array1::from_iter(
        r.axis_iter(Axis(1))
            .map(|r_col| coordinate_to_index(r_col[0], r_col[1], r_col[2], &nb_grid_points)),
    );
}

criterion_group!(
    benches,
    atom_to_cell_benchmarks,
    map_benchmarks,
    zip_benchmarks
);
criterion_main!(benches);

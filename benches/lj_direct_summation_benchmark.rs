use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rsmd::md_implementation::{self, atoms::Atoms};
use std::fs;

fn criterion_benchmark(c: &mut Criterion) {
    const EPSILON: f64 = 0.7;
    const SIGMA: f64 = 0.3;

    let input_files = &[
        "lj_cube_8.xyz",
        "lj_cube_27.xyz",
        "lj_cube_64.xyz",
        "lj_cube_125.xyz",
        "lj_cube_216.xyz",
        "lj_cube_343.xyz",
        "lj_cube_512.xyz",
        "lj_cube_729.xyz",
        "lj_cube_1000.xyz",
    ];

    for path in input_files {
        if !fs::metadata(path).is_ok() {
            panic!("input file \"{}\" doesn't exist!", path);
        }
    }

    let mut group = c.benchmark_group("different_sized_lj_clusters");
    for i in 0..input_files.len() {
        group.bench_function(
            BenchmarkId::new(
                "lj_direct_summation",
                "input_file_".to_owned() + &input_files[i].to_string(),
            ),
            |b| {
                b.iter_batched_ref(
                    || -> Atoms {
                        md_implementation::xyz::read_xyz(input_files[i].to_string()).unwrap()
                    },
                    |v| black_box(v.lj_direct_summation(Some(EPSILON), Some(SIGMA))),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
